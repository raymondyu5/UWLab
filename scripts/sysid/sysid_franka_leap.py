"""
System Identification for Franka LEAP arm using CMA-ES (Open-Loop Joint Position Replay).

Uses the manager-based env with FrankaLeapJointPositionAction. Arm-only sysid (7 joints).
ImplicitActuatorCfg, no delay. 35 params: armature*7, static_friction*7, dynamic_ratio*7,
viscous_friction*7, encoder_bias*7.

Expected real_data .pt format (convert from zarr/npy using a script):
  arm_joint_pos: (T, 7) - real achieved arm joint positions
  arm_joint_pos_target: (T, 7) - commanded arm joint positions
  hand_actions: (T, 16) - commanded hand joint positions
  initial_arm_joint_pos: (7,) - initial arm config
  initial_hand_joint_pos: (16,) - initial hand config
  dt: float - control period in seconds

Usage:
    python scripts/sysid/sysid_franka_leap.py --headless --num_envs 512 \
        --real_data sysid_data_franka_leap.pt --max_iter 200
"""

import argparse
import os
import time
import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka LEAP Arm System Identification via CMA-ES")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--real_data", type=str, required=True)
parser.add_argument("--max_iter", type=int, default=200)
parser.add_argument("--sigma", type=float, default=0.3)
parser.add_argument("--output_dir", type=str, default="logs/sysid")
parser.add_argument("--save_interval", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--settle_steps", type=int, default=30)
# Parameter bounds
parser.add_argument("--armature_min", type=float, default=0.0)
parser.add_argument("--armature_max", type=float, default=10.0)
parser.add_argument("--friction_min", type=float, default=0.0)
parser.add_argument("--friction_max", type=float, default=20.0)
parser.add_argument("--viscous_friction_min", type=float, default=0.0)
parser.add_argument("--viscous_friction_max", type=float, default=20.0)
parser.add_argument("--bias_min", type=float, default=-0.1)
parser.add_argument("--bias_max", type=float, default=0.1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.assets import Articulation

import uwlab_tasks  # noqa: F401  # register gym envs

from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.sysid_cfg import FrankaLeapSysidEnvCfg

# Franka arm: 7 joints
NUM_ARM_JOINTS = 7
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
NUM_HAND_JOINTS = 16


# ============================================================================
# CMA-ES Optimizer
# ============================================================================

class CMAES:
    """Lightweight CMA-ES wrapper using the cmaes library."""

    def __init__(self, num_params, population_size, sigma=0.3, bounds=None):
        from cmaes import CMA
        self.num_params = num_params
        self.population_size = population_size
        self.bounds = np.array(bounds)
        self.optimizer = CMA(
            mean=np.full(num_params, 0.5),
            sigma=sigma,
            population_size=population_size,
            bounds=np.column_stack([np.zeros(num_params), np.ones(num_params)]),
        )
        self._solutions = None

    def ask(self) -> np.ndarray:
        self._solutions = []
        for _ in range(self.population_size):
            self._solutions.append(self.optimizer.ask())
        normalized = np.array(self._solutions)
        return self.bounds[:, 0] + normalized * (self.bounds[:, 1] - self.bounds[:, 0])

    def tell(self, scores: np.ndarray):
        self.optimizer.tell(list(zip(self._solutions, scores.tolist())))

    @property
    def best_params(self) -> np.ndarray:
        mean_normalized = self.optimizer._mean
        return self.bounds[:, 0] + mean_normalized * (self.bounds[:, 1] - self.bounds[:, 0])


# ============================================================================
# Parameter Mapping
# ============================================================================

def build_bounds(args):
    """35 params: [armature*7, static_friction*7, dynamic_ratio*7, viscous_friction*7, encoder_bias*7]."""
    bounds = []
    for _ in range(NUM_ARM_JOINTS):
        bounds.append([args.armature_min, args.armature_max])
    for _ in range(NUM_ARM_JOINTS):
        bounds.append([args.friction_min, args.friction_max])
    for _ in range(NUM_ARM_JOINTS):
        bounds.append([0.0, 1.0])  # dynamic_ratio
    for _ in range(NUM_ARM_JOINTS):
        bounds.append([args.viscous_friction_min, args.viscous_friction_max])
    for _ in range(NUM_ARM_JOINTS):
        bounds.append([args.bias_min, args.bias_max])  # encoder_bias (rad)
    return bounds


def apply_params_to_envs(robot, params_tensor, arm_joint_ids, num_joints, device):
    """Apply 35 params to all envs (joint dynamics). Encoder bias applied to initial pos and score."""
    N = params_tensor.shape[0]
    env_ids = torch.arange(N, device=device)
    J = NUM_ARM_JOINTS

    armature_full = torch.zeros(N, num_joints, device=device)
    static_friction_full = torch.zeros(N, num_joints, device=device)
    dynamic_friction_full = torch.zeros(N, num_joints, device=device)
    viscous_friction_full = torch.zeros(N, num_joints, device=device)

    armature_full[:, arm_joint_ids] = params_tensor[:, 0:J]
    static_fric = params_tensor[:, J:2*J]
    dynamic_ratio = params_tensor[:, 2*J:3*J]
    static_friction_full[:, arm_joint_ids] = static_fric
    dynamic_friction_full[:, arm_joint_ids] = dynamic_ratio * static_fric
    viscous_friction_full[:, arm_joint_ids] = params_tensor[:, 3*J:4*J]

    robot.write_joint_armature_to_sim(armature_full, env_ids=env_ids)
    robot.write_joint_friction_coefficient_to_sim(
        static_friction_full,
        joint_dynamic_friction_coeff=dynamic_friction_full,
        joint_viscous_friction_coeff=viscous_friction_full,
        env_ids=env_ids,
    )


def reset_robot_to_joint_pos(env, robot, initial_arm, initial_hand, bias, arm_joint_ids, settle_steps):
    """Reset robot to initial config (arm + bias, hand) and settle."""
    N = robot.num_instances
    device = robot.device
    env_ids = torch.arange(N, device=device)

    # Full joint pos: arm (with bias) + hand
    arm_pos = initial_arm.unsqueeze(0).expand(N, -1) + bias
    hand_pos = initial_hand.unsqueeze(0).expand(N, -1)
    joint_pos = torch.cat([arm_pos, hand_pos], dim=-1)
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.reset(env_ids)

    # Settle: hold at initial pose for a few steps
    hold_action = joint_pos.clone()
    for _ in range(settle_steps):
        env.step(hold_action)


# ============================================================================
# Main
# ============================================================================

def main():
    args = args_cli
    device_str = args.device
    N = args.num_envs
    num_params = NUM_ARM_JOINTS * 5  # 35

    print("\n" + "="*60)
    print("Franka LEAP Arm System Identification - CMA-ES (Open-Loop Joint Replay)")
    print("="*60)
    print(f"Envs: {N}  Params: {num_params}  Iters: {args.max_iter}  Sigma: {args.sigma}")
    print(f"Controller: FrankaLeapJointPositionAction (same as RL)")
    print(f"Arm-only sysid (7 joints), ImplicitActuatorCfg, no delay")

    # Load real data
    print(f"\nLoading: {args.real_data}")
    real_data = torch.load(args.real_data, map_location="cpu", weights_only=False)
    real_arm_joint_pos = real_data["arm_joint_pos"]
    arm_joint_commands = real_data["arm_joint_pos_target"]
    hand_commands = real_data["hand_actions"]
    initial_arm = real_data["initial_arm_joint_pos"]
    initial_hand = real_data["initial_hand_joint_pos"]
    dt = real_data["dt"]

    T_samples = real_arm_joint_pos.shape[0]
    if args.max_steps is not None:
        T_samples = min(T_samples, args.max_steps)

    print(f"  {T_samples} samples ({T_samples*dt:.2f}s), dt={dt*1000:.1f}ms")

    # Move to GPU
    real_arm_joint_pos = real_arm_joint_pos[:T_samples].to(device_str).float()
    arm_joint_commands = arm_joint_commands[:T_samples].to(device_str).float()
    hand_commands = hand_commands[:T_samples].to(device_str).float()
    initial_arm_dev = initial_arm.to(device_str).float()
    initial_hand_dev = initial_hand.to(device_str).float()

    # Manager-based env
    env_cfg = FrankaLeapSysidEnvCfg()
    env_cfg.scene.num_envs = N
    env_cfg.scene.env_spacing = 2.0

    env = gym.make("UW-FrankaLeap-Sysid-v0", cfg=env_cfg)
    env.reset()

    unwrapped = env.unwrapped
    robot: Articulation = unwrapped.scene["robot"]
    device = unwrapped.device
    arm_joint_ids = robot.find_joints(ARM_JOINT_NAMES)[0]
    num_joints = robot.num_joints
    sim_dt = env_cfg.sim.dt
    action_dim = unwrapped.action_manager.total_action_dim  # 23 (7 arm + 16 hand)

    # Sim steps per real data sample
    sim_steps_per_sample = max(1, int(round(dt / sim_dt)))
    print(f"  sim_dt={sim_dt*1000:.1f}ms, {sim_steps_per_sample} sim steps per sample")

    bounds = build_bounds(args)
    cmaes = CMAES(num_params=num_params, population_size=N, sigma=args.sigma, bounds=bounds)

    print(f"\nBounds: armature[{args.armature_min},{args.armature_max}] "
          f"friction[{args.friction_min},{args.friction_max}] "
          f"dyn_ratio[0,1] viscous[{args.viscous_friction_min},{args.viscous_friction_max}] "
          f"bias[{args.bias_min},{args.bias_max}]")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}\n")

    best_score_ever = float('inf')
    best_params_ever = None
    history = []

    for iteration in range(args.max_iter):
        iter_start = time.time()

        params_np = cmaes.ask()
        params_tensor = torch.tensor(params_np, device=device, dtype=torch.float32)
        apply_params_to_envs(robot, params_tensor, arm_joint_ids, num_joints, device)

        bias = params_tensor[:, 4*NUM_ARM_JOINTS:5*NUM_ARM_JOINTS]  # (N, 7)

        env.reset()
        reset_robot_to_joint_pos(
            env, robot, initial_arm_dev, initial_hand_dev,
            bias, arm_joint_ids, args.settle_steps,
        )

        scores = torch.zeros(N, device=device)

        for s in range(T_samples):
            arm_cmd = arm_joint_commands[s].unsqueeze(0).expand(N, -1)
            hand_cmd = hand_commands[s].unsqueeze(0).expand(N, -1)
            action = torch.cat([arm_cmd, hand_cmd], dim=-1)

            for _ in range(sim_steps_per_sample):
                env.step(action)

            joint_pos = robot.data.joint_pos[:, arm_joint_ids]
            scores += torch.sum((joint_pos - bias - real_arm_joint_pos[s].unsqueeze(0)) ** 2, dim=1)

        scores = scores / T_samples
        scores_np = scores.cpu().numpy()
        cmaes.tell(scores_np)

        min_score = scores_np.min()
        mean_score = scores_np.mean()
        iter_time = time.time() - iter_start

        if min_score < best_score_ever:
            best_score_ever = min_score
            best_params_ever = params_np[scores_np.argmin()]

        history.append({"iteration": iteration, "min": float(min_score),
                        "mean": float(mean_score), "best": float(best_score_ever)})

        rmse_deg = np.degrees(np.sqrt(best_score_ever))
        print(f"[{iteration+1:3d}/{args.max_iter}] "
              f"min={min_score:.6f} mean={mean_score:.6f} best={best_score_ever:.6f} "
              f"({rmse_deg:.3f}\u00b0) {iter_time:.1f}s")

        if (iteration + 1) % args.save_interval == 0:
            ckpt = {"best_params": best_params_ever, "best_score": best_score_ever,
                    "iteration": iteration + 1, "history": history,
                    "bounds": bounds, "args": vars(args)}
            ckpt_path = os.path.join(output_dir, f"checkpoint_{iteration+1:04d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  -> {ckpt_path}")

    # Final results
    print(f"\n{'='*60}")
    print(f"DONE  RMSE: {np.degrees(np.sqrt(best_score_ever)):.4f}\u00b0")
    print(f"{'='*60}")

    arm = best_params_ever[:7]
    sfric = best_params_ever[7:14]
    dratio = best_params_ever[14:21]
    dfric = dratio * sfric
    vfric = best_params_ever[21:28]
    ebias = best_params_ever[28:35]

    print(f"\n  {'Joint':<20s} {'Arm':>8s} {'SFric':>8s} {'DRat':>8s} {'DFric':>8s} {'VFric':>8s} {'Bias°':>8s}")
    for i, name in enumerate(ARM_JOINT_NAMES):
        print(f"  {name:<20s} {arm[i]:8.4f} {sfric[i]:8.4f} {dratio[i]:8.4f} {dfric[i]:8.4f} {vfric[i]:8.4f} {np.degrees(ebias[i]):8.4f}")

    final = {"best_params": best_params_ever, "best_score": best_score_ever,
             "best_armature": arm.tolist(), "best_friction": sfric.tolist(),
             "best_dynamic_ratio": dratio.tolist(), "best_dynamic_friction": dfric.tolist(),
             "best_viscous_friction": vfric.tolist(), "best_encoder_bias": ebias.tolist(),
             "history": history, "bounds": bounds, "args": vars(args)}
    final_path = os.path.join(output_dir, "final_results.pt")
    torch.save(final, final_path)
    print(f"\nSaved: {final_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
