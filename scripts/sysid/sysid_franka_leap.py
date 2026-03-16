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

    # Multiple trajectories (random one per iteration, padding + masking for variable length):
    python scripts/sysid/sysid_franka_leap.py --headless --num_envs 512 \
        --real_data traj1.pt traj2.pt traj3.pt --max_iter 200

    # Save trajectory videos every 10 iterations (first trajectory only):
    python scripts/sysid/sysid_franka_leap.py --num_envs 512 \
        --real_data traj1.pt traj2.pt --max_iter 200 --record_video_every 10
"""

import argparse
import os
import time
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka LEAP Arm System Identification via CMA-ES")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--real_data", type=str, nargs="+", required=True,
                    help="One or more .pt trajectory files (e.g. traj1.pt traj2.pt or dir/*.pt)")
parser.add_argument("--max_iter", type=int, default=200)
parser.add_argument("--sigma", type=float, default=0.3)
parser.add_argument("--output_dir", type=str, default="logs/sysid")
parser.add_argument("--save_interval", type=int, default=5)
parser.add_argument("--record_video_every", type=int, default=0,
                    help="Save trajectory video every N iterations (0=disabled). Enables cameras automatically.")
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
parser.add_argument("--delay_max", type=int, default=5,
                    help="Max motor delay in physics steps. CMA-ES searches [0, delay_max].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras when recording (needed for env.render()), same as RFS train/play_bc
if args_cli.record_video_every > 0:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.assets import Articulation

import uwlab_tasks  # noqa: F401  # register gym envs

from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.sysid_cfg import GraspFrankaLeapSysidCfg
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import RL_MODE
from uwlab_tasks.manager_based.manipulation.grasp.mdp.observations import ee_pose_w
from isaaclab.managers import SceneEntityCfg
import uwlab_assets.robots.franka_leap as franka_leap

# Franka arm: 7 joints
NUM_ARM_JOINTS = 7
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
NUM_HAND_JOINTS = 16
# Arm actuator names (DelayedPDActuatorCfg per link)
ARM_ACTUATOR_NAMES = [f"panda_link{i}" for i in range(1, 8)]
DELAY_PARAM_IDX = NUM_ARM_JOINTS * 5  # 35


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
    """36 params: [armature*7, static_friction*7, dynamic_ratio*7, viscous_friction*7, encoder_bias*7, delay]."""
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
    bounds.append([0.0, float(args.delay_max)])  # motor delay (physics steps)
    return bounds


def apply_params_to_envs(robot, params_tensor, arm_joint_ids, num_joints, device):
    """Apply 36 params to all envs (joint dynamics + delay). Encoder bias applied to initial pos and score."""
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

    delay_int = torch.round(params_tensor[:, DELAY_PARAM_IDX]).clamp(min=0).to(torch.int)
    for name in ARM_ACTUATOR_NAMES:
        act = robot.actuators[name]
        act.positions_delay_buffer.set_time_lag(delay_int)
        act.velocities_delay_buffer.set_time_lag(delay_int)
        act.efforts_delay_buffer.set_time_lag(delay_int)


def record_trajectory_video(
    env,
    output_path,
    best_params,
    arm_joint_commands,
    hand_commands,
    initial_arm,
    initial_hand,
    T_samples,
    sim_steps_per_sample,
    arm_joint_ids,
    num_joints,
    device,
    settle_steps,
    dt,
):
    """Run replay with best params and save video. Uses viewport (sim.render) over env 0."""
    import imageio

    robot = env.unwrapped.scene["robot"]
    N = robot.num_instances
    bias = torch.tensor(best_params[4*NUM_ARM_JOINTS:5*NUM_ARM_JOINTS], device=device, dtype=torch.float32).unsqueeze(0).expand(N, -1)

    params_tensor = torch.tensor(best_params, device=device, dtype=torch.float32).unsqueeze(0).expand(N, -1)
    apply_params_to_envs(robot, params_tensor, arm_joint_ids, num_joints, device)

    env.reset()
    reset_robot_to_joint_pos(
        env, robot, initial_arm, initial_hand,
        bias, arm_joint_ids, settle_steps,
    )

    frames = []
    for s in range(T_samples):
        arm_cmd = arm_joint_commands[s].unsqueeze(0).expand(robot.num_instances, -1)
        hand_cmd = hand_commands[s].unsqueeze(0).expand(robot.num_instances, -1)
        action = torch.cat([arm_cmd, hand_cmd], dim=-1)
        for _ in range(sim_steps_per_sample):
            env.step(action)
        frame = env.render()
        if frame is None:
            raise RuntimeError(
                "env.render() returned None. Use --enable_cameras when record_video_every > 0 "
                "(set automatically). For headless, ensure Isaac Sim supports offscreen rendering."
            )
        rgb = np.asarray(frame)
        frames.append(rgb)

    # Match real-world timing: 1/dt Hz so video duration = T_samples * dt
    imageio.mimwrite(output_path, frames, fps=1.0 / dt)
    print(f"  -> {output_path}")


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


def _compute_ee_pose(env):
    """Get EE pose (7D: pos + quat) for env 0, relative to env origin."""
    ee = ee_pose_w(
        env.unwrapped,
        SceneEntityCfg("robot"),
        franka_leap.FRANKA_LEAP_EE_BODY,
        franka_leap.FRANKA_LEAP_EE_OFFSET,
    )
    return ee[0].cpu().numpy()


def run_best_trajectory_and_collect(
    env,
    robot,
    best_params,
    arm_joint_commands,
    hand_commands,
    initial_arm,
    initial_hand,
    real_arm_joint_pos,
    T_samples,
    sim_steps_per_sample,
    arm_joint_ids,
    num_joints,
    device,
    settle_steps,
):
    """Run best trajectory replay. Returns (sim_joints, sim_ee, real_joints, real_ee)."""
    N = robot.num_instances
    bias = torch.tensor(
        best_params[4 * NUM_ARM_JOINTS : 5 * NUM_ARM_JOINTS],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).expand(N, -1)
    params_tensor = torch.tensor(
        best_params, device=device, dtype=torch.float32
    ).unsqueeze(0).expand(N, -1)
    apply_params_to_envs(robot, params_tensor, arm_joint_ids, num_joints, device)

    env.reset()
    reset_robot_to_joint_pos(
        env, robot, initial_arm, initial_hand, bias, arm_joint_ids, settle_steps
    )

    sim_joints_list = []
    sim_ee_list = []
    for s in range(T_samples):
        arm_cmd = arm_joint_commands[s].unsqueeze(0).expand(N, -1)
        hand_cmd = hand_commands[s].unsqueeze(0).expand(N, -1)
        action = torch.cat([arm_cmd, hand_cmd], dim=-1)
        for _ in range(sim_steps_per_sample):
            env.step(action)
        sim_joints_list.append(
            robot.data.joint_pos[0, arm_joint_ids].cpu().numpy().copy()
        )
        sim_ee_list.append(_compute_ee_pose(env))

    sim_joints = np.array(sim_joints_list)
    sim_ee = np.array(sim_ee_list)

    # Real EE from FK: set joint state to real at each step, step, read EE
    real_joints = real_arm_joint_pos[:T_samples].cpu().numpy()
    real_ee_list = []
    env.reset()
    reset_robot_to_joint_pos(
        env, robot, initial_arm, initial_hand, bias, arm_joint_ids, settle_steps
    )
    unwrapped = env.unwrapped
    env_ids = torch.arange(N, device=device)
    for s in range(T_samples):
        arm_pos = real_arm_joint_pos[s].unsqueeze(0).expand(N, -1)
        hand_pos = initial_hand.unsqueeze(0).expand(N, -1)
        joint_pos = torch.cat([arm_pos, hand_pos], dim=-1)
        joint_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        unwrapped.sim.step()
        real_ee_list.append(_compute_ee_pose(env))
    real_ee = np.array(real_ee_list)

    return sim_joints, sim_ee, real_joints, real_ee


def plot_best_trajectory_comparison(
    output_dir,
    sim_joints,
    sim_ee,
    real_joints,
    real_ee,
    dt,
):
    """Plot joints and EE pose: sim vs real over the best trajectory."""
    T = sim_joints.shape[0]
    time_s = np.arange(T) * dt

    # Joints: 7 subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i in range(NUM_ARM_JOINTS):
        ax = axes[i]
        ax.plot(time_s, np.degrees(real_joints[:, i]), "b-", label="Real", linewidth=2)
        ax.plot(time_s, np.degrees(sim_joints[:, i]), "r--", label="Sim", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("deg")
        ax.set_title(f"{ARM_JOINT_NAMES[i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[7].axis("off")
    plt.suptitle(
        "Best Trajectory: Arm Joints (Sim vs Real)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "best_trajectory_joints.png"), dpi=150)
    plt.close()

    # EE pose: pos (x,y,z) + quat (qw,qx,qy,qz)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    labels = ["x (m)", "y (m)", "z (m)", "qw", "qx", "qy"]
    for i in range(6):
        ax = axes[i]
        ax.plot(time_s, real_ee[:, i], "b-", label="Real", linewidth=2)
        ax.plot(time_s, sim_ee[:, i], "r--", label="Sim", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(labels[i])
        ax.set_title(f"EE {labels[i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        "Best Trajectory: EE Pose (Sim vs Real)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "best_trajectory_ee_pose.png"), dpi=150)
    plt.close()

    # EE position error over time
    pos_error = np.linalg.norm(real_ee[:, :3] - sim_ee[:, :3], axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_s, pos_error * 1000, "b-", linewidth=2)
    ax.axhline(
        y=np.mean(pos_error) * 1000,
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(pos_error)*1000:.1f} mm",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (mm)")
    ax.set_title("EE Position Error Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_trajectory_ee_error.png"), dpi=150)
    plt.close()

    print(f"  -> {output_dir}/best_trajectory_joints.png")
    print(f"  -> {output_dir}/best_trajectory_ee_pose.png")
    print(f"  -> {output_dir}/best_trajectory_ee_error.png")


# ============================================================================
# Multi-trajectory loading
# ============================================================================

def load_trajectories(paths, max_steps, device):
    """Load trajectories from .pt files, pad to max length, return list of dicts."""
    trajectories = []
    lengths = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        T = data["arm_joint_pos"].shape[0]
        if max_steps is not None:
            T = min(T, max_steps)
        lengths.append(T)
        trajectories.append({
            "arm_joint_pos": data["arm_joint_pos"][:T],
            "arm_joint_pos_target": data["arm_joint_pos_target"][:T],
            "hand_actions": data["hand_actions"][:T],
            "initial_arm_joint_pos": data["initial_arm_joint_pos"],
            "initial_hand_joint_pos": data["initial_hand_joint_pos"],
            "dt": data["dt"],
        })

    T_max = max(lengths)
    dt = trajectories[0]["dt"]

    for i, traj in enumerate(trajectories):
        T = lengths[i]
        if T < T_max:
            # Pad by repeating last frame
            pad_arm = traj["arm_joint_pos"][-1:].expand(T_max - T, -1)
            pad_target = traj["arm_joint_pos_target"][-1:].expand(T_max - T, -1)
            pad_hand = traj["hand_actions"][-1:].expand(T_max - T, -1)
            traj["arm_joint_pos"] = torch.cat([traj["arm_joint_pos"], pad_arm], dim=0)
            traj["arm_joint_pos_target"] = torch.cat([traj["arm_joint_pos_target"], pad_target], dim=0)
            traj["hand_actions"] = torch.cat([traj["hand_actions"], pad_hand], dim=0)
        traj["valid_length"] = T
        traj["arm_joint_pos"] = traj["arm_joint_pos"].to(device).float()
        traj["arm_joint_pos_target"] = traj["arm_joint_pos_target"].to(device).float()
        traj["hand_actions"] = traj["hand_actions"].to(device).float()
        traj["initial_arm_joint_pos"] = traj["initial_arm_joint_pos"].to(device).float()
        traj["initial_hand_joint_pos"] = traj["initial_hand_joint_pos"].to(device).float()

    return trajectories, T_max, dt


# ============================================================================
# Main
# ============================================================================

def main():
    args = args_cli
    device_str = args.device
    N = args.num_envs
    num_params = NUM_ARM_JOINTS * 5 + 1  # 36 (dynamics + delay)

    print("\n" + "="*60)
    print("Franka LEAP Arm System Identification - CMA-ES (Open-Loop Joint Replay)")
    print("="*60)
    print(f"Envs: {N}  Params: {num_params}  Iters: {args.max_iter}  Sigma: {args.sigma}")
    print(f"Controller: FrankaLeapJointPositionAction (same as RL)")
    print(f"Arm-only sysid (7 joints), DelayedPDActuatorCfg, delay in search [0, {args.delay_max}]")

    # Load trajectories (multiple .pt files, padded to max length)
    print(f"\nLoading {len(args.real_data)} trajectory file(s):")
    for p in args.real_data:
        print(f"  {p}")
    trajectories, T_max, dt = load_trajectories(args.real_data, args.max_steps, device_str)
    traj0 = trajectories[0]
    print(f"  {len(trajectories)} trajectories, T_max={T_max} samples ({T_max*dt:.2f}s), dt={dt*1000:.1f}ms")

    # GraspFrankaLeapJointAbsCfg with real gains (same env as RL)
    env_cfg = GraspFrankaLeapSysidCfg()
    env_cfg.scene.num_envs = N
    env_cfg.scene.env_spacing = 2.0
    env_cfg.run_mode = RL_MODE  # no scene cameras; viewport (sim.render) used for recording

    env = gym.make("UW-FrankaLeap-Sysid-v0", cfg=env_cfg, render_mode="rgb_array" if args.record_video_every > 0 else None)
    env.reset()

    unwrapped = env.unwrapped
    robot: Articulation = unwrapped.scene["robot"]
    device = unwrapped.device
    arm_joint_ids = robot.find_joints(ARM_JOINT_NAMES)[0]
    num_joints = robot.num_joints
    sim_dt = env_cfg.sim.dt
    action_dim = unwrapped.action_manager.total_action_dim  # 23 (7 arm + 16 hand)

    # Sim steps per real data sample (dt from first trajectory)
    sim_steps_per_sample = max(1, int(round(dt / sim_dt)))
    print(f"  sim_dt={sim_dt*1000:.1f}ms, {sim_steps_per_sample} sim steps per sample")

    bounds = build_bounds(args)
    cmaes = CMAES(num_params=num_params, population_size=N, sigma=args.sigma, bounds=bounds)

    print(f"\nBounds: armature[{args.armature_min},{args.armature_max}] "
          f"friction[{args.friction_min},{args.friction_max}] "
          f"dyn_ratio[0,1] viscous[{args.viscous_friction_min},{args.viscous_friction_max}] "
          f"bias[{args.bias_min},{args.bias_max}] delay[0,{args.delay_max}]")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}\n")

    if args.record_video_every > 0:
        print(f"Video recording enabled: every {args.record_video_every} iterations (captures env 0)")

    best_score_ever = float('inf')
    best_params_ever = None
    history = []

    for iteration in range(args.max_iter):
        iter_start = time.time()

        traj_idx = np.random.randint(0, len(trajectories))
        traj = trajectories[traj_idx]
        valid_length = traj["valid_length"]

        params_np = cmaes.ask()
        params_tensor = torch.tensor(params_np, device=device, dtype=torch.float32)
        apply_params_to_envs(robot, params_tensor, arm_joint_ids, num_joints, device)

        bias = params_tensor[:, 4*NUM_ARM_JOINTS:5*NUM_ARM_JOINTS]  # (N, 7)

        env.reset()
        reset_robot_to_joint_pos(
            env, robot, traj["initial_arm_joint_pos"], traj["initial_hand_joint_pos"],
            bias, arm_joint_ids, args.settle_steps,
        )

        scores = torch.zeros(N, device=device)

        for s in range(T_max):
            arm_cmd = traj["arm_joint_pos_target"][s].unsqueeze(0).expand(N, -1)
            hand_cmd = traj["hand_actions"][s].unsqueeze(0).expand(N, -1)
            action = torch.cat([arm_cmd, hand_cmd], dim=-1)

            for _ in range(sim_steps_per_sample):
                env.step(action)

            if s < valid_length:
                joint_pos = robot.data.joint_pos[:, arm_joint_ids]
                scores += torch.sum(
                    (joint_pos - bias - traj["arm_joint_pos"][s].unsqueeze(0)) ** 2,
                    dim=1,
                )

        scores = scores / valid_length
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

        if iteration % args.save_interval == 0:
            ckpt = {"best_params": best_params_ever, "best_score": best_score_ever,
                    "iteration": iteration + 1, "history": history,
                    "bounds": bounds, "args": vars(args)}
            ckpt_path = os.path.join(output_dir, f"checkpoint_{iteration+1:04d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  -> {ckpt_path}")

        if args.record_video_every > 0 and iteration % args.record_video_every == 0:
            video_path = os.path.join(output_dir, f"trajectory_iter_{iteration+1:04d}.mp4")
            record_trajectory_video(
                env,
                video_path,
                best_params_ever,
                traj0["arm_joint_pos_target"],
                traj0["hand_actions"],
                traj0["initial_arm_joint_pos"],
                traj0["initial_hand_joint_pos"],
                traj0["valid_length"],
                sim_steps_per_sample,
                arm_joint_ids,
                num_joints,
                device,
                args.settle_steps,
                dt,
            )

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
    delay = int(round(best_params_ever[DELAY_PARAM_IDX]))

    print(f"\n  {'Joint':<20s} {'Arm':>8s} {'SFric':>8s} {'DRat':>8s} {'DFric':>8s} {'VFric':>8s} {'Bias°':>8s}")
    for i, name in enumerate(ARM_JOINT_NAMES):
        print(f"  {name:<20s} {arm[i]:8.4f} {sfric[i]:8.4f} {dratio[i]:8.4f} {dfric[i]:8.4f} {vfric[i]:8.4f} {np.degrees(ebias[i]):8.4f}")
    print(f"  delay (physics steps): {delay}")

    final = {"best_params": best_params_ever, "best_score": best_score_ever,
             "best_armature": arm.tolist(), "best_friction": sfric.tolist(),
             "best_dynamic_ratio": dratio.tolist(), "best_dynamic_friction": dfric.tolist(),
             "best_viscous_friction": vfric.tolist(), "best_encoder_bias": ebias.tolist(),
             "best_delay": delay,
             "history": history, "bounds": bounds, "args": vars(args)}
    final_path = os.path.join(output_dir, "final_results.pt")
    torch.save(final, final_path)
    print(f"\nSaved: {final_path}")

    # Plot best trajectory: joints and EE pose (sim vs real) — first trajectory only
    print("\nPlotting best trajectory comparison (first trajectory)...")
    sim_joints, sim_ee, real_joints, real_ee = run_best_trajectory_and_collect(
        env,
        robot,
        best_params_ever,
        traj0["arm_joint_pos_target"],
        traj0["hand_actions"],
        traj0["initial_arm_joint_pos"],
        traj0["initial_hand_joint_pos"],
        traj0["arm_joint_pos"],
        traj0["valid_length"],
        sim_steps_per_sample,
        arm_joint_ids,
        num_joints,
        device,
        args.settle_steps,
    )
    plot_best_trajectory_comparison(
        output_dir, sim_joints, sim_ee, real_joints, real_ee, dt
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
