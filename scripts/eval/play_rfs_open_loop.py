"""
Open-loop RFS evaluation: trajectory train-target actions vs RFS decode on (1) logged obs
and (2) sim observations after reset. Assumes ``residual_step == 1`` (one env action per PPO chunk).

Uses ``RFSWrapper.decode_ppo_to_env_action`` + ``open_loop_advance_buffers`` (no env step on the
real-obs branch). Loads BC train_cfg from ``diffusion_path`` for trajectory action vectors and
``policy_obs_from_traj_step`` keys.

Usage:
    ./uwlab.sh -p scripts/eval/play_rfs_open_loop.py --eval_cfg configs/eval/bottle_pour_bc_jointabs.yaml --trajectory_file data_storage/datasets/03_24_bourbon_pour/episode_68/episode_68.zarr --diffusion_path logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise  --rfs_cfg configs/rl/arm_rfs_joint_cfg.yaml  --rfs_checkpoint logs/rfs/PourBottle-JointAbs_0326_2312_28e64e/model_000650.zip --headless --enable_cameras
"""

import argparse
import json
import os
import sys

from uwlab.utils.paths import setup_third_party_paths

setup_third_party_paths()

from isaaclab.app import AppLauncher

_UWLAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_RFS_DIR = os.path.join(_UWLAB_DIR, "scripts/reinforcement_learning/sb3/rfs")
if _RFS_DIR not in sys.path:
    sys.path.insert(0, _RFS_DIR)

parser = argparse.ArgumentParser(description="RFS open-loop eval on recorded trajectory vs sim reset.")
parser.add_argument("--eval_cfg", type=str, required=True, help="YAML: task_id, checkpoint (BC dir), device, ...")
parser.add_argument("--trajectory_file", type=str, required=True, help="Episode zarr / dir for load_real_episode.")
parser.add_argument(
    "--diffusion_path",
    type=str,
    default=None,
    help="BC checkpoint directory (frozen CFM inside RFSWrapper). Overrides eval_cfg.",
)
parser.add_argument("--rfs_cfg", type=str, default="configs/rl/rfs_cfg.yaml", help="RFS yaml (noise/residual slices).")
parser.add_argument(
    "--rfs_checkpoint",
    type=str,
    default=None,
    help="PPO .zip; if omitted, Gaussian noise over full PPO action dim.",
)
parser.add_argument("--noise_dims", type=str, default=None, help="Override rfs_cfg ``start:end``.")
parser.add_argument("--residual_dims", type=str, default=None, help="Override rfs_cfg ``start:end``.")
parser.add_argument(
    "--action_type",
    type=str,
    choices=["delta_ee", "abs_ee", "joint", "delta_joint"],
    default="joint",
    help="Hold action for reset_to_real_joints.",
)
parser.add_argument(
    "--sim_type",
    type=str,
    choices=["eval", "distill", "rl"],
    default="rl",
    help="eval: synthetic seg_pc. rl: RL task mode (typical for RFS checkpoints). distill: rendered seg.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--sim_pc_downsample", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--camera", type=str, default="fixed_camera")
parser.add_argument(
    "--asymmetric_ac",
    action="store_true",
    default=False,
    help="Force asymmetric AC obs layout; else inferred from --rfs_checkpoint when set.",
)
parser.add_argument("overrides", nargs="*", help="eval_cfg overrides (e.g. checkpoint=/path)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO

import uwlab_tasks  # noqa: F401
import uwlab_assets.robots.franka_leap as franka_leap
from asymmetric_policy import AsymmetricActorCriticPolicy, resolve_asymmetric_ac
from eval_callback import _sb3_process_obs
from uwlab.utils.checkpoint import extract_ckpt_metadata, format_ckpt_metadata
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    ARM_RESET,
    DISTILL_MODE,
    EVAL_MODE,
    HAND_RESET,
    RL_MODE,
    parse_franka_leap_env_cfg,
)
from uwlab_tasks.manager_based.manipulation.grasp.mdp.events import reset_robot_joints
from uwlab_tasks.utils.trajectory_utils import load_real_episode
from wrapper import RFSWrapper


def _load_eval_cfg(path: str, overrides: list) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        key, _, val = kv.partition("=")
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.lstrip("-").replace(".", "").isdigit():
            val = float(val) if "." in val else int(val)
        cfg[key] = val
    return cfg


def _load_train_cfg(checkpoint_dir: str) -> dict:
    candidates = [
        os.path.join(checkpoint_dir, ".hydra", "config.yaml"),
        os.path.join(checkpoint_dir, "config.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Training config not found in {checkpoint_dir}")


def _find_checkpoint(checkpoint_dir: str, ckpt_name: str | None = None) -> str:
    if ckpt_name is not None:
        path = os.path.join(checkpoint_dir, "checkpoints", ckpt_name)
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    for name in ("best.ckpt", "latest.ckpt"):
        path = os.path.join(checkpoint_dir, "checkpoints", name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}/checkpoints/")


def _load_yaml(path: str) -> dict:
    p = path if os.path.isabs(path) else os.path.join(_UWLAB_DIR, path)
    with open(p) as f:
        return yaml.safe_load(f)


def _parse_dims(s: str | None):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected 'start:end', got: {s!r}")
    return (int(parts[0]), int(parts[1]))


# --- Trajectory / reset helpers (same as play_bc_open_loop; do not import that module). ---


def _extract_real_joints(obs: dict):
    if obs is None:
        return None, None

    def to_flat(a):
        a = np.asarray(a).reshape(-1)
        return a if a.size >= 23 else None

    policy = obs.get("policy")
    if isinstance(policy, dict):
        j = policy.get("joint_pos")
        if j is not None:
            j = to_flat(j)
            if j is not None:
                return j[:7], j[7:23]

    j = obs.get("joint_pos")
    if j is None:
        j = obs.get("joint_positions")
    if j is not None:
        j = to_flat(j)
        if j is not None:
            return j[:7], j[7:23]

    arm = obs.get("joint_positions")
    hand = obs.get("gripper_position")
    if arm is not None and hand is not None:
        arm = np.asarray(arm).reshape(-1)
        hand = np.asarray(hand).reshape(-1)
        if arm.size >= 7 and hand.size >= 16:
            return arm[:7], hand[:16]

    return None, None


def _build_hold_from_real_obs(obs: dict, action_type: str, device: torch.device, num_envs: int) -> torch.Tensor:
    arm, hand = _extract_real_joints(obs)
    if arm is None or hand is None:
        raise ValueError("Cannot extract joints from obs for hold")
    hand = np.asarray(hand).reshape(-1)[:16]

    if action_type == "joint":
        arr = np.concatenate([arm, hand], axis=-1)
    elif action_type == "delta_joint":
        arr = np.concatenate([np.zeros(7, dtype=np.float32), hand], axis=-1)
    elif action_type == "delta_ee":
        arr = np.concatenate([np.zeros(6, dtype=np.float32), hand], axis=-1)
    elif action_type == "abs_ee":
        ee = obs.get("cartesian_position")
        if ee is None:
            raise KeyError("obs needs cartesian_position for abs_ee hold")
        ee = np.asarray(ee).reshape(-1)[:7]
        arr = np.concatenate([ee, hand], axis=-1)
    else:
        raise ValueError(f"Unsupported action_type: {action_type}")

    return torch.tensor(arr, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)


def reset_to_real_joints(env, first_real_obs, num_warmup_steps: int = 10, hold_from_obs: bool = False, action_type: str = "joint"):
    arm, hand = _extract_real_joints(first_real_obs)
    if arm is None or hand is None:
        return None

    unwrapped = env.unwrapped
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
    reset_robot_joints(
        unwrapped,
        env_ids,
        SceneEntityCfg("robot"),
        arm.tolist(),
        hand.tolist(),
        arm_joint_limits=franka_leap.FRANKA_LEAP_ARM_JOINT_LIMITS,
    )
    if hold_from_obs:
        hold = _build_hold_from_real_obs(first_real_obs, action_type, unwrapped.device, unwrapped.num_envs)
    else:
        hold = torch.tensor(ARM_RESET + HAND_RESET, device=unwrapped.device, dtype=torch.float32).unsqueeze(0).repeat(
            unwrapped.num_envs, 1
        )

    obs_after_reset = None
    for _ in range(num_warmup_steps):
        obs_after_reset, _, _, _, _ = env.step(hold)
    return obs_after_reset


def _extract_sim_observation(obs: dict):
    policy = obs.get("policy")
    if not isinstance(policy, dict):
        return None
    ee_pose = policy.get("ee_pose")
    joint_pos = policy.get("joint_pos")
    if ee_pose is None or joint_pos is None:
        return None
    ee_pose_np = ee_pose[0].detach().cpu().numpy().reshape(-1)
    joint_pos_np = joint_pos[0].detach().cpu().numpy().reshape(-1)
    arm_joints = joint_pos_np[:7]
    hand_joints = joint_pos_np[7:7 + 16]
    return {
        "cartesian_position": ee_pose_np,
        "joint_positions": arm_joints,
        "gripper_position": hand_joints,
    }


def _traj_arm7(traj_obs_t: dict) -> np.ndarray:
    if "arm_joint_pos" in traj_obs_t:
        return np.asarray(traj_obs_t["arm_joint_pos"], dtype=np.float32).reshape(-1)[:7]
    if "joint_positions" in traj_obs_t:
        return np.asarray(traj_obs_t["joint_positions"], dtype=np.float32).reshape(-1)[:7]
    raise KeyError("Trajectory obs needs 'arm_joint_pos' or 'joint_positions'")


def _traj_hand16(traj_obs_t: dict) -> np.ndarray:
    if "hand_joint_pos" in traj_obs_t:
        return np.asarray(traj_obs_t["hand_joint_pos"], dtype=np.float32).reshape(-1)[:16]
    if "gripper_position" in traj_obs_t:
        return np.asarray(traj_obs_t["gripper_position"], dtype=np.float32).reshape(-1)[:16]
    raise KeyError("Trajectory obs needs 'hand_joint_pos' or 'gripper_position'")


def _traj_ee7(traj_obs_t: dict) -> np.ndarray:
    if "ee_pose" in traj_obs_t:
        return np.asarray(traj_obs_t["ee_pose"], dtype=np.float32).reshape(-1)[:7]
    if "cartesian_position" in traj_obs_t:
        return np.asarray(traj_obs_t["cartesian_position"], dtype=np.float32).reshape(-1)[:7]
    raise KeyError("Trajectory obs needs 'ee_pose' or 'cartesian_position'")


def _traj_seg_pc_b13n(traj_obs_t: dict) -> torch.Tensor:
    if "seg_pc" not in traj_obs_t:
        raise KeyError("Trajectory obs needs 'seg_pc'")
    pc = np.asarray(traj_obs_t["seg_pc"], dtype=np.float32)
    if pc.ndim != 2:
        raise ValueError(f"seg_pc must be 2D, got shape {pc.shape}")
    if pc.shape[1] == 3:
        pc = pc.T
    elif pc.shape[0] != 3:
        raise ValueError(f"seg_pc expected (N,3) or (3,N), got {pc.shape}")
    return torch.from_numpy(pc).unsqueeze(0)


def policy_obs_from_traj_step(traj_obs_t: dict, device: torch.device, obs_keys: list[str], image_keys: list[str]) -> dict:
    needed = set(obs_keys) | set(image_keys)
    out = {}

    if "ee_pose" in needed:
        ee = _traj_ee7(traj_obs_t)
        out["ee_pose"] = torch.from_numpy(ee).unsqueeze(0).to(device)
    if "arm_joint_pos" in needed:
        arm = _traj_arm7(traj_obs_t)
        out["arm_joint_pos"] = torch.from_numpy(arm).unsqueeze(0).to(device)
    if "hand_joint_pos" in needed:
        hand = _traj_hand16(traj_obs_t)
        out["hand_joint_pos"] = torch.from_numpy(hand).unsqueeze(0).to(device)
    if "joint_pos" in needed:
        arm = _traj_arm7(traj_obs_t)
        hand = _traj_hand16(traj_obs_t)
        jp = np.concatenate([arm, hand], axis=-1)
        out["joint_pos"] = torch.from_numpy(jp).unsqueeze(0).to(device)

    if "seg_pc" in needed:
        out["seg_pc"] = _traj_seg_pc_b13n(traj_obs_t).to(device)

    missing = (set(obs_keys) | set(image_keys)) - set(out.keys())
    if missing:
        raise KeyError(
            f"policy_obs_from_traj_step cannot build keys {sorted(missing)}; built {sorted(out.keys())}"
        )
    return out


def _training_action_vector_numpy(traj_obs_t: dict, train_cfg: dict, action_dim: int, raw_actions_row) -> np.ndarray:
    ds = train_cfg.get("dataset") or {}
    ak = ds.get("action_key", "actions")
    keys = list(ak) if hasattr(ak, "__iter__") and not isinstance(ak, str) else [ak]

    if len(keys) == 1 and keys[0] == "actions":
        a = np.asarray(raw_actions_row, dtype=np.float32).reshape(-1)
        if a.shape[0] != action_dim:
            raise ValueError(f"action_key 'actions' dim {a.shape[0]} != {action_dim}")
        return a.astype(np.float64)

    def segment_for_zarr_key(zkey: str) -> np.ndarray:
        if zkey == "arm_joint_pos_target":
            if "commanded_joint_positions" in traj_obs_t:
                v = traj_obs_t["commanded_joint_positions"]
            elif "ik_joint_pos_desired" in traj_obs_t:
                v = traj_obs_t["ik_joint_pos_desired"]
            else:
                raise KeyError(
                    "Need 'commanded_joint_positions' or 'ik_joint_pos_desired' in trajectory obs "
                    "for arm_joint_pos_target (matches zarr arm_joint_pos_target)."
                )
            return np.asarray(v, dtype=np.float32).reshape(-1)
        if zkey == "hand_action":
            if "hand_action" in traj_obs_t:
                v = traj_obs_t["hand_action"]
            else:
                v = traj_obs_t["hand_joint_pos"]
            return np.asarray(v, dtype=np.float32).reshape(-1)
        if zkey in traj_obs_t:
            return np.asarray(traj_obs_t[zkey], dtype=np.float32).reshape(-1)
        raise KeyError(f"No field for action segment '{zkey}'")

    parts = [segment_for_zarr_key(str(k)) for k in keys]
    a = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
    if a.shape[0] != action_dim:
        raise ValueError(f"action_key {keys} dim {a.shape[0]} != {action_dim}")
    return a.astype(np.float64)


class GaussianNoisePolicy:
    def __init__(self, rfs_env: RFSWrapper):
        self.rfs_env = rfs_env

    def predict(self, obs, deterministic: bool = False):
        b = self.rfs_env.num_envs
        d = self.rfs_env.n_residual_flat + self.rfs_env.n_noise * self.rfs_env.policy_horizon
        if deterministic:
            return np.zeros((b, d), dtype=np.float32), {}
        return np.random.randn(b, d).astype(np.float32), {}


def _run_rfs_open_loop_pass(rfs_env: RFSWrapper, model, device: torch.device, T: int, policy_obs_fn) -> np.ndarray:
    cfm_dim = rfs_env.cfm_action_dim
    out = np.zeros((T, cfm_dim), dtype=np.float64)
    rfs_env.reset_open_loop_buffers()
    with torch.inference_mode():
        for i in range(T):
            pol = policy_obs_fn(i)
            rfs_env.last_obs = {"policy": pol}
            if rfs_env.asymmetric_ac:
                rfs_env.last_pcd_embedding = rfs_env._compute_pcd_embedding()
            stripped = rfs_env._strip_ppo_obs(rfs_env.last_obs)
            obs_np = _sb3_process_obs(stripped)
            ppo_action, _ = model.predict(obs_np, deterministic=True)
            ppo_t = torch.from_numpy(np.asarray(ppo_action, dtype=np.float32)).to(device)
            env_act = rfs_env.decode_ppo_to_env_action(ppo_t)
            out[i] = env_act[0].detach().cpu().numpy()
            rfs_env.open_loop_advance_buffers(env_act)
    return out


def _plot_actions(output_path: str, actions_traj: np.ndarray, a_real: np.ndarray, a_sim: np.ndarray):
    T, D = a_real.shape
    assert actions_traj.shape == (T, D) and a_sim.shape == (T, D)
    n_cols = min(4, D)
    n_rows = int(np.ceil(D / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 2.8 * n_rows), squeeze=False)
    t = np.arange(T)
    for d in range(D):
        r, c = divmod(d, n_cols)
        ax = axes[r][c]
        ax.plot(t, actions_traj[:, d], label="trajectory (train target)", linewidth=1.5, color="#2ca02c")
        ax.plot(t, a_real[:, d], label="RFS | real obs", linewidth=1.4, color="#1f77b4")
        ax.plot(t, a_sim[:, d], label="RFS | sim reset", linewidth=1.4, alpha=0.9, color="#ff7f0e")
        ax.set_title(f"action[{d}]")
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7, loc="best")
    for d in range(D, n_rows * n_cols):
        r, c = divmod(d, n_cols)
        axes[r][c].axis("off")
    fig.suptitle("RFS: trajectory vs policy on real vs sim observations (CFM action space)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_joints(output_path: str, real_arm, real_hand, sim_arm, sim_hand, title_prefix: str):
    T = real_arm.shape[0]
    t = np.arange(T)
    idx = 0
    D = real_arm.shape[1] + real_hand.shape[1]
    n_cols = min(4, D)
    n_rows = int(np.ceil(D / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 2.5 * n_rows), squeeze=False)
    for j in range(real_arm.shape[1]):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.plot(t, real_arm[:, j], label="logged", linewidth=1.3)
        ax.plot(t, sim_arm[:, j], label="sim_after_reset", linewidth=1.3, alpha=0.85)
        ax.set_title(f"arm joint {j}")
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
        idx += 1
    for j in range(real_hand.shape[1]):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.plot(t, real_hand[:, j], label="logged", linewidth=1.3)
        ax.plot(t, sim_hand[:, j], label="sim_after_reset", linewidth=1.3, alpha=0.85)
        ax.set_title(f"hand joint {j}")
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)
        idx += 1
    for d in range(idx, n_rows * n_cols):
        r, c = divmod(d, n_cols)
        axes[r][c].axis("off")
    fig.suptitle(f"{title_prefix}: joint state (logged vs sim)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    eval_cfg = _load_eval_cfg(args_cli.eval_cfg, args_cli.overrides)

    diffusion_path = args_cli.diffusion_path or eval_cfg.get("diffusion_path") or eval_cfg.get("checkpoint")
    if diffusion_path is None:
        raise ValueError("Set --diffusion_path or eval_cfg checkpoint / diffusion_path (BC directory).")
    diffusion_path = os.path.expanduser(diffusion_path)

    device = torch.device(eval_cfg.get("device", "cuda:0"))

    ckpt_path = _find_checkpoint(diffusion_path, eval_cfg.get("checkpoint_name"))
    print(f"[play_rfs_open_loop] BC checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[play_rfs_open_loop] Metadata: {format_ckpt_metadata(extract_ckpt_metadata(ckpt))}")
    train_cfg = ckpt.get("cfg") or _load_train_cfg(diffusion_path)
    ds_cfg = train_cfg["dataset"]
    obs_keys = list(eval_cfg.get("obs_keys") or ds_cfg["obs_keys"])
    image_keys = list(ds_cfg["image_keys"])
    downsample_points = int(ds_cfg["downsample_points"])
    cfm_action_dim = int(ckpt["ema_model"]["normalizer.params_dict.action.scale"].shape[0])

    rfs_all = _load_yaml(args_cli.rfs_cfg)
    ppo_cfg = rfs_all["ppo"]
    rfs_cfg = rfs_all["rfs"]
    noise_dims = _parse_dims(args_cli.noise_dims or rfs_cfg["noise_dims"])
    residual_dims = _parse_dims(args_cli.residual_dims or rfs_cfg.get("residual_dims") or "")

    episode = load_real_episode(args_cli.trajectory_file)
    traj_obs = episode["obs"]
    traj_actions = episode["actions"]
    T = min(len(traj_obs), len(traj_actions))
    if T == 0:
        raise ValueError("Empty trajectory")
    traj_obs = traj_obs[:T]
    traj_actions = traj_actions[:T]

    task_id = eval_cfg["task_id"]
    if args_cli.sim_type == "eval":
        run_mode = EVAL_MODE
    elif args_cli.sim_type == "distill":
        run_mode = DISTILL_MODE
    else:
        run_mode = RL_MODE
    env_cfg = parse_franka_leap_env_cfg(
        task_id,
        run_mode=run_mode,
        device=str(device),
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg, "table_z_range"):
        env_cfg.table_z_range = (0.0, 0.0)
    if args_cli.sim_type == "eval":
        env_cfg.scene.train_camera = None
        if hasattr(env_cfg.events, "reset_camera"):
            env_cfg.events.reset_camera = None
    env_cfg.scene.fixed_camera = None
    if hasattr(env_cfg.events, "reset_fixed_camera"):
        env_cfg.events.reset_fixed_camera = None
    if args_cli.sim_type == "distill":
        env_cfg.distill_camera_name = args_cli.camera

    num_warmup_steps = int(eval_cfg.get("num_warmup_steps", getattr(env_cfg, "num_warmup_steps", 10)))
    max_steps = 2 * num_warmup_steps + 1 + 2 * T + 100
    env_cfg.episode_length_s = max_steps * env_cfg.decimation * env_cfg.sim.dt
    for term in ("bottle_dropped", "bottle_too_far", "cup_toppled"):
        if hasattr(env_cfg.terminations, term):
            setattr(env_cfg.terminations, term, None)
    if hasattr(env_cfg, "table_z_range"):
        env_cfg.table_z_range = (-0.02, -0.02)
    if hasattr(env_cfg.events, "reset_table_block") and env_cfg.events.reset_table_block is not None:
        env_cfg.events.reset_table_block.params["z_range"] = (-0.02, -0.02)
    if hasattr(env_cfg.events, "reset_object") and env_cfg.events.reset_object is not None:
        env_cfg.events.reset_object.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    sim_pc_ds = args_cli.sim_pc_downsample if args_cli.sim_pc_downsample is not None else downsample_points
    env_cfg.seg_pc_num_downsample_points = sim_pc_ds
    seg_term = getattr(getattr(env_cfg.observations.policy, "seg_pc", None), "func", None)
    if seg_term is not None:
        pc_obj = getattr(seg_term, "__self__", None)
        if pc_obj is not None and hasattr(pc_obj, "num_downsample_points"):
            pc_obj.num_downsample_points = env_cfg.seg_pc_num_downsample_points

    base_env = gym.make(task_id, cfg=env_cfg, render_mode=None)

    asymmetric_ac = resolve_asymmetric_ac(args_cli.asymmetric_ac, rfs_cfg, args_cli.rfs_checkpoint)

    rfs_env = RFSWrapper(
        base_env,
        diffusion_path=diffusion_path,
        residual_step=rfs_cfg["residual_step"],
        noise_dims=noise_dims,
        residual_dims=residual_dims,
        residual_scale=rfs_cfg["residual_scale"],
        clip_actions=rfs_cfg["clip_actions"],
        finger_smooth_alpha=rfs_cfg["finger_smooth_alpha"],
        finger_start_dim=rfs_cfg.get("finger_start_dim", 6),
        num_warmup_steps=rfs_cfg.get("num_warmup_steps", 0),
        asymmetric_ac=asymmetric_ac,
        gamma=float(ppo_cfg["gamma"]),
        ppo_history=rfs_cfg.get("ppo_history", False),
    )

    if rfs_env.residual_step != 1:
        print(
            f"[play_rfs_open_loop] WARNING: residual_step={rfs_env.residual_step} != 1; "
            "plots use the first substep decode only."
        )

    if rfs_env.cfm_action_dim != cfm_action_dim:
        raise ValueError(f"CFM action dim mismatch: wrapper {rfs_env.cfm_action_dim} vs ckpt {cfm_action_dim}")

    sb3_env = Sb3VecEnvWrapper(rfs_env)
    if args_cli.rfs_checkpoint:
        ckpt_ppo = os.path.expanduser(args_cli.rfs_checkpoint)
        custom_objects = {"policy_class": AsymmetricActorCriticPolicy} if asymmetric_ac else None
        model = PPO.load(ckpt_ppo, env=sb3_env, custom_objects=custom_objects, print_system_info=False)
    else:
        model = GaussianNoisePolicy(rfs_env)

    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        eval_base = os.path.join(os.path.dirname(diffusion_path.rstrip("/")), "eval")
        stem = os.path.splitext(os.path.basename(args_cli.eval_cfg))[0]
        output_dir = os.path.join(eval_base, stem, os.path.basename(diffusion_path.rstrip("/")), "rfs_open_loop")
    os.makedirs(output_dir, exist_ok=True)

    actions_traj = np.stack(
        [_training_action_vector_numpy(traj_obs[i], train_cfg, rfs_env.cfm_action_dim, traj_actions[i]) for i in range(T)]
    )

    def policy_obs_real(i: int) -> dict:
        return policy_obs_from_traj_step(traj_obs[i], device, obs_keys, image_keys)

    actions_rfs_real = _run_rfs_open_loop_pass(rfs_env, model, device, T, policy_obs_real)

    real_arm = np.stack([np.asarray(traj_obs[i]["joint_positions"], dtype=np.float64).reshape(-1)[:7] for i in range(T)])
    real_hand = np.stack([np.asarray(traj_obs[i]["gripper_position"], dtype=np.float64).reshape(-1)[:16] for i in range(T)])
    sim_arm = np.zeros((T, 7), dtype=np.float64)
    sim_hand = np.zeros((T, 16), dtype=np.float64)

    isaac_env = rfs_env.env

    def policy_obs_sim(i: int) -> dict:
        nw = num_warmup_steps if i == 0 else 1
        if i == 0:
            isaac_env.reset()
        obs = reset_to_real_joints(
            isaac_env,
            traj_obs[i],
            num_warmup_steps=nw,
            hold_from_obs=True,
            action_type=args_cli.action_type,
        )
        if obs is None:
            raise ValueError(f"reset_to_real_joints failed at step {i}")
        sim_ex = _extract_sim_observation(obs)
        if sim_ex is None:
            raise ValueError(f"_extract_sim_observation failed at step {i}")
        sim_arm[i] = sim_ex["joint_positions"]
        sim_hand[i] = sim_ex["gripper_position"]
        return obs["policy"]

    actions_rfs_sim = _run_rfs_open_loop_pass(rfs_env, model, device, T, policy_obs_sim)

    _plot_actions(
        os.path.join(output_dir, "actions_open_loop.png"),
        actions_traj,
        actions_rfs_real,
        actions_rfs_sim,
    )
    _plot_joints(
        os.path.join(output_dir, "joints_open_loop.png"),
        real_arm,
        real_hand,
        sim_arm,
        sim_hand,
        "RFS open-loop",
    )

    meta = {
        "trajectory_file": os.path.abspath(args_cli.trajectory_file),
        "diffusion_path": os.path.abspath(diffusion_path),
        "rfs_checkpoint": os.path.abspath(args_cli.rfs_checkpoint) if args_cli.rfs_checkpoint else None,
        "T": T,
        "cfm_action_dim": rfs_env.cfm_action_dim,
        "residual_step": rfs_env.residual_step,
        "obs_keys": obs_keys,
        "asymmetric_ac": asymmetric_ac,
    }
    with open(os.path.join(output_dir, "open_loop_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[play_rfs_open_loop] Saved plots under {output_dir}")
    rfs_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
