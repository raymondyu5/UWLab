"""
Open-loop BC evaluation. Action plots include three series per dimension:
  (1) trajectory / training-target actions (``dataset.action_key``, from recorded obs);
  (2) policy output given observations from the recorded trajectory only;
  (3) policy output given sim observations after resetting to each recorded state.

Usage:
    ./uwlab.sh -p scripts/eval/play_bc_open_loop.py --eval_cfg configs/eval/bottle_pour_bc_jointabs.yaml --trajectory_file data_storage/datasets/03_24_bourbon_pour/episode_68/episode_68.zarr checkpoint=logs/bc_cfm_pcd_bourbon_0324_absjoint_h16_hist4_extnoise  --headless --enable_cameras

Does not import play_bc.py or open_loop_pc_overlay.py (those launch AppLauncher at import time).
"""

import argparse
import os

from uwlab.utils.paths import setup_third_party_paths

setup_third_party_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BC open-loop eval on recorded trajectory vs sim-reset states.")
parser.add_argument("--eval_cfg", type=str, required=True, help="Path to eval config YAML (configs/eval/*.yaml)")
parser.add_argument(
    "--trajectory_file",
    type=str,
    required=True,
    help="Episode zarr or directory (see uwlab_tasks.utils.trajectory_utils.load_real_episode).",
)
parser.add_argument(
    "--action_type",
    type=str,
    choices=["delta_ee", "abs_ee", "joint", "delta_joint"],
    default="joint",
    help="Hold / reset action type for reset_to_real_joints (match training controller).",
)
parser.add_argument("--sim_type", type=str, choices=["eval", "distill"], default="eval")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default=None, help="Plots and logs; default under checkpoint eval dir.")
parser.add_argument(
    "--sim_pc_downsample",
    type=int,
    default=None,
    help="Override env seg_pc downsample (default: dataset downsample_points).",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--camera", type=str, default="fixed_camera", help="Distill camera name when sim_type=distill.")
parser.add_argument("overrides", nargs="*", help="Key=value overrides for eval config (e.g. checkpoint=/path)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from isaaclab.managers import SceneEntityCfg
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

import uwlab_tasks  # noqa: F401
from uwlab.eval.bc_obs_formatter import BCObsFormatter
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy
from uwlab.utils.checkpoint import extract_ckpt_metadata, format_ckpt_metadata
import uwlab_assets.robots.franka_leap as franka_leap
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import (
    ARM_RESET,
    DISTILL_MODE,
    EVAL_MODE,
    HAND_RESET,
    parse_franka_leap_env_cfg,
)
from uwlab_tasks.manager_based.manipulation.grasp.mdp.events import reset_robot_joints
from uwlab_tasks.utils.trajectory_utils import load_real_episode


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
    raise FileNotFoundError(
        f"Training config not found in {checkpoint_dir}. Expected .hydra/config.yaml or config.yaml"
    )


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


def load_cfm_policy(
    ckpt_path: str | None = None,
    *,
    ckpt: dict | None = None,
    train_cfg: dict | None = None,
    device: str | torch.device = "cpu",
    num_inference_steps: int | None = None,
) -> CFMPCDPolicy:
    if ckpt is None:
        assert ckpt_path is not None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if train_cfg is None:
        train_cfg = ckpt.get("cfg")
    if train_cfg is None:
        raise ValueError("No cfg in checkpoint and no train_cfg supplied.")

    ds_cfg = train_cfg["dataset"]
    pol_cfg = dict(train_cfg["policy"])
    pn_cfg = train_cfg["pointnet"]
    if num_inference_steps is not None:
        pol_cfg["num_inference_steps"] = int(num_inference_steps)

    sd = ckpt["ema_model"]
    low_obs_dim = sd["normalizer.params_dict.agent_pos.scale"].shape[0]
    action_dim = sd["normalizer.params_dict.action.scale"].shape[0]
    use_action_history = "normalizer.params_dict.past_actions.scale" in sd

    shape_meta = {
        "action": {"shape": [action_dim]},
        "obs": {"agent_pos": {"shape": [low_obs_dim], "type": "low_dim"}},
    }
    for key in list(ds_cfg["image_keys"]):
        shape_meta["obs"][key] = {"shape": [3, int(ds_cfg["downsample_points"])], "type": "pcd"}

    pcd_model = PointNet(
        in_channels=int(pn_cfg["in_channels"]),
        local_channels=tuple(pn_cfg["local_channels"]),
        global_channels=tuple(pn_cfg["global_channels"]),
        use_bn=bool(pn_cfg["use_bn"]),
    )
    obs_encoder = MultiPCDObsEncoder(shape_meta=shape_meta, pcd_model=pcd_model)

    policy = CFMPCDPolicy(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        noise_scheduler=ConditionalFlowMatcher(sigma=float(pol_cfg["sigma"])),
        horizon=int(train_cfg["horizon"]),
        n_action_steps=int(train_cfg["n_action_steps"]),
        n_obs_steps=int(train_cfg["n_obs_steps"]),
        num_inference_steps=int(pol_cfg["num_inference_steps"]),
        diffusion_step_embed_dim=int(pol_cfg["diffusion_step_embed_dim"]),
        down_dims=tuple(pol_cfg["down_dims"]),
        kernel_size=int(pol_cfg["kernel_size"]),
        n_groups=int(pol_cfg["n_groups"]),
        cond_predict_scale=bool(pol_cfg["cond_predict_scale"]),
        use_action_history=use_action_history,
    )
    policy.load_state_dict(sd)
    policy.to(device).eval()
    return policy


def _build_policy(
    train_cfg: dict,
    checkpoint_dir: str,
    device: torch.device,
    eval_overrides: dict | None = None,
    ckpt: dict | None = None,
):
    ds_cfg = train_cfg["dataset"]
    num_inference_steps = None
    if eval_overrides and "num_inference_steps" in eval_overrides:
        num_inference_steps = int(eval_overrides["num_inference_steps"])

    if ckpt is None:
        ckpt_path = _find_checkpoint(checkpoint_dir, eval_overrides and eval_overrides.get("checkpoint_name"))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    policy = load_cfm_policy(
        ckpt=ckpt, train_cfg=train_cfg, device=device, num_inference_steps=num_inference_steps
    )
    return policy, ds_cfg


# --- Copied from open_loop_pc_overlay (cannot import that module; it launches the app). ---


def _extract_real_joints(obs: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def reset_to_real_joints(
    env,
    first_real_obs,
    num_warmup_steps: int = 10,
    hold_from_obs: bool = False,
    action_type: str = "joint",
):
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


def _extract_sim_observation(obs: dict) -> dict | None:
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
    """7D arm joints: Isaac uses key ``arm_joint_pos``; zarr episodes use ``joint_positions``."""
    if "arm_joint_pos" in traj_obs_t:
        return np.asarray(traj_obs_t["arm_joint_pos"], dtype=np.float32).reshape(-1)[:7]
    if "joint_positions" in traj_obs_t:
        return np.asarray(traj_obs_t["joint_positions"], dtype=np.float32).reshape(-1)[:7]
    raise KeyError(
        "Trajectory obs needs arm joints under 'arm_joint_pos' or 'joint_positions' (zarr load_real_episode)."
    )


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


def policy_obs_from_traj_step(
    traj_obs_t: dict,
    device: torch.device,
    obs_keys: list[str],
    image_keys: list[str],
) -> dict:
    """Build the same keys as ``obs['policy']`` / ``BCObsFormatter`` expects.

    Training names (``obs_keys``) match Isaac observation terms, e.g. ``arm_joint_pos``.
    Zarr ``load_real_episode`` stores the same values under ``joint_positions``; we map here.
    """
    needed = set(obs_keys) | set(image_keys)
    out: dict = {}

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
            f"policy_obs_from_traj_step cannot build keys {sorted(missing)}; "
            f"add mapping in play_bc_open_loop.py. Built: {sorted(out.keys())}"
        )
    return out


def _concat_agent_pos(policy_obs: dict, obs_keys: list[str], device: torch.device) -> torch.Tensor:
    parts = []
    for k in obs_keys:
        val = policy_obs[k]
        if isinstance(val, torch.Tensor):
            parts.append(val.float())
        else:
            parts.append(torch.from_numpy(np.asarray(val)).to(device).float())
    return torch.cat(parts, dim=-1)


def _training_action_vector_numpy(
    traj_obs_t: dict,
    train_cfg: dict,
    action_dim: int,
    raw_actions_row,
) -> np.ndarray:
    """Ground-truth action vector aligned with BC training (``dataset.action_key`` / ZarrDataset).

    Built from the trajectory observation dict (commanded joints, ``hand_action``, etc.), not the
    policy. Used for past-action history and for plotting ``trajectory`` vs policy outputs.
    """
    ds = train_cfg.get("dataset") or {}
    ak = ds.get("action_key", "actions")
    if hasattr(ak, "__iter__") and not isinstance(ak, str):
        keys = list(ak)
    else:
        keys = [ak]

    if len(keys) == 1 and keys[0] == "actions":
        a = np.asarray(raw_actions_row, dtype=np.float32).reshape(-1)
        if a.shape[0] != action_dim:
            raise ValueError(
                f"dataset.action_key is 'actions' but stored row has dim {a.shape[0]}, "
                f"policy action_dim is {action_dim}. Set action_key to zarr columns used at train time "
                f"(e.g. [arm_joint_pos_target, hand_action]) so past actions match training."
            )
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
        raise KeyError(
            f"No trajectory field for action segment '{zkey}'. Keys: {list(traj_obs_t.keys())}"
        )

    parts = [segment_for_zarr_key(str(k)) for k in keys]
    a = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
    if a.shape[0] != action_dim:
        raise ValueError(
            f"Action from dataset.action_key {keys} has dim {a.shape[0]}, expected policy action_dim {action_dim}"
        )
    return a.astype(np.float64)


def _past_action_tensor_from_traj_obs(
    traj_obs_t: dict,
    train_cfg: dict,
    action_dim: int,
    device: torch.device,
    raw_actions_row,
) -> torch.Tensor:
    a = _training_action_vector_numpy(traj_obs_t, train_cfg, action_dim, raw_actions_row)
    return torch.from_numpy(a.astype(np.float32)).unsqueeze(0).to(device)


def _decode_action_step(
    policy: CFMPCDPolicy,
    policy_obs: dict,
    obs_keys: list[str],
    device: torch.device,
    chunk_relative: bool,
    action_seq: torch.Tensor,
) -> torch.Tensor:
    action_idx = 0
    if action_idx >= action_seq.shape[1]:
        action_idx = action_seq.shape[1] - 1
    if chunk_relative:
        chunk_start_obs = _concat_agent_pos(policy_obs, obs_keys, device)
        current_obs = chunk_start_obs
        action_step = (chunk_start_obs + action_seq[:, action_idx] - current_obs).clone()
    else:
        action_step = action_seq[:, action_idx].clone()
    return action_step


def _run_pass(
    policy: CFMPCDPolicy,
    formatter: BCObsFormatter,
    traj_obs: list,
    traj_actions: list,
    train_cfg: dict,
    obs_keys: list[str],
    device: torch.device,
    chunk_relative: bool,
    policy_obs_fn,
):
    """policy_obs_fn(i) -> policy obs dict for step i.

    Past-action history uses the same action vector as training (``dataset.action_key``), built from
    each ``traj_obs[i]`` (commanded joints / hand_action), not the raw ``actions`` zarr row when dims differ.
    """
    T = len(traj_obs)
    action_dim = policy.action_dim
    out = np.zeros((T, action_dim), dtype=np.float64)

    formatter.reset()
    with torch.inference_mode():
        for i in range(T):
            policy_obs = policy_obs_fn(i)

            obs_dict = formatter.format(policy_obs)
            result = policy.predict_action(obs_dict)
            action_seq = result["action_pred"]
            action_step = _decode_action_step(
                policy, policy_obs, obs_keys, device, chunk_relative, action_seq
            )
            out[i] = action_step[0].detach().cpu().numpy()

            past_a = _past_action_tensor_from_traj_obs(
                traj_obs[i], train_cfg, action_dim, device, traj_actions[i]
            )
            formatter.update_action(past_a)
    return out


def _extract_real_arm_hand(traj_obs_t: dict) -> tuple[np.ndarray, np.ndarray]:
    arm = np.asarray(traj_obs_t["joint_positions"], dtype=np.float64).reshape(-1)[:7]
    hand = np.asarray(traj_obs_t["gripper_position"], dtype=np.float64).reshape(-1)[:16]
    return arm, hand


def _plot_actions(
    output_path: str,
    actions_traj: np.ndarray,
    actions_policy_real_obs: np.ndarray,
    actions_policy_sim_reset: np.ndarray,
):
    """Plot three series per dimension: recorded trajectory targets, policy @ real obs, policy @ sim."""
    T, D = actions_policy_real_obs.shape
    assert actions_traj.shape == (T, D) and actions_policy_sim_reset.shape == (T, D)
    n_cols = min(4, D)
    n_rows = int(np.ceil(D / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 2.8 * n_rows), squeeze=False)
    t = np.arange(T)
    for d in range(D):
        r, c = divmod(d, n_cols)
        ax = axes[r][c]
        ax.plot(t, actions_traj[:, d], label="trajectory (train target)", linewidth=1.5, color="#2ca02c")
        ax.plot(t, actions_policy_real_obs[:, d], label="policy | real obs", linewidth=1.4, color="#1f77b4")
        ax.plot(
            t,
            actions_policy_sim_reset[:, d],
            label="policy | sim reset",
            linewidth=1.4,
            alpha=0.9,
            color="#ff7f0e",
        )
        ax.set_title(f"action[{d}]")
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7, loc="best")
    for d in range(D, n_rows * n_cols):
        r, c = divmod(d, n_cols)
        axes[r][c].axis("off")
    fig.suptitle("Actions: trajectory (dataset.action_key) vs policy on real vs sim observations")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_joints(
    output_path: str,
    real_arm: np.ndarray,
    real_hand: np.ndarray,
    sim_arm: np.ndarray,
    sim_hand: np.ndarray,
    title_prefix: str,
):
    T = real_arm.shape[0]
    t = np.arange(T)
    D = real_arm.shape[1] + real_hand.shape[1]
    n_cols = min(4, D)
    n_rows = int(np.ceil(D / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 2.5 * n_rows), squeeze=False)
    idx = 0
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

    checkpoint_dir = eval_cfg.get("checkpoint")
    if checkpoint_dir is None:
        raise ValueError("eval_cfg must specify 'checkpoint'")
    checkpoint_dir = os.path.expanduser(checkpoint_dir)

    device = torch.device(eval_cfg.get("device", "cuda:0"))

    ckpt_path = _find_checkpoint(checkpoint_dir, eval_cfg.get("checkpoint_name"))
    print(f"[play_bc_open_loop] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_meta = extract_ckpt_metadata(ckpt)
    print(f"[play_bc_open_loop] Checkpoint metadata: {format_ckpt_metadata(ckpt_meta)}")
    train_cfg = ckpt.get("cfg") or _load_train_cfg(checkpoint_dir)
    chunk_relative = bool((train_cfg.get("dataset") or {}).get("chunk_relative", False))
    policy, ds_cfg = _build_policy(train_cfg, checkpoint_dir, device, eval_overrides=eval_cfg, ckpt=ckpt)

    obs_keys = list(eval_cfg.get("obs_keys") or ds_cfg["obs_keys"])
    image_keys = list(ds_cfg["image_keys"])
    downsample_points = int(ds_cfg["downsample_points"])
    n_obs_steps = int(train_cfg["n_obs_steps"])
    
    formatter = BCObsFormatter(
        obs_keys,
        image_keys,
        downsample_points,
        device,
        n_obs_steps=n_obs_steps,
        action_dim=policy.action_dim if policy.use_action_history else 0,
    )

    print(
        f"[play_bc_open_loop] chunk_relative={chunk_relative}, obs_keys={obs_keys}, "
        f"image_keys={image_keys}, n_obs_steps={n_obs_steps}"
    )

    episode = load_real_episode(args_cli.trajectory_file)
    traj_obs = episode["obs"]
    traj_actions = episode["actions"]
    T = min(len(traj_obs), len(traj_actions))
    if T == 0:
        raise ValueError("Empty trajectory")
    traj_obs = traj_obs[:T]
    traj_actions = traj_actions[:T]

    task_id = eval_cfg["task_id"]
    run_mode = EVAL_MODE if args_cli.sim_type == "eval" else DISTILL_MODE
    env_cfg = parse_franka_leap_env_cfg(
        task_id,
        run_mode=run_mode,
        device=str(device),
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
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
    if hasattr(env_cfg.terminations, "bottle_dropped"):
        env_cfg.terminations.bottle_dropped = None
    if hasattr(env_cfg.terminations, "bottle_too_far"):
        env_cfg.terminations.bottle_too_far = None
    if hasattr(env_cfg.terminations, "cup_toppled"):
        env_cfg.terminations.cup_toppled = None

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

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)

    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        eval_base = os.path.join(os.path.dirname(checkpoint_dir.rstrip("/")), "eval")
        eval_config_stem = os.path.splitext(os.path.basename(args_cli.eval_cfg))[0]
        checkpoint_basename = os.path.basename(checkpoint_dir.rstrip("/"))
        output_dir = os.path.join(eval_base, eval_config_stem, checkpoint_basename, "open_loop")
    os.makedirs(output_dir, exist_ok=True)

    action_dim = policy.action_dim
    actions_traj = np.stack(
        [
            _training_action_vector_numpy(traj_obs[i], train_cfg, action_dim, traj_actions[i])
            for i in range(T)
        ]
    )

    # Pass A: policy on logged trajectory observations only (no sim)
    def policy_obs_real(i: int) -> dict:
        return policy_obs_from_traj_step(traj_obs[i], device, obs_keys, image_keys)

    actions_policy_real_obs = _run_pass(
        policy,
        formatter,
        traj_obs,
        traj_actions,
        train_cfg,
        obs_keys,
        device,
        chunk_relative,
        policy_obs_real,
    )

    real_arm = np.stack([_extract_real_arm_hand(traj_obs[i])[0] for i in range(T)])
    real_hand = np.stack([_extract_real_arm_hand(traj_obs[i])[1] for i in range(T)])
    sim_arm = np.zeros((T, 7), dtype=np.float64)
    sim_hand = np.zeros((T, 16), dtype=np.float64)

    def policy_obs_sim(i: int) -> dict:
        nw = num_warmup_steps if i == 0 else 1
        if i == 0:
            env.reset()
        obs = reset_to_real_joints(
            env,
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

    actions_policy_sim_reset = _run_pass(
        policy,
        formatter,
        traj_obs,
        traj_actions,
        train_cfg,
        obs_keys,
        device,
        chunk_relative,
        policy_obs_sim,
    )

    actions_path = os.path.join(output_dir, "actions_open_loop.png")
    _plot_actions(actions_path, actions_traj, actions_policy_real_obs, actions_policy_sim_reset)
    joints_path = os.path.join(output_dir, "joints_open_loop.png")
    _plot_joints(joints_path, real_arm, real_hand, sim_arm, sim_hand, "BC open-loop")

    meta = {
        "trajectory_file": os.path.abspath(args_cli.trajectory_file),
        "T": T,
        "chunk_relative": chunk_relative,
        "obs_keys": obs_keys,
        "action_dim": action_dim,
        "plot_actions": {
            "trajectory_train_target": "Per-timestep vector from dataset.action_key (same as BC labels / past-action buffer).",
            "policy_real_obs": "Policy decode at each step using observations from the recorded trajectory only.",
            "policy_sim_reset": "Policy decode after resetting sim to each recorded robot state.",
        },
    }
    with open(os.path.join(output_dir, "open_loop_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[play_bc_open_loop] Saved {actions_path}")
    print(f"[play_bc_open_loop] Saved {joints_path}")
    print(f"[play_bc_open_loop] Done. output_dir={output_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
