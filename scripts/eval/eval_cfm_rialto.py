"""
Evaluate a trained CFM PCD policy in simulation using the PPOEvalCallback infrastructure.

Produces the same outputs as a PPO eval run: scatter/success plots, metrics JSON,
and optional video.

Usage (inside container):
    ./uwlab.sh -p scripts/eval/eval_cfm_rialto.py \\
        --task UW-FrankaLeap-GraspBottleRandomResets-JointAbs-PPO-Collect-v0 \\
        --checkpoint logs/rialto/distilled_policies/cfm_pcd_bottle_grasp_rl_0519_... \\
        --num_envs 256 --num_trials 20 --headless

With video (requires --enable_cameras):
    ./uwlab.sh -p scripts/eval/eval_cfm_rialto.py \\
        --task ... --checkpoint ... --num_envs 8 --num_trials 5 \\
        --record_video --enable_cameras --headless
"""

import argparse
import os

from uwlab.utils.paths import setup_third_party_paths
setup_third_party_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a CFM PCD policy in Isaac Sim.")
parser.add_argument("--task", type=str, required=True,
                    help="Isaac task ID, e.g. UW-FrankaLeap-GraspBottleRandomResets-JointAbs-PPO-Collect-v0")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to CFM checkpoint directory (contains checkpoints/best.ckpt or latest.ckpt).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_trials", type=int, default=20,
                    help="Number of random-reset eval episodes.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Override output directory. Default: <checkpoint_dir>/eval/")
parser.add_argument("--record_video", action="store_true", default=False,
                    help="Record MP4 video (also requires --enable_cameras).")
parser.add_argument("--spawn", type=str, default=None,
                    help="Spawn config name from configs/eval/spawns/ (default: random resets).")
parser.add_argument("--success_key", type=str, default="is_success",
                    help="Metric key to use as success criterion.")
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.record_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── All Isaac-dependent imports after AppLauncher ──────────────────────────

import gymnasium as gym
import torch
import wandb
import yaml

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from uwlab.eval.bc_obs_formatter import BCObsFormatter
from uwlab.eval.spawn import SpawnCfg, load_spawn_cfg
from uwlab.policy.backbone.pcd.pointnet import PointNet
from uwlab.policy.backbone.multi_pcd_obs_encoder import MultiPCDObsEncoder
from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy

import uwlab_tasks  # noqa: F401  registers gym envs
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap import (
    RL_MODE,
    EVAL_MODE,
    parse_franka_leap_env_cfg,
)

# ppo_eval_callback.py has no package __init__, so load by file path
import importlib.util as _ilu
_cb_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "reinforcement_learning", "sb3", "ppo_eval_callback.py",
)
_cb_spec = _ilu.spec_from_file_location("ppo_eval_callback", _cb_path)
_cb_mod = _ilu.module_from_spec(_cb_spec)
_cb_spec.loader.exec_module(_cb_mod)
PPOEvalCallback = _cb_mod.PPOEvalCallback


# ── CFM checkpoint helpers (mirrors play_bc.py) ────────────────────────────

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


def _load_train_cfg(checkpoint_dir: str) -> dict:
    for rel in (".hydra/config.yaml", "config.yaml"):
        path = os.path.join(checkpoint_dir, rel)
        if os.path.isfile(path):
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Training config not found in {checkpoint_dir}")


def load_cfm_policy(ckpt: dict, train_cfg: dict, device: torch.device) -> CFMPCDPolicy:
    ds_cfg = train_cfg["dataset"]
    pol_cfg = dict(train_cfg["policy"])
    pn_cfg = train_cfg["pointnet"]

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


# ── CFM policy wrapper ─────────────────────────────────────────────────────

class CFMPolicyWrapper:
    """Wraps CFMPCDPolicy with a predict() interface for PPOEvalCallback.

    Manages BCObsFormatter (rolling obs history) and action-chunk caching so
    that CFM inference only runs every `chunk_len` steps instead of every step.
    """

    def __init__(self, policy, obs_keys, image_keys, downsample_points, device, n_obs_steps):
        self.policy = policy
        self.formatter = BCObsFormatter(
            obs_keys=obs_keys,
            image_keys=image_keys,
            downsample_points=downsample_points,
            device=device,
            n_obs_steps=n_obs_steps,
            action_dim=policy.action_dim if policy.use_action_history else 0,
        )
        # Re-run inference every chunk_len steps (capped by horizon length)
        self._chunk_len = min(policy.n_action_steps, policy.horizon)
        self._action_cache = None  # (B, horizon, A) tensor
        self._step = 0

    def reset_episode(self):
        self.formatter.reset()
        self._action_cache = None
        self._step = 0

    def predict(self, policy_obs_dict: dict, deterministic: bool = False):
        """Accept raw policy obs dict from _extract_obs; return (actions_np, None)."""
        if self._step % self._chunk_len == 0:
            obs_dict = self.formatter.format(policy_obs_dict)
            with torch.no_grad():
                result = self.policy.predict_action(obs_dict)
            self._action_cache = result["action_pred"]  # (B, horizon, A)

        action = self._action_cache[:, self._step % self._chunk_len]  # (B, A)
        self.formatter.update_action(action)
        self._step += 1
        return action.cpu().numpy(), None


# ── CFM eval callback ──────────────────────────────────────────────────────

class CFMEvalCallback(PPOEvalCallback):
    """PPOEvalCallback subclass wired to a CFMPolicyWrapper instead of an SB3 model.

    Overrides:
    - _extract_obs: returns raw policy obs dict (not a flat numpy array)
    - _on_episode_reset: resets the obs formatter + action cache
    """

    def _extract_obs(self, obs_dict: dict) -> dict:
        return obs_dict.get("policy", obs_dict)

    def _on_episode_reset(self) -> None:
        self.model.reset_episode()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    checkpoint_dir = os.path.expanduser(args_cli.checkpoint)
    device = torch.device("cuda:0")

    # ── Load CFM policy ────────────────────────────────────────────────────
    ckpt_path = _find_checkpoint(checkpoint_dir)
    print(f"[eval_cfm] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg = ckpt.get("cfg") or _load_train_cfg(checkpoint_dir)

    policy = load_cfm_policy(ckpt=ckpt, train_cfg=train_cfg, device=device)
    policy.eval()

    ds_cfg = train_cfg["dataset"]
    obs_keys = list(ds_cfg["obs_keys"])
    image_keys = list(ds_cfg["image_keys"])
    downsample_points = int(ds_cfg["downsample_points"])
    n_obs_steps = int(train_cfg["n_obs_steps"])

    print(f"[eval_cfm] obs_keys={obs_keys}  image_keys={image_keys}  "
          f"n_obs_steps={n_obs_steps}  chunk_len={min(policy.n_action_steps, policy.horizon)}")

    cfm_wrapper = CFMPolicyWrapper(
        policy=policy,
        obs_keys=obs_keys,
        image_keys=image_keys,
        downsample_points=downsample_points,
        device=device,
        n_obs_steps=n_obs_steps,
    )

    # ── Create Isaac Sim env ───────────────────────────────────────────────
    # RL_MODE: mesh-based seg_pc, no cameras (~25% faster). EVAL_MODE needed for video
    # since RL_MODE sets fixed_camera=None. Mesh-based seg_pc works in both modes.
    run_mode = EVAL_MODE if args_cli.record_video else RL_MODE
    env_cfg = parse_franka_leap_env_cfg(
        args_cli.task,
        run_mode=run_mode,
        device=str(device),
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed

    if run_mode == EVAL_MODE:
        # Disable train_camera (used for rendered seg_pc in DISTILL_MODE, not needed here)
        env_cfg.scene.train_camera = None
        if hasattr(env_cfg.events, "reset_camera"):
            env_cfg.events.reset_camera = None
    else:
        # RL_MODE: also disable fixed_camera (already None, but be explicit)
        env_cfg.scene.fixed_camera = None
        if hasattr(env_cfg.events, "reset_fixed_camera"):
            env_cfg.events.reset_fixed_camera = None

    render_mode = "rgb_array" if args_cli.record_video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    isaac_env = env.unwrapped

    # ── Spawn config ───────────────────────────────────────────────────────
    if args_cli.spawn is not None:
        spawn_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "eval", "spawns",
        )
        spawn_cfg = load_spawn_cfg(args_cli.spawn, spawn_dir)
    else:
        spawn_cfg = SpawnCfg(poses=[], num_trials=args_cli.num_trials)

    # ── Output dir ─────────────────────────────────────────────────────────
    if args_cli.output_dir:
        log_dir = args_cli.output_dir
    else:
        log_dir = os.path.join(checkpoint_dir, "eval")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[eval_cfm] Output dir: {log_dir}")

    # ── Wandb ──────────────────────────────────────────────────────────────
    if args_cli.wandb_project:
        run_name = args_cli.wandb_run_name or os.path.basename(checkpoint_dir.rstrip("/"))
        wandb.init(project=args_cli.wandb_project, name=run_name, config={
            "task": args_cli.task,
            "checkpoint": checkpoint_dir,
            "num_envs": args_cli.num_envs,
            "num_trials": args_cli.num_trials,
        })

    # ── Build callback and run eval ────────────────────────────────────────
    callback = CFMEvalCallback(
        isaac_env=isaac_env,
        gym_env=env,
        spawn_cfg=spawn_cfg,
        log_dir=log_dir,
        eval_interval=1,
        record_scatter=True,
        record_video=args_cli.record_video,
        verbose=1,
    )
    callback.model = cfm_wrapper

    # Gymnasium's OrderEnforcer requires env.reset() before render(). The callback
    # resets via isaac_env.reset() directly, bypassing the wrapper. One gym reset
    # sets _has_reset=True permanently so render() works for the whole eval run.
    if args_cli.record_video:
        env.reset()

    with torch.inference_mode():
        callback._run_eval()

    if args_cli.wandb_project:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
