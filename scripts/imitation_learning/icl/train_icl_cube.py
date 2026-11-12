"""Episodic in-context learning on the Franka-LEAP cube grasp task, with the CFM point-cloud BC policy as the expert.

Port of ManiSkill's icl_episodic.py (docs: docs/claude/implementations/icl-cube.md). The hidden task variable is the cube's xy position (fixed
across the k+1 episodes of a task). The student only sees proprioception (arm + hand joint positions, previous action, time) - no cube, no point
cloud. After each episode it gets privileged feedback relating the cube to the gripper; its context for the next episode is the previous episodes'
(state, action) steps plus these feedback tokens. The expert (CFM policy, sees the point cloud and so the cube) labels every state the student
visits (DAgger). Episodes are "terminated" at the first success (cube lifted): later steps are masked out of the loss/context/feedback.

Each DAgger iteration: sample N cube positions; roll out the student for k episodes with growing context (iteration 0 and a fraction of later
episodes are driven by an expert that is told the cube is at the default position, so the context holds realistic failures); roll out the expert
(true cube) for one episode; label all visited states with the expert's action; the expert episode is paired with every context size h = 0..k.
Evaluation: fresh fixed cube positions, student only, k+1 episodes; controls with shuffled / no feedback / no context.

Usage (inside the container):
    ./uwlab.sh -p scripts/imitation_learning/icl/train_icl_cube.py --exp_name icl_cube --headless
    ./uwlab.sh -p scripts/imitation_learning/icl/train_icl_cube.py --refs_only --headless   # expert reference levels only
"""
import argparse
import json
from collections import deque
import os
import sys
import time

from uwlab.utils.paths import setup_third_party_paths

setup_third_party_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ICL student on the cube grasp task.")
parser.add_argument("--task_id", type=str, default="UW-FrankaLeap-GraspCube-JointAbs-v0")
parser.add_argument("--expert", type=str,
                    default="logs/base/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast", help="CFM (base diffusion) checkpoint dir")
parser.add_argument("--rfs_checkpoint", type=str, default="",
                    help="SB3 PPO .zip of an RFS (noise+residual) policy on top of --expert (the base diffusion dir). If set, the expert is the RFS policy "
                         "queried every step; if empty, the plain CFM policy.")
parser.add_argument("--rfs_cfg", type=str, default="configs/rl/arm_rfs_joint_cfg.yaml")
parser.add_argument("--exp_name", type=str, default="icl_cube")
parser.add_argument("--out_dir", type=str, default="")
parser.add_argument("--num_envs", type=int, default=128, help="tasks (cube positions) per iteration / evaluation")
parser.add_argument("--demos_per_iter", type=int, default=0,
                    help="tasks (each = k student episodes + 1 expert relabel episode) added to the dataset per iteration; 0 -> num_envs. "
                         "Rollouts still run on num_envs envs (collected in ceil(n/num_envs) passes, extras dropped) so eval is unchanged.")
parser.add_argument("--bc_full_info", action="store_true",
                    help="baseline: plain BC on expert-only data (num_envs expert episodes per iteration, no student rollouts, no context) with the true cube "
                         "position appended to the student's state; evaluated as a single episode per task")
parser.add_argument("--bc_pcd", action="store_true",
                    help="baseline: like --bc_full_info (expert-only BC, no context) but the student sees the episode's CLEAN initial point cloud (alpha=0) instead of the "
                         "cube position")
parser.add_argument("--study", type=str, default="", choices=["", "collect", "bc_converge", "bc_ext", "icl"],
                    help="checkerboard step-budget study (docs/claude/experiments/2026-11-03-icl-checker-steps-study.md; needs --checker). collect: expert episodes on "
                         "--ext_region -> --cache_out (use --bc_pcd). bc_converge: one BC policy on D_0, trained until the held-out loss stops improving, evaluated once (use --bc_pcd). "
                         "bc_ext / icl: start from the fixed pi_bc (--relabeler); every iteration adds num_envs * k episodes (bc_ext: expert episodes from --ext_pool, use --bc_pcd; "
                         "icl: own noisy-cloud rollouts relabeled by the fixed pi_bc on the clean cloud, use --vision) and trains to held-out-loss convergence on batches that are half "
                         "D_0, half new data. Env steps per iteration = num_envs * k * T for every method.")
parser.add_argument("--ext_region", type=str, default="white", choices=["white", "black"], help="--study collect: region of the sampled tasks")
parser.add_argument("--val_frac", type=float, default=0.05, help="--study bc_converge: fraction of D_0 trajectories held out to detect convergence")
parser.add_argument("--max_steps", type=int, default=100000, help="--study bc_converge: cap on gradient steps")
parser.add_argument("--min_steps", type=int, default=0, help="--study icl/bc_ext: train at least this many steps per iteration; the best-held-out-loss weights are "
                    "chosen among evals at or after it (so a short-lived early minimum cannot cut training short)")
parser.add_argument("--d0_clean", action="store_true", help="--study icl: D_0 half-batch uses the clean cloud (alpha=0) instead of random alpha")
parser.add_argument("--dense_fb", action="store_true", help="--study icl: dense correction: every context step token also carries (ee - current cube position) at that step (3 dims); the per-episode feedback token is zeroed")
parser.add_argument("--patience", type=int, default=20, help="--study icl/bc_ext: stop after this many evals (500 steps each) without a 1%% held-out loss improvement")
parser.add_argument("--n_trajs", type=int, default=2000, help="--study collect: expert episodes to store")
parser.add_argument("--cache_out", type=str, default="", help="--study collect: output .pt")
parser.add_argument("--d0_cache", type=str, default="logs/icl/study/d0_white.pt", help="D_0 cache written by --study collect")
parser.add_argument("--ext_pool", type=str, default="", help="--study bc_ext: expert pool cache (white or black) written by --study collect")
parser.add_argument("--relabeler", type=str, default="", help="--study icl: policy_bc_fixed.pt of a --study bc_converge run (the frozen pi_bc)")
parser.add_argument("--k", type=int, default=2, help="student episodes per task: context sizes h = 0..k")
parser.add_argument("--num_iterations", type=int, default=30)
parser.add_argument("--cube_range", type=float, default=0.2, help="cube xy ~ U(-r, r) around the default spawn (the env's own range)")
parser.add_argument("--x_min", type=float, default=None,
                    help="lower bound of the cube x (env frame) for all sampled tasks; default default_x - cube_range. The workspace becomes the rectangle "
                         "[x_min, default_x + cube_range] x [-cube_range, cube_range] (the expert fails for x < ~0.4)")
parser.add_argument("--checker", action="store_true",
                    help="train/test split: the workspace (cube_range square) is a --checker_cells x --checker_cells checkerboard; training cube positions come from white "
                         "squares only, and eval is reported separately on white (in distribution, 'id') and black (out of distribution, 'ood') squares")
parser.add_argument("--checker_cells", type=int, default=4, help="cells per side of the checkerboard")
parser.add_argument("--checker_margin", type=float, default=0.0, help="sampled points are kept this far (m) from every cell edge")
parser.add_argument("--eval_tasks", type=int, default=256, help="fixed eval tasks per region (multiple of --num_envs)")
parser.add_argument("--feedback", type=str, default="init_cube", choices=["init_cube", "final_cube", "success", "language_tokens"],
                    help="init_cube: [initial cube - final gripper, final gripper] (3+3); final_cube: same with the final cube position; "
                         "success: 0/1 flag; language_tokens: the init_cube displacement as a templated text command (\"move the hand 12.3 cm forward and "
                         "4.5 cm left and 0.7 cm down\") -> frozen flan-t5 encoder -> one feature vector per word piece")
parser.add_argument("--token_encoder", type=str, default="google/flan-t5-small", help="frozen encoder for --feedback language_tokens (offline, from $HF_HOME)")
parser.add_argument("--tok_len", type=int, default=40, help="word pieces per command (padded / truncated)")
parser.add_argument("--vision", action="store_true",
                    help="student also gets the episode's initial point cloud, blended with gaussian noise: (1-alpha)*pcd + alpha*noise, alpha~U(0,1) "
                         "drawn independently per episode (the student is not told alpha)")
parser.add_argument("--ctx_stride", type=int, default=8, help="context tokens: every ctx_stride-th step of a past episode")
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--grad_steps", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--prior_expert_frac", type=float, default=0.25,
                    help="fraction of student-episode slots (after iteration 0) driven by an expert told the default cube position")
parser.add_argument("--num_inference_steps", type=int, default=None, help="override the expert's CFM ODE steps (speed)")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--eval_seed", type=int, default=12345)
parser.add_argument("--viz", type=str, default="", help="path to a trained policy_*.pt: render env 0 for h=0 vs h=1 on the same cubes, then exit")
parser.add_argument("--viz_rounds", type=int, default=8)
parser.add_argument("--viz_alphas", type=str, default="0,0.5,1",
                    help="--vision viz: point-cloud noise levels, cycled over the rounds (the same alpha for both episodes of a round)")
parser.add_argument("--alpha_sweep", type=str, default="",
                    help="--vision: path to a trained policy_*.pt; evaluate h=0..k at point-cloud noise alpha = 0, 0.1, ..., 1 and write alpha_sweep.json, then exit")
parser.add_argument("--eval_dump", type=str, default="",
                    help="path to a trained policy_*.pt (pass the same --vision / --bc_pcd / --bc_full_info / --checker flags as its training run): evaluate it on the fixed eval "
                         "tasks and write per-task success to <out_dir>/eval_dump.npz (xy, success per region and h), then exit")
parser.add_argument("--expert_dump", action="store_true",
                    help="run the RFS expert (true cube) --expert_reps times on the fixed eval tasks and write per-task success to <out_dir>/eval_dump.npz, then exit")
parser.add_argument("--expert_reps", type=int, default=3)
parser.add_argument("--pcd_dump", type=str, default="", help="save the clean initial point clouds + cube positions of --num_envs random tasks to this .npz, then exit")
parser.add_argument("--viz_eye", type=str, default="", help="x,y,z camera position relative to the env origin (default: the task's fixed_camera)")
parser.add_argument("--viz_look_at", type=str, default="", help="x,y,z camera target relative to the env origin")
parser.add_argument("--refs_only", action="store_true", help="only measure the expert reference levels, then exit")
parser.add_argument("--debug_checks", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import uwlab_tasks  # noqa: F401  registers gym envs
import numpy as np
import yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO

from uwlab.eval.bc_obs_formatter import BCObsFormatter
from uwlab.policy.cfm_loading import load_cfm_policy_dir
from uwlab.policy.icl_policy import ICLPolicy
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.grasp_franka_leap import EVAL_MODE, RL_MODE, parse_franka_leap_env_cfg
from uwlab_tasks.manager_based.manipulation.grasp.config.franka_leap.tasks.cube import CUBE_SPAWN_POS

_RFS_DIR = os.path.join(os.getcwd(), "scripts/reinforcement_learning/sb3/rfs")
if _RFS_DIR not in sys.path:
    sys.path.insert(0, _RFS_DIR)
from asymmetric_policy import AsymmetricActorCriticPolicy, resolve_asymmetric_ac  # noqa: E402
from wrapper import RFSWrapper  # noqa: E402

DEFAULT_XY = (CUBE_SPAWN_POS[0], CUBE_SPAWN_POS[1])
PRIOR_RADIUS = 0.06  # point-cloud points within this distance of the cube centre are treated as cube points when faking the default position
device = torch.device("cuda:0")
N, K = args.num_envs, args.k

torch.manual_seed(args.seed)

# ---------------------------------------------------------------- env + expert
USE_RFS = bool(args.rfs_checkpoint)
# viz: EVAL_MODE keeps the per-env fixed_camera (RL_MODE removes it); observations are otherwise identical
env_cfg = parse_franka_leap_env_cfg(args.task_id, run_mode=RL_MODE if (USE_RFS and not args.viz) else EVAL_MODE, device=str(device), num_envs=N)
env_cfg.seed = args.seed
if not USE_RFS:
    env_cfg.scene.train_camera = None
if not args.viz:
    env_cfg.scene.fixed_camera = None
for ev in ("reset_camera",) + (() if args.viz else ("reset_fixed_camera",)):
    if hasattr(env_cfg.events, ev):
        setattr(env_cfg.events, ev, None)
if args.viz:
    env_cfg.scene.train_camera = None  # one camera per env is already heavy; only fixed_camera is needed
    env_cfg.events.reset_camera = None
    env_cfg.scene.fixed_camera.data_types = ["rgb"]
    for key, val in (("eye_offset", args.viz_eye), ("look_at_offset", args.viz_look_at)):
        if val:
            env_cfg.events.reset_fixed_camera.params[key] = tuple(float(x) for x in val.split(","))
env = gym.make(args.task_id, cfg=env_cfg)
isaac_env = env.unwrapped
A = isaac_env.action_manager.total_action_dim
obj = isaac_env.scene["grasp_object"]
warmup_act = isaac_env.cfg.warmup_action(isaac_env)

if USE_RFS:
    rfs_yaml = yaml.safe_load(open(args.rfs_cfg))
    rfs_c, ppo_c = rfs_yaml["rfs"], rfs_yaml["ppo"]
    assert rfs_c["residual_step"] == 1 and rfs_c["finger_smooth_alpha"] >= 1.0 and not rfs_c["clip_actions"]  # the stateless per-step path below
    noise_dims = tuple(int(x) for x in str(rfs_c["noise_dims"]).split(":"))
    resid_dims = tuple(int(x) for x in str(rfs_c["residual_dims"]).split(":"))
    ac = resolve_asymmetric_ac(False, rfs_c, args.rfs_checkpoint)
    assert ac, "expert path assumes an asymmetric actor/critic RFS checkpoint"
    rfs_env = RFSWrapper(env, diffusion_path=args.expert, residual_step=1, noise_dims=noise_dims, residual_dims=resid_dims,
                         residual_scale=rfs_c["residual_scale"], clip_actions=False, finger_smooth_alpha=1.0,
                         finger_start_dim=rfs_c.get("finger_start_dim", 6), num_warmup_steps=0, asymmetric_ac=True, gamma=float(ppo_c["gamma"]))
    ppo = PPO.load(args.rfs_checkpoint, env=Sb3VecEnvWrapper(rfs_env), custom_objects={"policy_class": AsymmetricActorCriticPolicy}, print_system_info=False)
    expert, formatter = rfs_env.policy, rfs_env.formatter
    WARMUP = int(rfs_c.get("num_warmup_steps", 0))
    image_key = list(rfs_env.formatter.image_keys)[0]
    print(f"[icl] RFS expert: base={args.expert} ckpt={args.rfs_checkpoint} warmup={WARMUP}", flush=True)
else:
    expert, train_cfg = load_cfm_policy_dir(args.expert, device, args.num_inference_steps)
    ds_cfg = train_cfg["dataset"]
    image_key = list(ds_cfg["image_keys"])[0]
    formatter = BCObsFormatter(list(ds_cfg["obs_keys"]), list(ds_cfg["image_keys"]), int(ds_cfg["downsample_points"]), device,
                               n_obs_steps=int(train_cfg["n_obs_steps"]), action_dim=expert.action_dim if expert.use_action_history else 0)
    WARMUP = 0
if USE_RFS:
    env.reset()
    for _ in range(WARMUP):
        env.step(warmup_act)
    T = int(isaac_env.max_episode_length - isaac_env.episode_length_buf[0].item())  # controlled steps per episode (after the warmup)
else:
    T = isaac_env.max_episode_length - isaac_env.cfg.num_warmup_steps
print(f"[icl] T={T} action_dim={A}", flush=True)

PCD = args.vision or args.bc_pcd  # the student gets the episode's point cloud (noisy with --vision, clean with --bc_pcd)
S = 2 * A + 1 + (3 if args.bc_full_info else 0)  # student obs: joint positions (arm + hand = A), previous action (A), time fraction (+ cube xyz for --bc_full_info)
LANG = args.feedback == "language_tokens"
if LANG:
    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(args.token_encoder)
    lm_enc = T5EncoderModel.from_pretrained(args.token_encoder).to(device).eval()
    for q_ in lm_enc.parameters():
        q_.requires_grad_(False)
    Y = (args.tok_len, lm_enc.config.d_model + 1)
else:
    Y = 1 if args.feedback == "success" else 6
text_rng = np.random.RandomState(args.seed)
Y_SHAPE = Y if isinstance(Y, tuple) else (Y,)
ctx_idx = torch.arange(args.ctx_stride - 1, T, args.ctx_stride, device=device)
if ctx_idx[-1] != T - 1:
    ctx_idx = torch.cat([ctx_idx, torch.tensor([T - 1], device=device)])
Tc = len(ctx_idx)
F_FB = 3 if args.dense_fb else 0  # dense per-step feedback dims appended to the context step tokens


# ---------------------------------------------------------------- task: hidden cube xy
def sample_tasks(n, gen=None, region="all"):
    """(n, 2) cube xy in the env frame, uniform over the workspace rectangle (default: +-cube_range around the default spawn; --x_min raises the lower x
    bound). region 'white' / 'black': only that colour of the --checker_cells x --checker_cells checkerboard over the rectangle (cell (ix, iy) is white iff
    (ix + iy) is even), points at least --checker_margin from every cell edge."""
    lo = torch.tensor([DEFAULT_XY[0] - args.cube_range if args.x_min is None else args.x_min, DEFAULT_XY[1] - args.cube_range])
    width = torch.tensor([DEFAULT_XY[0] + args.cube_range, DEFAULT_XY[1] + args.cube_range]) - lo
    if region == "all":
        xy = lo + torch.rand(n, 2, generator=gen) * width
    else:
        C, par = args.checker_cells, int(region == "black")
        size = width / C
        cells = torch.tensor([(i, j) for i in range(C) for j in range(C) if (i + j) % 2 == par], dtype=torch.float32)
        c = cells[torch.randint(len(cells), (n,), generator=gen)]
        u = args.checker_margin + torch.rand(n, 2, generator=gen) * (size - 2 * args.checker_margin)
        xy = lo + c * size + u
    return xy.to(device)


TRAIN_REGION = "white" if args.checker else "all"


def cube_pos():
    return obj.data.root_pos_w - isaac_env.scene.env_origins


def start_episode(task_xy):
    """Reset every env, move the cube to task_xy (keeping its z / orientation), return the fresh policy obs."""
    env.reset()
    for _ in range(WARMUP):
        env.step(warmup_act)
    pos = obj.data.root_pos_w.clone()
    pos[:, :2] = isaac_env.scene.env_origins[:, :2] + task_xy
    ids = torch.arange(N, device=device)
    obj.write_root_pose_to_sim(torch.cat([pos, obj.data.root_quat_w], -1), env_ids=ids)
    obj.write_root_velocity_to_sim(torch.zeros(N, 6, device=device), env_ids=ids)
    pobs = isaac_env.observation_manager.compute()["policy"]
    if USE_RFS:
        expert_begin(pobs)
    else:
        formatter.reset()
    return pobs


def fake_raw(pcd, c):
    """Raw point cloud (B,3,P) with the cube points moved to the default cube position."""
    near = ((pcd - c[:, :, None]).norm(dim=1) < PRIOR_RADIUS)[:, None]
    delta = torch.cat([torch.tensor(DEFAULT_XY, device=device)[None] - c[:, :2], torch.zeros(c.shape[0], 1, device=device)], -1)
    return pcd + near * delta[:, :, None]


# RFS expert: PPO (noise + residual) on top of the CFM policy, queried every step; replicates RFSWrapper.reset_to_spawn/step bookkeeping
# (obs history, past actions, one-step-lagged point-cloud embedding) for two streams: the true cube and the cube faked at the default position.
_emb, _emb_new = {}, {}


def expert_begin(pobs):
    formatter.reset()
    rfs_env._ppo_history_buf = None
    rfs_env._ppo_past_action_buf = None
    rfs_env._update_ppo_history(pobs)
    rfs_env._ppo_past_action_buf = deque([torch.zeros(N, A, device=device) for _ in range(rfs_env._ppo_n_obs_steps - 1)],
                                         maxlen=rfs_env._ppo_n_obs_steps - 1)
    rfs_env.last_obs = {"policy": pobs}
    _emb["true"] = rfs_env._compute_pcd_embedding()
    rfs_env.last_obs = {"policy": dict(pobs, **{image_key: fake_raw(pobs[image_key], pobs_cube_local())})}
    _emb["fake"] = rfs_env._compute_pcd_embedding()


def pobs_cube_local():
    return cube_pos()


def rfs_query(pobs, stream, od):
    rfs_env.last_pcd_embedding = _emb[stream]
    ppo_obs = {k: v.detach().cpu().float().numpy() for k, v in rfs_env._asymmetric_ppo_obs({"policy": pobs})["policy"].items()}
    ppo_a, _ = ppo.predict(ppo_obs, deterministic=False)
    out = torch.as_tensor(ppo_a, device=device).float().clamp(-3.0, 3.0)
    noise = torch.zeros(N, rfs_env.policy_horizon, A, device=device)
    noise[:, :, rfs_env.noise_slice] = out[:, rfs_env.n_residual_flat:].reshape(N, rfs_env.policy_horizon, rfs_env.n_noise)
    res = expert.predict_action(od, noise=noise)
    a = res["action_pred"][:, 0].clone()
    a[:, rfs_env.residual_slice] += out[:, :rfs_env.n_residual_flat] * rfs_env.residual_scale
    _emb_new[stream] = res["pcd_feat"].detach().clone()
    return a


def query(pobs, od, fake=False):
    if fake:
        od = fake_default_cube(od, cube_pos())
    if USE_RFS:
        return rfs_query(pobs, "fake" if fake else "true", od)
    return expert.predict_action(od)["action_pred"][:, 0]


def advance(pobs, a):
    """After env.step: push the executed action / new obs into the expert's histories."""
    formatter.update_action(a)
    if USE_RFS:
        rfs_env._update_ppo_history(pobs)
        rfs_env._update_ppo_past_actions(a)
        _emb.update(_emb_new)
        _emb_new.clear()


def student_obs(pobs, prev_a, t):
    q = torch.cat([pobs["arm_joint_pos"], pobs["hand_joint_pos"]], -1).float()
    o = [q, prev_a, torch.full((q.shape[0], 1), t / T, device=device)]
    if args.bc_full_info:
        o.append(cube_pos().float())
    return torch.cat(o, -1)


def fake_default_cube(od, c):
    """Expert obs with the cube points moved to the default position (an expert 'told the default cube position')."""
    pcd = od[image_key]  # (B, 1, 3, P) env frame
    near = ((pcd - c[:, None, :, None]).norm(dim=2) < PRIOR_RADIUS)[:, :, None]  # (B,1,1,P)
    delta = torch.cat([torch.tensor(DEFAULT_XY, device=device)[None] - c[:, :2], torch.zeros(c.shape[0], 1, device=device)], -1)  # (B, 3)
    od2 = dict(od)
    od2[image_key] = pcd + near * delta[:, None, :, None]
    return od2


_pcd_stats = {}


def student_pcd(pobs, alpha):
    """(N, P, 3): the episode's initial point cloud (env frame) standardised with fixed global per-axis stats, interpolated towards N(0, I) noise by
    alpha (N,): alpha=0 clean, alpha=1 pure noise."""
    pcd = pobs[image_key].float().transpose(1, 2)
    if not _pcd_stats:
        _pcd_stats.update(mu=pcd.mean((0, 1)), sd=pcd.std((0, 1)).clamp_min(1e-3))
        print(f"[icl] student pcd points={pcd.shape[1]} mean={_pcd_stats['mu'].tolist()} std={_pcd_stats['sd'].tolist()}", flush=True)
    z = (pcd - _pcd_stats["mu"]) / _pcd_stats["sd"]
    return (1 - alpha[:, None, None]) * z + alpha[:, None, None] * torch.randn_like(z)


_last_text = [""]  # command of env 0 for the last make_feedback call (video overlay)


def command_text(delta):
    """(n,3) displacement [m, env frame: x forward, y left, z up] -> templated commands (0.1 cm resolution): 3 openers x 6 axis orders."""
    names = [("forward", "backward"), ("left", "right"), ("up", "down")]
    out = []
    for d in delta.tolist():
        parts = [f"{abs(v) * 100:.1f} cm {names[a][0] if v >= 0 else names[a][1]}" for a, v in enumerate(d)]
        opener = ("move the hand", "shift the hand", "move it")[text_rng.randint(3)]
        out.append(opener + " " + " and ".join(parts[i] for i in text_rng.permutation(3)))
    return out


def encode_text(texts):
    """Frozen LM encoder: (n, tok_len, D + 1) = per word piece features (fp16) and the attention mask as the last channel."""
    b = tok(texts, return_tensors="pt", padding="max_length", max_length=args.tok_len, truncation=True).to(device)
    h = lm_enc(input_ids=b.input_ids, attention_mask=b.attention_mask).last_hidden_state
    return torch.cat([h, b.attention_mask[..., None].to(h.dtype)], -1).half()


def make_feedback(kind, cube0, cube_f, ee_f, succ):
    if kind == "success":
        return succ.float()[:, None]
    if kind == "language_tokens":
        texts = command_text(cube0 - ee_f)
        _last_text[0] = texts[0]
        return encode_text(texts)
    c = cube0 if kind == "init_cube" else cube_f
    return torch.cat([c - ee_f, ee_f], -1)


@torch.no_grad()
def rollout(task_xy, mode, policy=None, ctx=None, ctx_mode="normal", prior_mask=None, alpha=None, relabel=None):
    """One episode on every task. mode: 'student' (envs with prior_mask are driven by the default-position expert instead) | 'expert' (true cube) |
    'prior_expert' (told the default position). ctx: dict(steps, y, n, valid) of previous episodes.
    relabel: a frozen no-context policy; if given, labels are its actions on the CLEAN cloud (no expert queries).
    Returns states (N,T,S), executed actions, expert labels, alive mask, feedback and success_once for this episode."""
    pobs = start_episode(task_xy)
    # with relabel (study icl) the STORED cloud is the clean one; the noisy cloud only drives the student's actions
    spcd_store = student_pcd(pobs, torch.zeros(N, device=device)) if PCD and relabel is not None else None
    cube0 = cube_pos().clone()
    if PCD:
        if alpha is None:
            alpha = torch.zeros(N, device=device) if args.bc_pcd else torch.rand(N, device=device)
        spcd = student_pcd(pobs, alpha)
    if args.debug_checks:
        near = ((pobs[image_key] - cube0[:, :, None]).norm(dim=1) < 0.04).sum(1).float().mean().item()
        print(f"[debug] cube target xy[0]={task_xy[0].tolist()} sim cube xy[0]={cube0[0, :2].tolist()} pcd points within 4 cm of cube: {near:.1f}", flush=True)
    use_student = mode == "student" and (prior_mask is None or not bool(prior_mask.all()))
    if use_student:
        n, cy = ctx["n"], ctx["y"]
        if ctx_mode == "shuf_y":
            cy = cy[torch.randperm(N, device=device)]
        elif ctx_mode == "no_y":
            cy = torch.zeros_like(cy)
        elif ctx_mode == "no_ctx":
            n = torch.zeros_like(n)
        memory, pad = policy.encode_context(ctx["steps"], cy, n, ctx["valid"])
        pf = policy.encode_pcd(spcd) if PCD else None
    if relabel is not None:
        c0 = new_ctx()
        mem0, pad0 = relabel.encode_context(c0["steps"], c0["y"], c0["n"], c0["valid"])
        pf0 = relabel.encode_pcd(student_pcd(pobs, torch.zeros(N, device=device)))
    prev_a = warmup_act.clone()
    succ = torch.zeros(N, dtype=torch.bool, device=device)
    ee_f, cube_f = pobs["ee_pose"][:, :3].clone(), cube0.clone()
    xs, acts, labs, alive, fbs = [], [], [], [], []
    frames, succ_t = [], None
    for t in range(T):
        od = formatter.format(pobs)
        lab = query(pobs, od) if relabel is None else relabel.decode(student_obs(pobs, prev_a, t)[:, None], mem0, pad0, pcd_feat=pf0)[:, 0]
        if mode == "expert":
            a = lab
        else:
            if mode == "prior_expert" or (prior_mask is not None and bool(prior_mask.any())):
                a_prior = query(pobs, od, fake=True)
            if mode == "prior_expert":
                a = a_prior
            else:
                s = student_obs(pobs, prev_a, t)
                a = policy.decode(s[:, None], memory, pad, pcd_feat=pf)[:, 0] if use_student else a_prior
                if prior_mask is not None and use_student:
                    a = torch.where(prior_mask[:, None], a_prior, a)
        xs.append(student_obs(pobs, prev_a, t))
        fbs.append(pobs["ee_pose"][:, :3] - cube_pos())  # dense feedback: gripper position relative to the true CURRENT cube position
        acts.append(a)
        labs.append(lab)
        alive.append(~succ)
        live = ~succ
        ee_f = torch.where(live[:, None], pobs["ee_pose"][:, :3], ee_f)
        cube_f = torch.where(live[:, None], cube_pos(), cube_f)
        succ = succ | torch.from_numpy(isaac_env.metrics.get_metrics()["is_success"]).to(device).bool()
        if args.viz and succ_t is None and bool(succ[0]):
            succ_t = t
        pobs = env.step(a)[0]["policy"]
        if args.viz:
            frames.append(isaac_env.scene["fixed_camera"].data.output["rgb"][0, ..., :3].cpu().numpy())
        advance(pobs, a)
        prev_a = a
    alive = torch.stack(alive, 1)
    return dict(obs=torch.stack(xs, 1), fb=torch.stack(fbs, 1), act=torch.stack(acts, 1), lab=torch.stack(labs, 1), alive=alive,
                y=make_feedback(args.feedback, cube0, cube_f, ee_f, succ), text=_last_text[0], success_once=succ, ep_len=alive.sum(1).float(),
                dist=(cube0[:, :2] - ee_f[:, :2]).norm(dim=1), frames=frames, succ_t=succ_t,
                pcd=(spcd_store if spcd_store is not None else spcd).half() if PCD else None,
                alpha=alpha if PCD else None)


# ---------------------------------------------------------------- context
def new_ctx():
    return dict(steps=torch.zeros(N, K, Tc, S + A + F_FB, device=device), y=torch.zeros(N, K, *Y_SHAPE, device=device, dtype=torch.float16 if LANG else torch.float32),
                valid=torch.zeros(N, K, Tc, dtype=torch.bool, device=device), n=torch.zeros(N, dtype=torch.long, device=device))


def add_to_ctx(ctx, ep):
    i = ctx["n"][0].item()
    ctx["steps"][:, i] = torch.cat([ep["obs"][:, ctx_idx], ep["act"][:, ctx_idx]] + ([ep["fb"][:, ctx_idx]] if F_FB else []), -1)
    ctx["valid"][:, i] = ep["alive"][:, ctx_idx]
    if not F_FB:
        ctx["y"][:, i] = ep["y"]  # dense feedback replaces the per-episode token (left at zero)
    ctx["n"] += 1


def collect(policy, tasks, student_mode, ctx_mode="normal", frac_prior=0.0, alphas=None, relabel=None):
    """k student episodes with growing context (+ the final measured one in eval mode). student_mode: 'prior' | 'mixed' | 'eval'."""
    ctx = new_ctx()
    eps = []
    for e in range(K + (1 if student_mode == "eval" else 0)):
        if student_mode == "prior":
            ep = rollout(tasks, "prior_expert")
        else:
            pm = (torch.rand(N, device=device) < frac_prior) if student_mode == "mixed" and frac_prior > 0 else None
            ep = rollout(tasks, "student", policy, dict(ctx, n=torch.full_like(ctx["n"], e)), ctx_mode, pm,
                         alpha=None if alphas is None else alphas[:, e], relabel=relabel)
        ep["ctx_n"] = e
        eps.append(ep)
        if e < K:
            add_to_ctx(ctx, ep)
    return eps, ctx


def summary(ep):
    out = dict(success_once=ep["success_once"].float().mean().item(), ep_len=ep["ep_len"].mean().item(), dist=ep["dist"].mean().item())
    if ep.get("alpha") is not None:  # success split by how much of the point cloud was replaced by noise
        for name, m in (("lo", ep["alpha"] < 0.5), ("hi", ep["alpha"] >= 0.5)):
            out[f"success_alpha_{name}"] = ep["success_once"][m].float().mean().item() if bool(m.any()) else float("nan")
    return out


def mean_rows(rows):
    """Average a list of equal-sized chunks' dicts of floats."""
    return {k: (rows[0][k] if k == "h" else float(np.nanmean([r[k] for r in rows]))) for k in rows[0]}


# ---------------------------------------------------------------- run
out_dir = args.out_dir or os.path.join("logs", "icl", args.exp_name)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=1, default=str)

eval_gen = torch.Generator().manual_seed(args.eval_seed + 1000)
assert args.eval_tasks % N == 0, "--eval_tasks must be a multiple of --num_envs"
eval_sets = {"id": sample_tasks(args.eval_tasks, eval_gen, TRAIN_REGION)}
if args.checker:
    eval_sets["ood"] = sample_tasks(args.eval_tasks, eval_gen, "black")

if args.pcd_dump:
    dump_tasks = sample_tasks(N, torch.Generator().manual_seed(args.eval_seed + 99))
    dump_pobs = start_episode(dump_tasks)
    os.makedirs(os.path.dirname(args.pcd_dump) or ".", exist_ok=True)
    np.savez(args.pcd_dump, pcd=dump_pobs[image_key].cpu().numpy(), cube=cube_pos().cpu().numpy(), tasks=dump_tasks.cpu().numpy())
    print(f"[dump] saved {args.pcd_dump} pcd {tuple(dump_pobs[image_key].shape)}", flush=True)
    sys.stdout.flush()
    os._exit(0)

refs = {}
for name, mode in (() if args.viz or args.alpha_sweep or args.eval_dump or args.expert_dump else (("expert_true_cube", "expert"), ("expert_default_cube", "prior_expert"))):
    t0 = time.time()
    refs[name] = {r: mean_rows([summary(rollout(t[i:i + N], mode)) for i in range(0, len(t), N)]) for r, t in eval_sets.items()}
    print(f"reference {name:20s} {refs[name]} ({time.time() - t0:.0f}s)", flush=True)
if args.refs_only:
    with open(os.path.join(out_dir, "refs.json"), "w") as f:
        json.dump(refs, f, indent=1)
    sys.stdout.flush()
    os._exit(0)

eval_alphas = torch.rand(args.eval_tasks, K + 1, generator=torch.Generator().manual_seed(args.eval_seed + 2000)).to(device)  # fixed per eval task and episode

policy = ICLPolicy(S, A, Y, K, Tc, args.d_model, 4, args.n_layers, args.n_layers, use_pcd=PCD, f_dim=F_FB).to(device)
if F_FB:
    policy.f_std.fill_(0.2)  # metres: cube_range is +-0.2 m
if args.viz:
    import imageio
    from PIL import Image, ImageDraw
    policy.load_state_dict(torch.load(args.viz, map_location=device))
    policy.eval()
    fps = 1.0 / (isaac_env.cfg.sim.dt * isaac_env.cfg.decimation)
    vdir = os.path.join(out_dir, "videos")
    os.makedirs(vdir, exist_ok=True)

    def panel(ep, title, extra):
        n = len(ep["frames"]) if ep["succ_t"] is None else min(len(ep["frames"]), ep["succ_t"] + 20)
        res = f"SUCCESS (step {ep['succ_t']})" if ep["succ_t"] is not None else "FAIL"
        out = []
        for f in ep["frames"][:n]:
            im = Image.fromarray(f).convert("RGB")
            d = ImageDraw.Draw(im)
            for i, line in enumerate([title, extra, res]):
                d.text((6, 6 + 14 * i), line, fill=(255, 255, 0))
            out.append(np.asarray(im))
        return out

    cam_p = env_cfg.events.reset_fixed_camera.params
    cam_eye, cam_at = torch.tensor(cam_p["eye_offset"], device=device), torch.tensor(cam_p["look_at_offset"], device=device)
    cam_f = 24.0 / 20.955 * 480  # fixed_camera: focal_length 24, horizontal_aperture 20.955, 480 px

    def pcd_image(ep, title, extra):
        """What the student sees: its (noisy) point cloud, drawn through the fixed camera (the same view as the rollout), coloured by height."""
        pts = ep["pcd"][0].float() * _pcd_stats["sd"] + _pcd_stats["mu"]  # back to env-frame metres
        fwd = torch.nn.functional.normalize(cam_at - cam_eye, dim=0)
        right = torch.nn.functional.normalize(torch.cross(fwd, torch.tensor([0.0, 0.0, 1.0], device=device), dim=0), dim=0)
        up = torch.cross(right, fwd, dim=0)
        rel = pts - cam_eye
        x, y, z = rel @ right, rel @ up, rel @ fwd
        u, v = (240 + cam_f * x / z.clamp_min(1e-3)).cpu().numpy(), (240 - cam_f * y / z.clamp_min(1e-3)).cpu().numpy()
        hgt = ((pts[:, 2] - 0.0) / 0.6).clamp(0, 1).cpu().numpy()
        img = np.full((480, 480, 3), 20, np.uint8)
        for i in np.argsort(-z.cpu().numpy()):
            if z[i] > 0.1 and 1 <= u[i] < 478 and 1 <= v[i] < 478:
                c = (np.array([60, 120, 255]) * (1 - hgt[i]) + np.array([255, 220, 60]) * hgt[i]).astype(np.uint8)
                img[int(v[i]) - 1:int(v[i]) + 2, int(u[i]) - 1:int(u[i]) + 2] = c
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        for i, line in enumerate([title, extra]):
            d.text((6, 6 + 14 * i), line, fill=(255, 255, 0))
        return np.asarray(im)

    g = torch.Generator().manual_seed(args.eval_seed + 7)
    viz_alphas = [float(a) for a in args.viz_alphas.split(",")]
    for r in range(args.viz_rounds):
        tasks = sample_tasks(N, g)
        al = viz_alphas[r % len(viz_alphas)] if args.vision else None
        eps, _ = collect(policy, tasks, "eval", alphas=(torch.full_like(eval_alphas[:N], al) if args.vision else None))
        y = eps[0]["y"][0].tolist() if not LANG else [0.0] * 3
        cube_xy = tasks[0].tolist()
        p0 = panel(eps[0], "h=0: no context (blind)", f"cube xy=({cube_xy[0]:.2f},{cube_xy[1]:.2f})")
        p1 = panel(eps[1], "h=1: context = ep0 + feedback",
                   f"feedback: {eps[0]['text']}" if LANG else f"feedback cube-gripper (cm): dx={100*y[0]:.0f} dy={100*y[1]:.0f} dz={100*y[2]:.0f}")
        L = max(len(p0), len(p1))
        p0 += [p0[-1]] * (L - len(p0))
        p1 += [p1[-1]] * (L - len(p1))
        tag = f"h0{'S' if eps[0]['succ_t'] is not None else 'F'}_h1{'S' if eps[1]['succ_t'] is not None else 'F'}"
        if args.vision:  # per row: [policy's point cloud | rollout]
            c0 = pcd_image(eps[0], "h=0 policy input: point cloud", f"noise alpha={al:.2f}")
            c1 = pcd_image(eps[1], "h=1 policy input: point cloud", f"noise alpha={al:.2f} (fresh noise)")
            frames = [np.concatenate([np.concatenate([c0, a], 1), np.concatenate([c1, b], 1)], 0) for a, b in zip(p0, p1)]
            tag = f"alpha{al:.2f}_" + tag
        else:
            frames = [np.concatenate([a, b], 1) for a, b in zip(p0, p1)]
        path = os.path.join(vdir, f"round{r}_{tag}.mp4")
        imageio.mimsave(path, frames, fps=fps)
        print(f"[viz] {path} cube={cube_xy} fb={y} h0={eps[0]['succ_t']} h1={eps[1]['succ_t']}", flush=True)
    sys.stdout.flush()
    os._exit(0)

opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_iterations * args.grad_steps, eta_min=args.lr * 0.02)




def pass_rows(eps):
    """Per-try rows for successive episodes on the same tasks: success_once of try j, pass_any (solved at any try <= j), rescue (first solved at try j)."""
    seen = torch.zeros_like(eps[0]["success_once"]).bool()
    rows = []
    for j, ep in enumerate(eps):
        so = ep["success_once"].bool()
        new = so & ~seen
        seen = seen | so
        rows.append(dict(h=j, **summary(ep), pass_any=seen.float().mean().item(), rescue=new.float().mean().item()))
    return rows


def evaluate(ctx_mode="normal", alpha=None, clean_last=False):
    """{region: [per-h dict]} over the fixed eval tasks, in chunks of num_envs. alpha: None -> the fixed U(0,1) draws; a float forces that
    point-cloud noise level for every episode."""
    policy.eval()
    out = {}
    for region, tasks in eval_sets.items():
        chunks = []
        for i in range(0, len(tasks), N):
            al = eval_alphas[i:i + N] if alpha is None else torch.full_like(eval_alphas[i:i + N], alpha)
            if clean_last:  # context episodes noisy, the scored (last) episode sees the clean cloud
                al = al.clone()
                al[:, K] = 0.0
            eps, _ = collect(policy, tasks[i:i + N], "eval", ctx_mode, alphas=al if args.vision else None)
            chunks.append(pass_rows(eps))
        out[region] = [mean_rows([c[j] for c in chunks]) for j in range(len(chunks[0]))]
    policy.train()
    return out


def fmt(ev, key="success_once"):
    """'id: a b c | ood: a b c' for an evaluate() result."""
    return " | ".join(f"{r}: " + " ".join(f"{e[key]:.3f}" for e in rows) for r, rows in ev.items())


if args.expert_dump:  # the expert's actions are sampled, so repeat
    dump = {}
    for region, tasks in eval_sets.items():
        succ = torch.stack([torch.cat([rollout(tasks[i:i + N], "expert")["success_once"] for i in range(0, len(tasks), N)]) for _ in range(args.expert_reps)]).cpu()
        dump[f"{region}_xy"], dump[f"{region}_success"] = tasks.cpu().numpy(), succ.numpy()
        print(f"[expert_dump] {region}: success by rep " + " ".join(f"{x:.3f}" for x in succ.float().mean(1)), flush=True)
    np.savez(os.path.join(out_dir, "eval_dump.npz"), **dump)
    sys.stdout.flush()
    os._exit(0)

if args.eval_dump:
    policy.load_state_dict(torch.load(args.eval_dump, map_location=device))
    policy.eval()
    bc = args.bc_full_info or args.bc_pcd
    no_ctx = new_ctx()
    dump = {}
    for region, tasks in eval_sets.items():
        succ = []  # per chunk: (h, N) success_once
        for i in range(0, len(tasks), N):
            if bc:
                succ.append(torch.stack([rollout(tasks[i:i + N], "student", policy, no_ctx)["success_once"]]).cpu())
            else:
                eps, _ = collect(policy, tasks[i:i + N], "eval", alphas=eval_alphas[i:i + N] if args.vision else None)
                succ.append(torch.stack([e["success_once"] for e in eps]).cpu())
        dump[f"{region}_xy"] = tasks.cpu().numpy()
        dump[f"{region}_success"] = torch.cat(succ, 1).numpy()
        print(f"[eval_dump] {region}: success by h " + " ".join(f"{x:.3f}" for x in dump[f"{region}_success"].mean(1)), flush=True)
    np.savez(os.path.join(out_dir, "eval_dump.npz"), **dump)
    sys.stdout.flush()
    os._exit(0)

if args.alpha_sweep:
    policy.load_state_dict(torch.load(args.alpha_sweep, map_location=device))
    sweep = {}
    for a_ in [round(0.1 * i, 1) for i in range(11)]:
        sweep[str(a_)] = evaluate(alpha=a_)
        print(f"[sweep] alpha={a_} success_once by h: " + fmt(sweep[str(a_)]), flush=True)
    with open(os.path.join(out_dir, "alpha_sweep.json"), "w") as f:
        json.dump(sweep, f, indent=1)
    sys.stdout.flush()
    os._exit(0)

# ---------------------------------------------------------------- checkerboard step-budget study
NK = N * K  # episodes (= NK * T env steps) of extra data per iteration, for every method


def load_cache(path):
    return torch.load(path, map_location="cpu")


def set_pcd_stats(raw):
    """Fix the point-cloud standardisation from D_0's clean clouds so every run / cache uses the same one."""
    raw = raw.float().to(device)
    _pcd_stats.update(mu=raw.mean((0, 1)), sd=raw.std((0, 1)).clamp_min(1e-3))


def std_pcd(raw):
    return ((raw.float().to(device) - _pcd_stats["mu"]) / _pcd_stats["sd"]).half()


if args.study:
    assert args.checker, "--study needs --checker"
if args.study == "collect":
    assert args.bc_pcd and args.cache_out
    cols = dict(obs=[], lab=[], alive=[], pcd=[], xy=[], success=[])
    for _ in range(-(-args.n_trajs // N)):
        tasks = sample_tasks(N, region=args.ext_region)
        ex = rollout(tasks, "expert")
        for key, val in (("obs", ex["obs"]), ("lab", ex["lab"]), ("alive", ex["alive"]), ("xy", tasks), ("success", ex["success_once"]),
                         ("pcd", (ex["pcd"].float() * _pcd_stats["sd"] + _pcd_stats["mu"]).half())):
            cols[key].append(val.cpu())
        print(f"[collect] {sum(len(x) for x in cols['xy'])}/{args.n_trajs} {args.ext_region} episodes, expert success {summary(ex)['success_once']:.3f}", flush=True)
    os.makedirs(os.path.dirname(args.cache_out) or ".", exist_ok=True)
    torch.save({k_: torch.cat(v)[:args.n_trajs] for k_, v in cols.items()}, args.cache_out)
    sys.stdout.flush()
    os._exit(0)

if args.study in ("icl", "bc_ext"):
    # Both start from the fixed pi_bc (--relabeler), keep D_0 and add --k * --num_envs episodes of new data per iteration (icl: its own noisy-cloud rollouts relabeled by
    # the fixed pi_bc on the clean cloud; bc_ext: expert episodes from --ext_pool). After every addition the policy is trained until the held-out loss stops improving;
    # every batch is half D_0 (D_0 held-out split as in bc_converge), half new data. Gradient steps are therefore not a tuned quantity.
    ICLM = args.study == "icl"
    assert args.relabeler and (args.vision if ICLM else args.bc_pcd and args.ext_pool) and args.feedback == "init_cube"
    d0 = load_cache(args.d0_cache)
    set_pcd_stats(d0["pcd"])
    Q0, L0, QA0, PC0 = d0["obs"].to(device), d0["lab"].to(device), d0["alive"].to(device), std_pcd(d0["pcd"])
    perm = torch.randperm(Q0.shape[0], generator=torch.Generator().manual_seed(1)).to(device)  # the same D_0 split as the fixed pi_bc's bc_converge run (seed 1)
    nv = max(1, int(args.val_frac * len(perm)))
    d0_va, d0_tr = perm[:nv], perm[nv:]
    al_va = (torch.rand(nv, generator=torch.Generator().manual_seed(7)) if ICLM and not args.d0_clean else torch.zeros(nv)).to(device)  # fixed noise levels for the D_0 val loss
    sd0 = torch.load(args.relabeler, map_location=device)
    if F_FB:  # the fixed pi_bc has no dense-feedback inputs: new step_in columns start at zero
        w = torch.zeros_like(policy.step_in.weight)
        w[:, : S + A] = sd0["step_in.weight"]
        sd0.update({"step_in.weight": w, "f_mean": policy.f_mean, "f_std": policy.f_std})
    policy.load_state_dict(sd0)
    if ICLM:
        bcstar = ICLPolicy(S, A, Y, K, Tc, args.d_model, 4, args.n_layers, args.n_layers, use_pcd=True).to(device)
        bcstar.load_state_dict(torch.load(args.relabeler, map_location=device))
        bcstar.eval()
        for q_ in bcstar.parameters():
            q_.requires_grad_(False)
        policy.y_mean.zero_()
        policy.y_std.fill_(0.1)  # feedback token scale (metres); the fixed pi_bc never saw feedback, so its stats are degenerate
    nc = new_ctx()

    def null_mem(B):
        return policy.encode_context(nc["steps"][:1].expand(B, -1, -1, -1), nc["y"][:1].expand(B, -1, -1), torch.zeros(B, dtype=torch.long, device=device),
                                     nc["valid"][:1].expand(B, -1, -1))

    def seq_loss(Qb, Lb, QAb, pcd, mem, pad):
        pred = policy.decode(Qb, mem, pad, normalized=True, pcd_feat=policy.encode_pcd(pcd.float()))
        m = QAb.float()
        return (((pred - policy.norm_a(Lb)) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)

    def d0_loss(ix, al):
        z = PC0[ix].float()
        mem, pad = null_mem(len(ix))
        return seq_loss(Q0[ix], L0[ix], QA0[ix], (1 - al[:, None, None]) * z + al[:, None, None] * torch.randn_like(z), mem, pad)

    NEW = {}  # per-episode tensors: obs, lab, alive, pcd (+ icl: ctx steps / feedback / valid and the number of context episodes h), isval

    def new_loss(ix):
        mem, pad = policy.encode_context(NEW["cs"][ix], NEW["cy"][ix], NEW["h"][ix], NEW["cv"][ix]) if ICLM else null_mem(len(ix))
        return seq_loss(NEW["obs"][ix], NEW["lab"][ix], NEW["alive"][ix], NEW["pcd"][ix], mem, pad)

    def val_mask(n, g):  # exactly 10% (at least one) of n items held out
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.randperm(n, generator=g)[:max(1, round(0.1 * n))]] = True
        return m.to(device)

    def add_new(cols, isval):
        for k_, v in cols.items():
            NEW[k_] = v if k_ not in NEW else torch.cat([NEW[k_], v])
        NEW["isval"] = isval if "isval" not in NEW else torch.cat([NEW["isval"], isval])

    def chunked(fn, ix, *extra):
        return float(sum(fn(ix[i:i + args.batch_size], *[e[i:i + args.batch_size] for e in extra]).item() * len(ix[i:i + args.batch_size])
                         for i in range(0, len(ix), args.batch_size)) / len(ix))

    def fit(has_new):
        """Train (at least --min_steps) until the held-out loss (mean of the D_0 and new-data val losses when there is new data) has not improved by 1% for --patience
        evals; keep the best weights from the evals after --min_steps.
        Returns the (train, val) loss curve."""
        new_tr, new_va = (~NEW["isval"]).nonzero()[:, 0], NEW["isval"].nonzero()[:, 0]
        opt_c = torch.optim.Adam(policy.parameters(), lr=args.lr)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_c, factor=0.3, patience=5, min_lr=args.lr * 0.01)
        best, best_step, best_sd, since, curve, tl = float("inf"), 0, None, 0, [], []
        B0 = args.batch_size // 2 if has_new else args.batch_size
        step, vl = 0, float("nan")  # max_steps 0 keeps the (warm-started) weights untouched
        for step in range(1, args.max_steps + 1):
            loss = d0_loss(d0_tr[torch.randint(0, len(d0_tr), (B0,), device=device)], torch.rand(B0, device=device) if ICLM and not args.d0_clean else torch.zeros(B0, device=device))
            if has_new:
                loss = 0.5 * (loss + new_loss(new_tr[torch.randint(0, len(new_tr), (args.batch_size - B0,), device=device)]))
            opt_c.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt_c.step()
            tl.append(loss.item())
            if step % 500 == 0:
                with torch.no_grad():
                    v0 = chunked(d0_loss, d0_va, al_va)
                    v1 = chunked(new_loss, new_va) if has_new else None
                vl = 0.5 * (v0 + v1) if has_new else v0
                plateau.step(vl)
                curve.append(dict(step=step, train_loss=sum(tl[-500:]) / 500, val_loss=vl, val_d0=v0, val_new=v1, lr=opt_c.param_groups[0]["lr"]))
                if step >= args.min_steps:
                    since = 0 if vl < best * 0.99 else since + 1
                    if vl < best:
                        best, best_step, best_sd = vl, step, {k_: v.clone() for k_, v in policy.state_dict().items()}
                    if since >= args.patience:
                        break
        if best_sd is None:  # max_steps < min_steps
            best, best_step = vl, step
        else:
            policy.load_state_dict(best_sd)
        return curve, best_step, step, best

    no_ctx = new_ctx()

    def evaluate_policy():
        policy.eval()
        if ICLM:
            evs = evaluate(alpha=0.0)  # clean cloud for every episode at every h
            out = dict(eval_id=evs["id"], eval_ood=evs["ood"])
        else:
            # no context / feedback: the same K tries per task, each an independent episode of the same policy (pass@1, pass@2 control for the ICL's retry)
            chunks_r = {r: [pass_rows([rollout(t[i:i + N], "student", policy, no_ctx) for _ in range(K)]) for i in range(0, len(t), N)] for r, t in eval_sets.items()}
            ev = {r: [mean_rows([c[h] for c in cs]) for h in range(K)] for r, cs in chunks_r.items()}
            out = dict(eval_id=ev["id"], eval_ood=ev["ood"])
        policy.train()
        return out

    if args.study == "bc_ext":
        pool = load_cache(args.ext_pool)
        assert len(pool["xy"]) >= args.num_iterations * NK, "expert pool too small"
    log = []
    for it in range(args.num_iterations + 1):
        t0 = time.time()
        if it > 0:
            g = torch.Generator().manual_seed(1000 + it)
            if ICLM:
                tasks = sample_tasks(N, region=TRAIN_REGION)
                # cold start: before the context path has seen any data (iteration 1) all episodes are driven by the h=0 policy (context ignored when acting, but the
                # previous episodes + corrections are still stored as each episode's context for training)
                eps, ctx = collect(policy, tasks, "mixed", ctx_mode="no_ctx" if it == 1 else "normal", relabel=bcstar)
                val_task = val_mask(N, g)  # held out by task: all episodes of a task stay together
                add_new(dict(obs=torch.cat([e["obs"] for e in eps]), lab=torch.cat([e["lab"] for e in eps]), alive=torch.cat([e["alive"] for e in eps]),
                             pcd=torch.cat([e["pcd"] for e in eps]), cs=ctx["steps"].repeat(K, 1, 1, 1), cy=ctx["y"].repeat(K, 1, 1), cv=ctx["valid"].repeat(K, 1, 1),
                             h=torch.arange(K, device=device).repeat_interleave(N)), val_task.repeat(K))
                print(f"[collect] it {it} rollout success by episode: " + " ".join(f"{summary(e)['success_once']:.2f}" for e in eps), flush=True)
            else:
                sl = slice((it - 1) * NK, it * NK)
                add_new(dict(obs=pool["obs"][sl].to(device), lab=pool["lab"][sl].to(device), alive=pool["alive"][sl].to(device), pcd=std_pcd(pool["pcd"][sl])),
                        val_mask(NK, g))
        if it == 0:
            NEW["isval"] = torch.zeros(0, dtype=torch.bool, device=device)  # iteration 0 (icl and bc_ext alike): D_0 only
        curve, best_step, steps_run, best = fit(it > 0)
        ev_row = evaluate_policy()
        row = dict(iteration=it, env_steps=it * NK * T, n_new_episodes=int(len(NEW["isval"])) if NEW else 0, steps_run=steps_run, best_step=best_step, best_val=best,
                   curve=curve, seconds=time.time() - t0, **ev_row)
        log.append(row)
        f_ = lambda e: " ".join(f"{x['pass_any']:.3f}" for x in e)  # pass@any: solved at any try up to h
        print(f"it {it:2d} steps {row['env_steps']} trained {steps_run} (best {best_step}) val {best} | " + (f"id {f_(ev_row['eval_id'])} | ood {f_(ev_row['eval_ood'])}"
              if ev_row else "no eval (= fixed pi_bc)")
              + f" | {row['seconds']:.0f}s", flush=True)
        with open(os.path.join(out_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        torch.save(policy.state_dict(), os.path.join(out_dir, "policy_latest.pt"))
    sys.stdout.flush()
    os._exit(0)

if args.study == "bc_converge":  # one BC policy trained on D_0 until the held-out loss stops improving, then evaluated once: the fixed pi_bc
    assert args.bc_pcd
    d0 = load_cache(args.d0_cache)
    set_pcd_stats(d0["pcd"])
    Q0, L0, QA0, PC0 = d0["obs"].to(device), d0["lab"].to(device), d0["alive"].to(device), std_pcd(d0["pcd"])
    perm = torch.randperm(Q0.shape[0], generator=torch.Generator().manual_seed(args.seed)).to(device)
    nv = max(1, int(args.val_frac * len(perm)))
    va, tr = perm[:nv], perm[nv:]
    policy.set_norm(Q0[tr].flatten(0, 1), L0[tr].flatten(0, 1), torch.zeros(2, Y))
    no_ctx = new_ctx()

    def bc_loss(ix):
        B = len(ix)
        mem, pad = policy.encode_context(no_ctx["steps"][:1].expand(B, -1, -1, -1), no_ctx["y"][:1].expand(B, -1, -1),
                                         torch.zeros(B, dtype=torch.long, device=device), no_ctx["valid"][:1].expand(B, -1, -1))
        pred = policy.decode(Q0[ix], mem, pad, normalized=True, pcd_feat=policy.encode_pcd(PC0[ix].float()))
        m = QA0[ix].float()
        return (((pred - policy.norm_a(L0[ix])) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)

    opt_c = torch.optim.Adam(policy.parameters(), lr=args.lr)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_c, factor=0.3, patience=5, min_lr=args.lr * 0.01)
    best, best_step, best_sd, since, curve, tl = float("inf"), 0, None, 0, [], []
    eval_every, patience = 500, 20  # stop after 20 evals (10k steps) without a 1% improvement of the held-out loss
    for step in range(1, args.max_steps + 1):
        loss = bc_loss(tr[torch.randint(0, len(tr), (args.batch_size,), device=device)])
        opt_c.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_c.step()
        tl.append(loss.item())
        if step % eval_every == 0:
            with torch.no_grad():
                vl = float(sum(bc_loss(va[i:i + args.batch_size]).item() * len(va[i:i + args.batch_size]) for i in range(0, len(va), args.batch_size)) / len(va))
            plateau.step(vl)
            curve.append(dict(step=step, train_loss=sum(tl[-eval_every:]) / eval_every, val_loss=vl, lr=opt_c.param_groups[0]["lr"]))
            since = 0 if vl < best * 0.99 else since + 1
            if vl < best:
                best, best_step, best_sd = vl, step, {k_: v.clone() for k_, v in policy.state_dict().items()}
            print(f"[bc_converge] step {step} train {curve[-1]['train_loss']:.4f} val {vl:.4f} (best {best:.4f} @ {best_step}) lr {curve[-1]['lr']:.1e}", flush=True)
            if since >= patience:
                break
    policy.load_state_dict(best_sd)
    policy.eval()
    ev = {r: mean_rows([dict(h=0, **summary(rollout(t[i:i + N], "student", policy, no_ctx))) for i in range(0, len(t), N)]) for r, t in eval_sets.items()}
    print(f"[bc_converge] converged at best step {best_step} (ran {step}); eval success_once id {ev['id']['success_once']:.3f} ood {ev['ood']['success_once']:.3f}", flush=True)
    torch.save(policy.state_dict(), os.path.join(out_dir, "policy_bc_fixed.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(dict(refs=refs, best_step=best_step, steps_run=step, best_val_loss=best, eval=ev, curve=curve), f, indent=1)
    sys.stdout.flush()
    os._exit(0)

if args.bc_full_info or args.bc_pcd:  # baseline: expert-only BC (privileged cube position, or clean point cloud), same per-iteration expert budget (num_envs episodes)
    D = dict(obs=[], lab=[], alive=[], pcd=[])
    log = []
    no_ctx = new_ctx()  # n = 0 everywhere: the decoder only sees the null memory token

    def bc_eval():
        policy.eval()
        out = {r: [mean_rows([dict(h=0, **summary(rollout(t[i:i + N], "student", policy, no_ctx))) for i in range(0, len(t), N)])] for r, t in eval_sets.items()}
        policy.train()
        return out

    for it in range(args.num_iterations):
        t0 = time.time()
        for _ in range(-(-(args.demos_per_iter or N) // N)):
            ex = rollout(sample_tasks(N, region=TRAIN_REGION), "expert")
            D["obs"].append(ex["obs"]), D["lab"].append(ex["lab"]), D["alive"].append(ex["alive"])
            if args.bc_pcd:
                D["pcd"].append(ex["pcd"])
        Q, L, QA = torch.cat(D["obs"]), torch.cat(D["lab"]), torch.cat(D["alive"])
        PC = torch.cat(D["pcd"]) if args.bc_pcd else None
        if it == 0:
            policy.set_norm(Q.flatten(0, 1), L.flatten(0, 1), torch.zeros(2, Y))  # >= 2 rows: std of one sample is NaN
        M = Q.shape[0]
        losses = []
        for _ in range(args.grad_steps):
            idx = torch.randint(0, M, (args.batch_size,), device=device)
            memory, pad = policy.encode_context(no_ctx["steps"][:1].expand(args.batch_size, -1, -1, -1), no_ctx["y"][:1].expand(args.batch_size, -1, -1),
                                                torch.zeros(args.batch_size, dtype=torch.long, device=device), no_ctx["valid"][:1].expand(args.batch_size, -1, -1))
            pred = policy.decode(Q[idx], memory, pad, normalized=True, pcd_feat=policy.encode_pcd(PC[idx].float()) if args.bc_pcd else None)
            m = QA[idx].float()
            loss = (((pred - policy.norm_a(L[idx])) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        ev = bc_eval()
        row = dict(iteration=it, loss=sum(losses[-100:]) / 100, n_tasks=M, eval=ev["id"], eval_id=ev["id"], seconds=time.time() - t0, expert_success=summary(ex)["success_once"])
        if "ood" in ev:
            row["eval_ood"] = ev["ood"]
        log.append(row)
        print(f"it {it:2d} loss {row['loss']:.4f} expert episodes {M} | eval success_once: {fmt(ev)} | dist {ev['id'][0]['dist']:.3f} | {row['seconds']:.0f}s", flush=True)
        with open(os.path.join(out_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    torch.save(policy.state_dict(), os.path.join(out_dir, "policy_final.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(dict(refs=refs, final_eval=log[-1]["eval"], final_eval_ood=log[-1].get("eval_ood")), f, indent=1)
    sys.stdout.flush()
    os._exit(0)

D = dict(q_obs=[], q_lab=[], q_alive=[], q_pcd=[], cs=[], cy=[], cv=[])
log = []
for it in range(args.num_iterations):
    t0 = time.time()
    n_new = args.demos_per_iter or N
    for _ in range(-(-n_new // N)):
        tasks = sample_tasks(N, region=TRAIN_REGION)
        eps, ctx = collect(policy, tasks, "prior" if it == 0 else "mixed", frac_prior=args.prior_expert_frac)
        ex = rollout(tasks, "expert")
        n_keep = min(N, n_new)
        n_new -= n_keep
        D["q_obs"].append(torch.stack([e["obs"] for e in eps] + [ex["obs"]], 1)[:n_keep])  # (n, k+1, T, S)
        D["q_lab"].append(torch.stack([e["lab"] for e in eps] + [ex["lab"]], 1)[:n_keep])
        D["q_alive"].append(torch.stack([e["alive"] for e in eps] + [ex["alive"]], 1)[:n_keep])
        if args.vision:
            D["q_pcd"].append(torch.stack([e["pcd"] for e in eps] + [ex["pcd"]], 1)[:n_keep])  # (n, k+1, P, 3)
        D["cs"].append(ctx["steps"][:n_keep]), D["cy"].append(ctx["y"][:n_keep]), D["cv"].append(ctx["valid"][:n_keep])
    Q, L, QA = torch.cat(D["q_obs"]), torch.cat(D["q_lab"]), torch.cat(D["q_alive"])
    PC = torch.cat(D["q_pcd"]) if args.vision else None
    CS, CY, CV = torch.cat(D["cs"]), torch.cat(D["cy"]), torch.cat(D["cv"])
    if it == 0:
        policy.set_norm(Q.flatten(0, 1), L.flatten(0, 1), CY.flatten(0, 1))
    M = Q.shape[0]
    losses = []
    for _ in range(args.grad_steps):
        idx = torch.randint(0, M, (args.batch_size,), device=device)
        j = torch.randint(0, K + 1, (args.batch_size,), device=device)  # query episode: 0..k-1 student (own context), k expert (any context)
        h = torch.where(j == K, torch.randint(0, K + 1, (args.batch_size,), device=device), j)
        memory, pad = policy.encode_context(CS[idx], CY[idx], h, CV[idx])
        pred = policy.decode(Q[idx, j], memory, pad, normalized=True, pcd_feat=policy.encode_pcd(PC[idx, j].float()) if args.vision else None)
        m = QA[idx, j].float()
        loss = (((pred - policy.norm_a(L[idx, j])) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
    evs = evaluate()
    ev = evs["id"]
    row = dict(iteration=it, loss=sum(losses[-100:]) / 100, n_tasks=M, eval=ev, eval_id=ev, seconds=time.time() - t0,
               train_success=[summary(e)["success_once"] for e in eps], expert_success=summary(ex)["success_once"])
    if "ood" in evs:
        row["eval_ood"] = evs["ood"]
    log.append(row)
    print(f"it {it:2d} loss {row['loss']:.4f} tasks {M} | eval success_once by h: " + fmt(evs)
          + " | dist cube-gripper (id): " + " ".join(f"{e['dist']:.3f}" for e in ev)
          + (" | alpha<.5 / >=.5 (id): " + " ".join(f"{e['success_alpha_lo']:.2f}/{e['success_alpha_hi']:.2f}" for e in ev) if args.vision else "")
          + f" | {row['seconds']:.0f}s", flush=True)
    with open(os.path.join(out_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    torch.save(policy.state_dict(), os.path.join(out_dir, "policy_latest.pt"))

controls = {"normal": {r: log[-1]["eval_ood" if r == "ood" else "eval"] for r in eval_sets}}
for cm in ("shuf_y", "no_y", "no_ctx") + (("alpha0", "alpha1") if args.vision else ()):
    controls[cm] = evaluate(*(("normal", float(cm[-1])) if cm.startswith("alpha") else (cm,)))
    print(f"control {cm:7s} success_once by h: " + fmt(controls[cm]), flush=True)
torch.save(policy.state_dict(), os.path.join(out_dir, "policy_final.pt"))
with open(os.path.join(out_dir, "results.json"), "w") as f:
    json.dump(dict(refs=refs, controls=controls, final_eval=log[-1]["eval"], final_eval_ood=log[-1].get("eval_ood")), f, indent=1)

fig, axes = plt.subplots(1, len(eval_sets), figsize=(5 * len(eval_sets), 4), squeeze=False)
for ax, region in zip(axes[0], eval_sets):
    for cm, evr in controls.items():
        ax.plot([e["h"] for e in evr[region]], [e["success_once"] for e in evr[region]], marker="o", label=cm)
    ax.axhline(refs["expert_true_cube"][region]["success_once"], color="k", ls="--", label="expert (true cube)")
    ax.axhline(refs["expert_default_cube"][region]["success_once"], color="gray", ls=":", label="expert (default cube)")
    ax.set_title(region)
    ax.set_xlabel("h (previous episodes in context)")
    ax.set_ylabel("success_once")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "icl_curve.png"), dpi=120)
sys.stdout.flush()
os._exit(0)
