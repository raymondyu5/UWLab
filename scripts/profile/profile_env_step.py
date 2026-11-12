"""Per-phase timing of ManagerBasedRLEnv.step for an RFS task (no policy; zero-ish actions).

Usage (inside container):
    ./uwlab.sh -p scripts/profile/profile_env_step.py --task UW-FrankaLeap-GraspCube-JointAbs-v0 --num_envs 512 --headless

Each wrapped phase is bracketed by cuda syncs, so absolute numbers are slightly inflated; use them for ratios.
Only steady-state steps are counted (warmup steps are excluded).
"""

import argparse
import time

from isaaclab.app import AppLauncher
from uwlab.utils.paths import setup_third_party_paths

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--fast_substeps", action="store_true", help="enable grasp.mdp.fast_step.enable_fast_substeps")
parser.add_argument("--check_steps", type=int, default=100, help="deterministic rollout length for the behavior checksum")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
setup_third_party_paths()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import uwlab_tasks  # noqa: F401

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.run_mode = "rl_mode"
env_cfg.seed = 0
torch.manual_seed(0)
env = gym.make(args.task, cfg=env_cfg)
u = env.unwrapped
if args.fast_substeps:
    from uwlab_tasks.manager_based.manipulation.grasp.mdp.fast_step import enable_fast_substeps

    print("[ENVPROFILE] fast_substeps patched:", enable_fast_substeps(env), flush=True)
env.reset()
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
if hasattr(u.cfg, "warmup_action"):
    action = u.cfg.warmup_action(u)

acc: dict[str, float] = {}


def wrap(obj, name, label):
    fn = getattr(obj, name)

    def timed(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        acc[label] = acc.get(label, 0.0) + (time.perf_counter() - t0) * 1e3
        return out

    setattr(obj, name, timed)


targets = [
    (u.sim, "step", "sim.step (physx+kit, all substeps)"),
    (u.scene, "write_data_to_sim", "scene.write_data_to_sim"),
    (u.scene, "update", "scene.update"),
    (u.observation_manager, "compute", "obs.compute"),
    (u.reward_manager, "compute", "reward.compute"),
    (u.termination_manager, "compute", "termination.compute"),
    (u.action_manager, "process_action", "action.process"),
    (u.action_manager, "apply_action", "action.apply"),
    (u, "_reset_idx", "reset_idx"),
]
sc = u.scene
for group in ("articulations", "rigid_objects", "deformable_objects", "sensors"):
    for ename, ent in getattr(sc, group, {}).items():
        targets.append((ent, "update", f"  {group}/{ename}.update"))
        targets.append((ent, "write_data_to_sim", f"  {group}/{ename}.write_data_to_sim"))
        for aname, act in getattr(ent, "actuators", {}).items():
            targets.append((act, "compute", f"    actuator {ename}/{aname}.compute ({type(act).__name__})"))
print("[ENVPROFILE] scene entities:", {g: list(getattr(sc, g, {}).keys()) for g in ("articulations", "rigid_objects", "sensors")}, flush=True)
for obj, name, label in targets:
    if hasattr(obj, name):
        wrap(obj, name, label)

for _ in range(args.warmup):
    env.step(action)
acc.clear()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(args.steps):
    env.step(action)
torch.cuda.synchronize()
total = (time.perf_counter() - t0) * 1e3 / args.steps
acc_snapshot = dict(acc)  # the checksum rollout below also runs through the timers
# Behavior checksum: deterministic noisy-action rollout; compare between runs with/without --fast_substeps.
gen = torch.Generator(device=u.device).manual_seed(123)
rew_sum = torch.zeros((), device=u.device, dtype=torch.float64)
force_sum = torch.zeros((), device=u.device, dtype=torch.float64)
n_resets = 0
from isaaclab.sensors import ContactSensor

sensors = [s for s in u.scene.sensors.values() if isinstance(s, ContactSensor)]
for _ in range(args.check_steps):
    a = action + 0.05 * torch.randn(action.shape, device=u.device, generator=gen)
    _, r, term, trunc, _ = env.step(a)
    rew_sum += r.double().sum()
    n_resets += int((term | trunc).sum())
    for s in sensors:
        force_sum += s._data.net_forces_w.double().abs().sum()
robot = u.scene["robot"]
print(
    f"[ENVCHECK] fast={args.fast_substeps} steps={args.check_steps} reward_sum={rew_sum.item():.6f} "
    f"force_sum={force_sum.item():.6f} resets={n_resets} joint_pos_sum={robot.data.joint_pos.double().sum().item():.6f}",
    flush=True,
)
print(f"[ENVPROFILE] task={args.task} num_envs={args.num_envs} decimation={u.cfg.decimation} steps={args.steps}", flush=True)
print(f"[ENVPROFILE] total wall per env.step: {total:.2f} ms (incl. timer syncs)", flush=True)
for k, v in sorted(acc_snapshot.items(), key=lambda kv: -kv[1]):
    print(f"[ENVPROFILE]   {k:40s} {v / args.steps:8.2f} ms", flush=True)
print(f"[ENVPROFILE] note: scene.update/write_data_to_sim run every substep; reset_idx may nest inside other phases", flush=True)
env.close()
simulation_app.close()
