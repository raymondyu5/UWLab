# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log training stats every n timesteps (via LogEveryNTimesteps callback).")
parser.add_argument("--sb3_log_interval", type=int, default=10, help="SB3 internal log_interval passed to agent.learn(): log after every N PPO updates (default 10 = every 10 rollouts).")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name. If set, enables wandb logging.")
parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name override.")
parser.add_argument("--eval_interval", type=int, default=50, help="Run eval every N rollouts (PPO updates).")
parser.add_argument("--eval_spawn", type=str, default=None, help="Spawn config name (from configs/eval/spawns/). If not set, uses random eval with 1 trial.")
parser.add_argument("--eval_spawn_dir", type=str, default=None, help="Directory containing spawn YAML files. Defaults to configs/eval/spawns relative to script.")
parser.add_argument("--eval_video", action="store_true", default=False, help="Record eval rollout videos via viewport (no per-env cameras needed, compatible with rl_mode).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import gymnasium as gym
import logging
import numpy as np
import os
import re
import random
from datetime import datetime
from uuid import uuid4

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)
# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device


    # generate uwlab-style run name: VanillaPPO-{short_task}_{MMDD}_{HHMM}_{uuid6}
    timestamp = datetime.now().strftime("%m%d_%H%M")
    short_task = re.sub(r"-v\d+$", "", args_cli.task or "ppo")
    short_task = re.sub(r"^UW-[^-]+-", "", short_task)  # strip "UW-FrankaLeap-"
    run_name = args_cli.wandb_run_name or f"VanillaPPO-{short_task}_{timestamp}_{uuid4().hex[:6]}"
    log_root_path = os.path.abspath(os.path.join("logs", "ppo"))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {run_name}")
    log_dir = os.path.join(log_root_path, run_name)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    need_render = args_cli.video or args_cli.eval_video
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if need_render else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # capture references before any wrappers (needed for PPOEvalCallback)
    isaac_env = env.unwrapped
    gym_env = env  # gym env with viewport render() support

    # wrap for video recording
    control_hz = 1 / (env_cfg.sim.dt * env_cfg.decimation) 
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "fps": control_hz
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # wandb setup
    if args_cli.wandb_project:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        command = " ".join(sys.orig_argv)
        wandb.init(
            project=args_cli.wandb_project,
            name=run_name,
            config={"command": command, **agent_cfg},
            sync_tensorboard=True,
            save_code=False,
        )
        print(f"[INFO] Wandb run: {wandb.run.get_url()}")

    # create agent from stable baselines
    agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
    if args_cli.checkpoint is not None:
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval)]
    if args_cli.wandb_project:
        callbacks.append(WandbCallback(verbose=2))

    # reward term + eval callbacks
    from ppo_eval_callback import PPOEvalCallback, WandbRewardTermCallback
    from uwlab.eval.spawn import SpawnCfg, load_spawn_cfg

    if args_cli.wandb_project:
        callbacks.append(WandbRewardTermCallback(isaac_env, verbose=0))

    if args_cli.eval_spawn is not None:
        spawn_dir = args_cli.eval_spawn_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "configs", "eval", "spawns"
        )
        spawn_cfg = load_spawn_cfg(args_cli.eval_spawn, spawn_dir)
    else:
        spawn_cfg = SpawnCfg(poses=[], num_trials=1)

    eval_callback = PPOEvalCallback(
        isaac_env=isaac_env,
        gym_env=gym_env,
        spawn_cfg=spawn_cfg,
        log_dir=log_dir,
        eval_interval=args_cli.eval_interval,
        record_scatter=True,
        record_video=args_cli.eval_video,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # train the agent
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=args_cli.sb3_log_interval,
        )
    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    print("Saving to:")
    print(os.path.join(log_dir, "model.zip"))

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    if args_cli.wandb_project:
        wandb.finish()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
