"""
Convert a real Franka LEAP episode (zarr or npy) to sysid_data format (.pt).

The output .pt file can be used with sysid_franka_leap.py --real_data.

Usage:
    python scripts/sysid/convert_episode_to_sysid_data.py --episode path/to/episode_0.zarr --output sysid_data_franka_leap.pt --dt 0.05
"""

import argparse
import torch
from uwlab_tasks.utils.trajectory_utils import load_real_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=str, required=True, help="Path to episode (zarr or npy)")
    parser.add_argument("--output", type=str, default="sysid_data_franka_leap.pt")
    parser.add_argument("--dt", type=float, default=0.05, help="Control period in seconds (e.g. 0.05 for 20Hz)")
    args = parser.parse_args()

    episode = load_real_episode(args.episode)
    obs_list = episode["obs"]
    actions = episode["actions"]

    T = min(len(obs_list), len(actions))
    if T == 0:
        raise ValueError("Episode has no steps")

    obs_list = obs_list[:T]
    actions = actions[:T]

    arm_joint_pos = torch.tensor(
        [obs["joint_positions"][:7] for obs in obs_list],
        dtype=torch.float32,
    )
    hand_joint_pos = torch.tensor(
        [obs["gripper_position"] for obs in obs_list],
        dtype=torch.float32,
    )
    arm_joint_pos_target = torch.tensor(
        [obs["ik_joint_pos_desired"] for obs in obs_list],
        dtype=torch.float32,
    )
    hand_actions = torch.tensor(
        [a[-16:] for a in actions],
        dtype=torch.float32,
    )

    data = {
        "arm_joint_pos": arm_joint_pos,
        "arm_joint_pos_target": arm_joint_pos_target,
        "hand_actions": hand_actions,
        "initial_arm_joint_pos": arm_joint_pos[0].clone(),
        "initial_hand_joint_pos": hand_joint_pos[0].clone(),
        "dt": args.dt,
    }
    torch.save(data, args.output)
    print(f"Saved {T} samples to {args.output} (dt={args.dt}s)")


if __name__ == "__main__":
    main()
