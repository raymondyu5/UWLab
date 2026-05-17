from __future__ import annotations

import torch


def ball_in_cup(env, ball_name: str, cup_name: str, cup_inner_radius: float, cup_height: float) -> torch.Tensor:
    """Sparse +1 when ball XY is within cup radius and Z is between cup bottom and rim."""
    ball_pos = env.scene[ball_name].data.root_pos_w - env.scene.env_origins
    cup_pos = env.scene[cup_name].data.root_pos_w - env.scene.env_origins
    xy_dist = torch.norm(ball_pos[:, :2] - cup_pos[:, :2], dim=1)
    z_above = ball_pos[:, 2] - cup_pos[:, 2]
    in_xy = xy_dist < cup_inner_radius
    in_z = (z_above > 0.0) & (z_above < cup_height)
    return (in_xy & in_z).float()


def ball_proximity_to_cup(env, ball_name: str, cup_name: str, max_dist: float = 0.5) -> torch.Tensor:
    """Dense reward: 1 - (dist / max_dist), clamped [0, 1]."""
    ball_pos = env.scene[ball_name].data.root_pos_w - env.scene.env_origins
    cup_pos = env.scene[cup_name].data.root_pos_w - env.scene.env_origins
    dist = torch.norm(ball_pos[:, :3] - cup_pos[:, :3], dim=1)
    return torch.clamp(1.0 - dist / max_dist, min=0.0, max=1.0)


def ball_too_far(env, ball_name: str, max_xy_dist: float = 1.0) -> torch.Tensor:
    """Termination: ball has flown too far from env origin."""
    ball_pos = env.scene[ball_name].data.root_pos_w - env.scene.env_origins
    xy_dist = torch.norm(ball_pos[:, :2], dim=1)
    return xy_dist > max_xy_dist


def reset_ball_in_cup(env, env_ids, ball_name: str, cup_name: str,
                      x_offset: float = 0.0, y_offset: float = 0.0, z_offset: float = 0.05):
    """Place ball above the cup at the cup's current position + offsets."""
    ball = env.scene[ball_name]
    cup = env.scene[cup_name]
    cup_pos = cup.data.root_state_w[env_ids, :3].clone()
    ball_pos = cup_pos.clone()
    ball_pos[:, 0] += x_offset
    ball_pos[:, 1] += y_offset
    ball_pos[:, 2] += z_offset
    ball_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).expand(len(env_ids), -1)
    ball.write_root_pose_to_sim(torch.cat([ball_pos, ball_quat], dim=-1), env_ids)
    ball.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids)
