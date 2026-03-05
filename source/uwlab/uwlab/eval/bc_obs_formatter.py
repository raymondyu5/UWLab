from typing import Dict, List
import numpy as np
import torch


class BCObsFormatter:
    """
    Formats raw Isaac Sim env observations into the dict expected by CFMPCDPolicy.

    At train time, SimZarrDataset concatenates obs_keys -> agent_pos and
    image_keys -> pcd tensors. This class replicates that at eval time.

    Args:
        obs_keys:         list of env obs keys to concatenate into agent_pos
                          e.g. ["right_hand_joint_pos", "right_ee_pose"]
                          These are keys under obs["policy"][key]
        image_keys:       list of pcd keys, e.g. ["seg_pc"]
        downsample_points: number of points to randomly sample from each PCD
        device:           torch device for output tensors
    """

    def __init__(
        self,
        obs_keys: List[str],
        image_keys: List[str],
        downsample_points: int,
        device: torch.device,
    ):
        self.obs_keys = obs_keys
        self.image_keys = image_keys
        self.downsample_points = downsample_points
        self.device = device

    def format(self, raw_obs: dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            raw_obs: env observation dict, e.g. obs["policy"] from Isaac Sim.
                     Values are (B, D) or (B, 3, N) tensors already on device.

        Returns:
            {"agent_pos": (B, 1, D_total), "seg_pc": (B, 1, 3, downsample_points), ...}
            Shape matches what CFMPCDPolicy.predict_action expects (with n_obs_steps=1).
        """
        policy_obs = raw_obs

        # --- agent_pos: concatenate obs_keys ---
        obs_parts = []
        for key in self.obs_keys:
            val = policy_obs[key]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).to(self.device)
            obs_parts.append(val.float())

        agent_pos = torch.cat(obs_parts, dim=-1)  # (B, D_total)
        agent_pos = agent_pos.unsqueeze(1)        # (B, 1, D_total)

        result = {"agent_pos": agent_pos}

        # --- pcd: random downsample, add time dim ---
        for key in self.image_keys:
            pcd = policy_obs[key]  # (B, 3, N) from env
            if isinstance(pcd, np.ndarray):
                pcd = torch.from_numpy(pcd).to(self.device)
            pcd = pcd.float()

            N = pcd.shape[-1]
            if N > self.downsample_points:
                perm = torch.randperm(N, device=self.device)[:self.downsample_points]
                pcd = pcd[:, :, perm]  # (B, 3, downsample_points)

            pcd = pcd.unsqueeze(1)  # (B, 1, 3, downsample_points)
            result[key] = pcd

        return result