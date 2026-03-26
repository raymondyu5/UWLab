import numpy as np
import torch
from torch.utils.data import Dataset

from uwlab_tasks.manager_based.manipulation.grasp.dataset.zarr_dataset import ZarrDataset


class CotrainZarrDataset(Dataset):
    """
    Wraps two ZarrDataset instances (sim + real) for cotraining.

    Each sample is drawn from sim with probability sim_ratio and from real
    with probability (1 - sim_ratio). Epoch length is defined by the sim dataset.

    Args:
        sim_dataset:  ZarrDataset for simulation data
        real_dataset: ZarrDataset for real-world data
        sim_ratio:    fraction of samples drawn from sim (default 0.95)
    """

    def __init__(self, sim_dataset: ZarrDataset, real_dataset: ZarrDataset, sim_ratio: float = 0.95):
        self.sim_dataset = sim_dataset
        self.real_dataset = real_dataset
        self.sim_ratio = sim_ratio

        sim_eps = sim_dataset.replay_buffer.n_episodes
        real_eps = real_dataset.replay_buffer.n_episodes
        sim_steps = sim_dataset.replay_buffer.episode_ends[-1]
        real_steps = real_dataset.replay_buffer.episode_ends[-1]
        print(
            f"CotrainZarrDataset: {sim_eps} sim episodes ({sim_steps} steps), "
            f"{real_eps} real episodes ({real_steps} steps), "
            f"sim_ratio={sim_ratio}"
        )

    def __len__(self) -> int:
        return len(self.sim_dataset)

    def __getitem__(self, idx: int):
        if np.random.random() < self.sim_ratio:
            return self.sim_dataset[idx]
        else:
            real_idx = np.random.randint(len(self.real_dataset))
            return self.real_dataset[real_idx]

    @property
    def action_dim(self):
        return self.sim_dataset.action_dim

    @property
    def low_obs_dim(self):
        return self.sim_dataset.low_obs_dim

    def get_normalizer(self):
        return self.sim_dataset.get_normalizer()

    def get_maniflow_normalizer(self, mode: str = "limits"):
        return self.sim_dataset.get_maniflow_normalizer(mode=mode)

    def get_validation_dataset(self) -> "CotrainZarrDataset":
        return CotrainZarrDataset(
            sim_dataset=self.sim_dataset.get_validation_dataset(),
            real_dataset=self.real_dataset.get_validation_dataset(),
            sim_ratio=self.sim_ratio,
        )
