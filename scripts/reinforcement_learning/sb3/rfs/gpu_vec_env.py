import torch

from isaaclab_rl.sb3 import Sb3VecEnvWrapper


class GpuSb3VecEnvWrapper(Sb3VecEnvWrapper):
    """Sb3VecEnvWrapper that returns GPU tensors instead of numpy arrays.

    Keeps observations, rewards, and done flags on the simulation device
    throughout the rollout loop, eliminating per-step CPU<->GPU transfers.

    Requires RegularizedPPO with GpuDictRolloutBuffer(gpu_buffer=True).
    infos is returned as a flat dict (not a list) with batched GPU tensors:
        "episode":              {"r": (n_envs,), "l": (n_envs,), "mask": (n_envs,) bool}
        "terminal_observation": dict[str, (n_envs, ...)]  — full batched obs
        "terminal_mask":        (n_envs,) bool — True for envs that reset this step
        "TimeLimit.truncated":  (n_envs,) bool
    """

    def __init__(self, env):
        super().__init__(env)
        device = self.sim_device
        self._ep_rew_buf = torch.zeros(self.num_envs, device=device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=device)

    def reset(self) -> dict:
        obs_dict, _ = self.env.reset()
        self._ep_rew_buf.zero_()
        self._ep_len_buf.zero_()
        return self._process_obs(obs_dict)

    def step_wait(self):
        obs_dict, rew, terminated, truncated, extras = self.env.step(self._async_actions)
        dones = terminated | truncated

        obs = self._process_obs(obs_dict)
        rewards = rew.detach()
        terminated = terminated.detach()
        truncated = truncated.detach()
        dones = dones.detach()

        self._ep_rew_buf += rewards
        self._ep_len_buf += 1

        reset_ids = dones.nonzero(as_tuple=True)[0]
        infos = self._process_extras(obs, terminated, truncated, extras, reset_ids)

        self._ep_rew_buf[reset_ids] = 0.0
        self._ep_len_buf[reset_ids] = 0

        return obs, rewards, dones, infos

    def _process_obs(self, obs_dict):
        obs = obs_dict["policy"]
        if isinstance(obs, dict):
            for key, value in obs.items():
                if key in self.observation_processors:
                    obs[key] = self.observation_processors[key](value)
            return obs
        if isinstance(obs, torch.Tensor):
            return obs
        raise NotImplementedError(f"Unsupported obs type: {type(obs)}")

    def _process_extras(self, obs, terminated, truncated, extras, reset_ids):
        device = self.sim_device
        time_limit_truncated = truncated & ~terminated

        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        if reset_ids.numel() > 0:
            reset_mask[reset_ids] = True

        ep_rew = torch.zeros(self.num_envs, device=device)
        ep_len = torch.zeros(self.num_envs, device=device)
        if reset_ids.numel() > 0:
            ep_rew[reset_mask] = self._ep_rew_buf[reset_mask]
            ep_len[reset_mask] = self._ep_len_buf[reset_mask]

        return {
            "episode": {"r": ep_rew, "l": ep_len, "mask": reset_mask},
            "TimeLimit.truncated": time_limit_truncated,
            "terminal_observation": obs,
            "terminal_mask": reset_mask,
        }
