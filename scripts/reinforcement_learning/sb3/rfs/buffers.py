import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples


class GpuDictRolloutBuffer(DictRolloutBuffer):
    """DictRolloutBuffer that stores all data as GPU tensors.

    Eliminates CPU<->GPU transfers during rollout collection by keeping
    observations, actions, rewards, values, and advantages on the device
    throughout. Falls back to standard numpy behavior when gpu_buffer=False.

    Must be used together with GpuSb3VecEnvWrapper (which returns GPU tensors
    from step/reset) and the collect_rollouts override in RegularizedPPO.
    """

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        gae_lambda=1,
        gamma=0.99,
        n_envs=1,
        gpu_buffer=True,
        **kwargs,
    ):
        # Must be set before super().__init__() because __init__ calls reset().
        self.gpu_buffer = gpu_buffer
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
            **kwargs,
        )

    def reset(self) -> None:
        if not self.gpu_buffer:
            super().reset()
            return

        self.observations = {
            key: th.zeros(
                (self.buffer_size, self.n_envs, *shape),
                dtype=th.float32,
                device=self.device,
            )
            for key, shape in self.obs_shape.items()
        }
        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=th.float32,
            device=self.device,
        )
        self.rewards = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.returns = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.episode_starts = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.values = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.log_probs = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.advantages = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, episode_start, value, log_prob) -> None:
        if not self.gpu_buffer:
            return super().add(obs, action, reward, episode_start, value, log_prob)

        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = obs[key].clone()
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        action = action.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos] = action.clone()
        self.rewards[self.pos] = reward.clone()

        if isinstance(episode_start, th.Tensor):
            self.episode_starts[self.pos] = episode_start.to(
                device=self.device, dtype=th.float32
            )
        else:
            self.episode_starts[self.pos] = th.tensor(
                episode_start, device=self.device, dtype=th.float32
            )

        self.values[self.pos] = value.clone().flatten()
        self.log_probs[self.pos] = log_prob.clone()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values, dones) -> None:
        if not self.gpu_buffer:
            return super().compute_returns_and_advantage(last_values, dones)

        last_values = last_values.clone().flatten()

        if isinstance(dones, th.Tensor):
            dones_f = dones.to(dtype=th.float32)
        else:
            dones_f = th.tensor(dones, dtype=th.float32, device=self.device)

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones_f
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def _get_samples(self, batch_inds, env=None):
        if not self.gpu_buffer:
            return super()._get_samples(batch_inds, env=env)

        return DictRolloutBufferSamples(
            observations={key: obs[batch_inds] for key, obs in self.observations.items()},
            actions=self.actions[batch_inds],
            old_values=self.values[batch_inds].flatten(),
            old_log_prob=self.log_probs[batch_inds].flatten(),
            advantages=self.advantages[batch_inds].flatten(),
            returns=self.returns[batch_inds].flatten(),
        )
