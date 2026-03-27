# Wandb logging callbacks for RFS training.
#
# WandbNoisePredCallback  — logs noise/residual stats from the RFSWrapper each rollout.
# WandbRewardTermCallback — logs per-term Isaac reward sums each rollout.
# WandbOutputFormat       — forwards SB3 logger flushes (ep_rew_mean, losses, etc.) to wandb.

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter


class WandbNoisePredCallback(BaseCallback):
    """Logs PPO noise and residual prediction stats to wandb at each rollout end."""

    def __init__(self, rfs_env, verbose=0):
        super().__init__(verbose)
        self._rfs_env = rfs_env

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if wandb.run is None:
            return
        log_dict = {}
        noise = self._rfs_env.last_noise_flat
        if noise is not None:
            log_dict["noise/mean"] = noise.mean().item()
            log_dict["noise/std"] = noise.std().item()
            log_dict["noise/abs_mean"] = noise.abs().mean().item()
        residual = self._rfs_env.last_residual
        if residual is not None:
            log_dict["residual/mean"] = residual.mean().item()
            log_dict["residual/std"] = residual.std().item()
            log_dict["residual/abs_mean"] = residual.abs().mean().item()
        if log_dict:
            wandb.log(log_dict, step=self.num_timesteps)


class WandbRewardTermCallback(BaseCallback):
    """Logs individual Isaac reward terms to wandb at the end of each rollout."""

    def __init__(self, isaac_env, verbose=0):
        super().__init__(verbose)
        self._isaac_env = isaac_env

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if wandb.run is None:
            return
        mgr = self._isaac_env.unwrapped.reward_manager
        term_means = {
            name: val.mean().item()
            for name, val in mgr._episode_sums.items()
        }
        log_dict = {f"rewards/{name}": mean_val for name, mean_val in term_means.items()}
        log_dict["rewards/total"] = sum(term_means.values())
        for name, term_cfg in zip(mgr._term_names, mgr._term_cfgs):
            func = getattr(term_cfg, "func", None)
            obj = getattr(func, "__self__", None)
            if obj is not None and hasattr(obj, "_component_sums") and obj._component_count > 0:
                for comp, total in obj._component_sums.items():
                    log_dict[f"rewards/{name}/{comp}"] = total / obj._component_count
                obj._component_sums.clear()
                obj._component_count = 0
        wandb.log(log_dict, step=self.num_timesteps)


class WandbOutputFormat(KVWriter):
    """Forwards SB3 logger flushes directly to wandb.

    SB3 calls write() via _dump_logs() after both collect_rollouts() and train(),
    so ep_rew_mean, value_loss, etc. are all present at flush time.
    """

    def write(self, key_values, key_excluded, step=0):
        if wandb.run is None:
            return
        log_dict = {}
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "wandb" in excluded:
                continue
            if isinstance(value, (int, float, np.floating, np.integer)):
                log_dict[key] = float(value)
        if log_dict:
            wandb.log(log_dict, step=step)

    def close(self):
        pass
