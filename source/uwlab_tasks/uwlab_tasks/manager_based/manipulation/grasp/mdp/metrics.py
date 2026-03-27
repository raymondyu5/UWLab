from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class MetricSpec:
    """A named metric callable producing per-env values."""

    name: str
    fn: Callable[[Any], torch.Tensor]


class EnvMetrics:
    """Runtime metric access for eval/diagnostics.

    Metric callables are expected to take as input the environment and return a tensor with shape (num_envs,)
    (or something safely squeezeable to that).

    So metric_fn(env) -> torch.Tensor(num_envs,)
    """

    def __init__(self, specs: Mapping[str, Any], env: Any | None = None):
        """
        Args:
            specs: Either a mapping of name->callable, or name->MetricSpec.
            env: Optional env reference; if provided, `get_metrics()` does not
                 require an explicit env argument.
        """
        self._env_ref = weakref.ref(env) if env is not None else None
        self._specs: dict[str, MetricSpec] = self._normalize_specs(specs)

    @staticmethod
    def _normalize_specs(specs: Mapping[str, Any]) -> dict[str, MetricSpec]:
        normalized: dict[str, MetricSpec] = {}
        for name, spec in specs.items():
            if isinstance(spec, MetricSpec):
                normalized[spec.name] = spec
            elif callable(spec):
                normalized[name] = MetricSpec(name=name, fn=spec)
            else:
                raise TypeError(
                    f"Invalid metric spec for {name!r}: expected MetricSpec or callable, got {type(spec)}"
                )
        return normalized

    def _get_env(self):
        if self._env_ref is None:
            raise RuntimeError("EnvMetrics has no env bound; call get_metrics(env=...) or bind env in constructor.")
        env = self._env_ref()
        if env is None:
            raise RuntimeError("EnvMetrics env reference is no longer valid.")
        return env

    def get_metrics(self, env: Any | None = None) -> dict[str, np.ndarray]:
        """Compute all metrics and return them as NumPy arrays on CPU."""
        if env is None:
            env = self._get_env()

        if not self._specs:
            raise RuntimeError(
                "EnvMetrics has no metrics_spec attached to this env. "
                "Set `metrics_spec` in the task config (e.g. bottle_pour.py)."
            )

        out: dict[str, np.ndarray] = {}
        for name, spec in self._specs.items():
            t = spec.fn(env)
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Metric {name!r} must return a torch.Tensor, got {type(t)}")

            t = t.detach().cpu()
            if t.numel() == 0:
                raise ValueError(f"Metric {name!r} returned an empty tensor")

            t = t.squeeze()
            if t.dim() != 1:
                raise ValueError(
                    f"Metric {name!r} expected shape (num_envs,), got {tuple(t.shape)}"
                )

            # Most eval metrics are boolean-ish ("is_*"). Convert numeric masks into bools.
            if t.dtype == torch.bool:
                arr = t.numpy().astype(bool, copy=False)
            elif t.dtype in (torch.int32, torch.int64, torch.uint8):
                arr = (t.numpy().astype(np.int64, copy=False) != 0)
            else:
                arr = (t.numpy() > 0.5)

            out[name] = arr
        return out

