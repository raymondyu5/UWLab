from typing import List, Optional
import os
import yaml
from dataclasses import dataclass, field


@dataclass
class SpawnPose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    name: Optional[str] = None


@dataclass
class SpawnCfg:
    poses: List[SpawnPose] = field(default_factory=list)
    num_trials: int = 1


def load_spawn_cfg(spawn_name: str, spawn_dir: str) -> SpawnCfg:
    """
    Load a spawn YAML from configs/eval/spawns/{spawn_name}.yaml.

    YAML format:
        poses:
          - {name: center, x: 0.0, y: 0.0, yaw: 0.0}
          - ...
        num_trials: 4   # repeat each pose N times

    For random spawns (no poses), just set num_trials and omit poses.
    """
    path = os.path.join(spawn_dir, f"{spawn_name}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Spawn config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    poses = []
    for p in data.get("poses", []):
        poses.append(SpawnPose(
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            yaw=float(p.get("yaw", 0.0)),
            name=p.get("name", None),
        ))

    return SpawnCfg(
        poses=poses,
        num_trials=int(data.get("num_trials", 1)),
    )