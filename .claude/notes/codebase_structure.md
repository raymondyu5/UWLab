# Codebase Structure

## Python Packages (source/)

```
source/
  uwlab/             # Core framework: envs, policies, controllers, utils
  uwlab_assets/      # Robot asset definitions (USD, joint limits, actions)
  uwlab_rl/          # RL framework extensions (RSL-RL, SKRL)
  uwlab_tasks/       # Task environment definitions + gymnasium registrations
```

## Key File Paths

### Task Definition (PourBottle)
- Task config: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/grasp/config/franka_leap/tasks/bottle_pour.py`
- Base grasp env: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/grasp/config/franka_leap/grasp_franka_leap.py`
- Grasp env base: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/grasp/grasp_env.py`
- Rewards: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/grasp/config/franka_leap/tasks/rewards/pour_rewards.py`
- Gym registration: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/grasp/config/franka_leap/__init__.py`

### Robot Assets (Franka-LEAP)
- Main definitions: `source/uwlab_assets/uwlab_assets/robots/franka_leap/franka_leap.py`
- Action configs: `source/uwlab_assets/uwlab_assets/robots/franka_leap/actions.py`
- Joint limits, EE offset, reset poses all in `franka_leap.py`

### RFS Training
- Train script: `scripts/reinforcement_learning/sb3/rfs/train.py`
- RFS wrapper: `scripts/reinforcement_learning/sb3/rfs/wrapper.py`
- Eval callback: `scripts/reinforcement_learning/sb3/rfs/eval_callback.py`

### BC Policy
- CFM policy: `source/uwlab/uwlab/policy/cfm_pcd_policy.py`
- PointNet encoder: `source/uwlab/uwlab/policy/backbone/pcd/pointnet.py`
- Multi-PCD encoder: `source/uwlab/uwlab/policy/backbone/multi_pcd_obs_encoder.py`
- BC obs formatter: `source/uwlab/uwlab/eval/bc_obs_formatter.py`

### Configs
- RFS configs: `configs/rl/`
- BC training configs: `configs/bc/`
- Eval spawn configs: `configs/eval/spawns/`

## Eval Spawn Configs (`configs/eval/spawns/`)

| Config | Description |
|--------|-------------|
| `random_1_trial.yaml` | 1 trial at random pose (poses: [], num_trials: 1) |
| `random.yaml` | Multiple random trials |
| `cardinal_3x3.yaml` | 3x3 grid of spawn positions |
| `bottle_pour_narrow.yaml` | Narrow ±5cm grid around center (for pour task) |

## Third-Party Integrations

- `third_party/diffusion_policy/`: Vision-based diffusion policy code (from Columbia)
- `third_party/ManiFlow_Policy/`: ManiFlow policy implementation
- `third_party/pip_packages/`: Extra packages (dill, etc.) needed since Isaac resets PYTHONPATH

## Isaac Lab Integration

The codebase is built on **NVIDIA Isaac Lab** (Isaac Sim 5.1.0):
- `ManagerBasedRLEnv`: Isaac Lab's base RL environment class
- `ManagerBasedRLEnvCfg`: declarative config class using `@configclass`
- **Managers**: ObservationManager, ActionManager, RewardManager, EventManager, TerminationManager
- Each manager has "terms" (ObsTerm, RewTerm, EventTerm, DoneTerm)
- Terms are functions that take `env` and return tensors

### Key Isaac Lab Patterns

```python
@configclass
class TaskCfg(BaseEnvCfg):
    observations: ObsCfg = ObsCfg()
    rewards: RewardCfg = RewardCfg()
    events: EventCfg = EventCfg()
    terminations: TermCfg = TermCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Add task-specific terms here
```

All terms are defined **declaratively** in `__post_init__` — the managers are instantiated at runtime by Isaac Lab.

## MDP Utilities

Custom MDP functions in `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/grasp/mdp/`:
- `reset_object_pose`: randomized object placement on reset
- `reset_robot_joints`: reset robot to initial configuration
- `reset_table_block`: optional table height randomization
- `capture_bottle_reset_height`: store bottle z after reset
- `CachedSamplePC`: mesh-based segmented point cloud sampling
- `bottle_dropped`, `bottle_too_far`, `cup_toppled`: termination functions

## Franka-LEAP Robot

- **Arm**: Panda (7 DOF), joints `panda_joint1` through `panda_joint7`
- **Hand**: LEAP hand (16 DOF), joints `j0` through `j15`
- **Total**: 23 DOF
- **EE body**: `panda_link7` with an offset (FRANKA_LEAP_EE_OFFSET)
- **Arm limits**: Conservative joint limits stored in `FRANKA_LEAP_ARM_JOINT_LIMITS`
- **Reset pose**: Arm `[0.311, 0.004, -0.311, -2.051, 0.001, 2.055, 0.781]`, hand at HAND_RESET

## Multi-Env Vectorization

Isaac Lab natively supports thousands of parallel envs with GPU-resident physics:
- All tensors are on CUDA
- Env origins: each env has a separate `env_origins` 3D offset
- Object positions are stored in world frame; local frame = `pos_w - env_origin`
- SB3 interface: `GpuSb3VecEnvWrapper` keeps tensors on GPU instead of converting to numpy
