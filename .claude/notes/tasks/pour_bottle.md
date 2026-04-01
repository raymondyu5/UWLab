# Task: PourBottle (UW-FrankaLeap-PourBottle-*)

**Source**: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/grasp/config/franka_leap/tasks/bottle_pour.py`

## Registered Environments

| ID | Config Class |
|----|-------------|
| `UW-FrankaLeap-PourBottle-JointAbs-v0` | `PourBottleFrankaLeapJointAbsCfg` |
| `UW-FrankaLeap-PourBottle-JointRel-v0` | `PourBottleFrankaLeapJointRelCfg` |
| `UW-FrankaLeap-PourBottle-IkRel-v0` | `PourBottleFrankaLeapIkRelCfg` |
| `UW-FrankaLeap-PourBottle-IkAbs-v0` | `PourBottleFrankaLeapIkAbsCfg` |

## Task Goal

Grasp the bourbon bottle and pour it over the pink cup — specifically, position the bottle cap tip within 4cm XY of the cup center, at a height 17–30cm above the cup.

**Success** (`is_success`): cap tip XY dist to cup center < 4cm AND cap tip z ∈ [cup_z + 0.17m, cup_z + 0.30m]

## Key Geometry

- **Bottle cap offset**: `(-0.132179, 0.0, 0.0)` in bottle local frame — cap is at X = -13.2cm
- **Cup height**: ~15cm tall; cup origin is at its base, so cup rim ≈ cup_z + 0.15m
- **Healthy pouring z band**: cap tip z ∈ [cup_z + 0.17m, cup_z + 0.30m] (2–15cm above rim)
- **Near-miss radius**: 15cm (broad reward shaping zone)
- **Success radius**: 4cm (tight zone)

## Timing

- Horizon: 176 steps
- Control: 10 Hz (60 Hz / 6 decimation)
- Episode length: (176 + 15 warmup) × 6 × (1/60) ≈ 19.1 seconds

## Scene

- **Bottle**: `assets/bourbon/rigid_object.usd`, scale (1.1, 1.1, 1.1) — inherited from `GraspBottleSceneCfg`
- **Pink cup**: `assets/pink_cup/rigid_object.usd`, scale (1.0, 1.0, 1.0) — added by `PourBottleSceneCfg`
- **Table block**: `assets/table/table_block.usd`, kinematic, scale (1.2, 1.0, 0.10), centered at `(0.55, -0.10, 0.0)`
- `distill_include_entity_names = ("robot", "grasp_object", "pink_cup")` — both objects in seg_pc

## Spawn / Reset Randomizations

Spawn positions are calibrated to real-world workspace bounds (comments in source):
- Bottle: real x ∈ [0.42, 0.64], y ∈ [-0.20, -0.01] → center (0.53, -0.105), range ±(0.11, 0.095)
- Cup: real x ∈ [0.41, 0.65], y ∈ [0.17, 0.29] → center (0.53, 0.23), range ±(0.12, 0.06)

### Bottle
- Default pos: `(0.53, -0.105, 0.11)` (env-local)
- Default rot: `BOTTLE_SPAWN_ROT = (0.707, 0.0, 0.0, -0.707)`
- Pose range:
  - x: ±0.11 m → x ∈ [0.42, 0.64]
  - y: ±0.095 m → y ∈ [-0.20, -0.01]
  - z/roll/pitch/yaw: 0

### Pink Cup
- Default pos: `(0.53, 0.23, 0.07)` (env-local)
- Default rot: `(0.707, 0.707, 0.0, 0.0)` — upright
- Pose range:
  - x: ±0.12 m → x ∈ [0.41, 0.65]
  - y: ±0.06 m → y ∈ [0.17, 0.29]
  - z/roll/pitch/yaw: 0

### Table Block Height
- `table_z_range = (0.0, 0.0)` — **no table height randomization** (disabled for pour task)

### Robot Reset
- Arm + hand reset to `ARM_RESET + HAND_RESET`

### No Material/Mass Randomization
Unlike the grasp tasks, pour has no `randomize_object_material` or `randomize_object_mass` events.

### Reset References Capture (`capture_bottle_reset_height`)
- Runs AFTER `reset_object` AND `reset_cup_object`
- Records bottle z at reset for downstream use

## Observations

| Key | Shape | Description |
|-----|-------|-------------|
| `joint_pos` | (23,) | All joint positions |
| `arm_joint_pos` | (7,) | Arm joints only |
| `hand_joint_pos` | (16,) | LEAP hand joints |
| `ee_pose` | (7,) | EE pos+quat (world frame) |
| `cup_pose` | (3,) | Pink cup XYZ position (env-local) |
| `target_object_pose` | (7,) | Computed target bottle pose: cap 10cm above cup, centered (env-local) |
| `manipulated_object_pose` | (7,) | Bottle pos+quat (env-local) |
| `seg_pc` | (3, 2048) | PCD: arm (256) + hand (64) + bottle (128) + cup (128), **no noise** (pcd_noise=0.0) |

### Target Object Pose Computation
```
target_bottle_pos = cup_center_xy, z = cup_top_z + 0.10
# Then back-compute bottle root position so cap offset lands at desired_cap_pos
```
The target pose gives the policy the "pouring configuration" as a goal signal.

### Key difference from grasp tasks
- `pcd_noise=0.0` (no noise added to PCD during RL training)
- Has both bottle AND cup in seg_pc (128 pts each)
- Cup pose and target pose are explicit observations

## Reward Function

Uses **separate `RewTerm` entries** (unlike grasp tasks which use a single combined GraspReward). Each term is independently weighted:

| Term | Weight | Function | Description |
|------|--------|----------|-------------|
| `grasped` | +1.0 | `pour_grasped` | Binary: cap tip z > 0.05m |
| `xy_healthy` | +2.0 | `pour_xy_healthy` | Dense XY alignment × is_healthy_z gate |
| `xy_near_miss` | +5.0 | `pour_xy_near_miss` | Dense XY alignment × is_near_miss gate |
| `success` | +10.0 | `pour_success` | Binary: is_success |
| `cup_topple` | -10.0 | `pour_cup_topple` | Binary: cup tilted >30° |
| `joint_vel` | -1e-3 | `pour_joint_vel_l2` | L2 joint velocity |
| `joint_limit` | -6e-1 | `pour_joint_pos_limits` | Joint near limit (soft, 90% of limit) |
| `action_rate` | -5e-3 | `pour_action_rate_l2` | L2 action change per step |

**Max possible reward per step**: 1 + 2 + 5 + 10 = 18

### Dense XY Reward
```python
xy_reward = clip(1.0 - xy_dist / MAX_XY_DIST, 0, 1)
# MAX_XY_DIST = 0.60m
```
The XY reward is gated:
- `pour_xy_healthy`: requires `is_healthy_z` — cap tip z must be in [cup_z+17cm, cup_z+30cm]
- `pour_xy_near_miss`: requires `is_near_miss` — also requires XY dist < 15cm

### Reward Shaping Philosophy
Progressive gating: reward is designed so the policy must:
1. First grasp (lift tip above 5cm)
2. Then reach a healthy z height (to unlock XY rewards)
3. Then approach the cup center XY (to get near-miss bonus)
4. Finally be within 4cm (success)

## Metrics (`metrics_spec`)

Used by eval scripts to log beyond reward:

| Metric | Function | Description |
|--------|----------|-------------|
| `is_success` | `is_success(env)` | Cap tip XY <4cm, z in band |
| `is_grasped` | `is_grasped(env)` | Cap tip z > 5cm |
| `is_healthy_z` | `is_healthy_z(env)` | Cap tip z in pouring band |
| `is_near_miss` | `is_near_miss(env)` | Within 15cm XY AND healthy z |

## Terminations

| Condition | Parameters | Type |
|-----------|------------|------|
| `time_out` | — | 176 steps |
| `bottle_too_far` | `max_xy_dist=1.0m` | Early stop: bottle XY > 1m from spawn |
| `cup_toppled` | `angle_thresh_rad=0.524` (~30°) | Early stop: cup knocked over |

Note: `bottle_dropped` termination exists in code but is **commented out** — the robot is not penalized for dropping the bottle (only for losing it too far or knocking the cup).

## Action Variants

| Suffix | Action Space | Warmup |
|--------|-------------|--------|
| `JointAbs` | 23D absolute joint positions | Hold at ARM_RESET + HAND_RESET |
| `JointRel` | 7D delta arm + 16D abs hand | Zero action |
| `IkRel` | 6D delta EE + 16D abs hand | Zero action |
| `IkAbs` | 7D abs EE pose + 16D abs hand | Hold current EE + current hand |

## Inheritance Chain

```
PourBottleFrankaLeapJointAbsCfg
  └── PourBottleFrankaLeapCfg
        └── FrankaLeapGraspEnvCfg
              └── GraspEnvCfg
                    └── ManagerBasedRLEnvCfg

PourBottleSceneCfg          (adds pink_cup to scene)
  └── GraspBottleSceneCfg   (has bottle + table_block)
        └── FrankaLeapGraspSceneCfg  (has robot + cameras)
              └── GraspSceneCfg     (has table + ground + dome light)
```

## Key Differences vs Grasp Tasks

| Feature | PourBottle | GraspBottle / GraspPinkCup |
|---------|------------|---------------------------|
| Goal | Pour (cap over cup) | Lift (height threshold) |
| Objects in scene | Bottle + Pink cup | Single object |
| Table z range | 0.0, 0.0 (fixed) | 0.0, 0.05 (randomized) |
| Material/mass random | No | Yes (every 800 steps) |
| PCD noise | 0.0 | 0.02 |
| Reward structure | Separate `RewTerm` per component | Single `GraspReward` callable |
| Terminations | timeout + bottle_too_far + cup_toppled | timeout only |
| Has `metrics_spec` | Yes | No (uses `is_success` method instead) |
| Horizon | 176 | 180 (bottle) / 200 (cup) |
