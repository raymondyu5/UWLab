# RFS eval, all tasks (2026-10-30)

Checkpoints copied from Raymond's `/gscratch/weirdlab/raymond/UWLab/logs` into `logs/rfs/` and `logs/base/` (klone). Eval YAMLs: `configs/eval/<name>_rfs_joint_abs.yaml` (each holds `rfs_checkpoint`).
Run: `scripts/cluster/eval_rfs.sbatch` (NAME=<name>), `random_5_trials`, 64 envs, 5 episodes. Outputs: `logs/eval/<name>_rfs/eval/step_0000000000/{results.json,videos}`.
Slurm reports COMPLETED even when the script crashed; check `results.json` / Traceback.

| Task | ckpt | success_rate | success_ever |
|---|---|---|---|
| GraspBottleRandomResets | 350 | 0.984 | 0.994 |
| GraspPinkCupRandomResets | 350 | 0.978 | 0.981 |
| ScrewLightbulb-RealLightbulb | 150 | 0.766 | 0.772 |
| GraspCreditCard | 300 | 0.734 | 0.894 |
| GraspCube | 150 | 0.703 | 0.744 |
| PlateInDishRack | 400 | 0.659 | 0.750 |
| SoccerPush | 450 | 0.494 | 0.872 |
| CupPour | 200 | 0.184 | 0.184 |

Fixes: SoccerPush/CupPour envs gained a `ball_velocity` obs after the ckpts were trained (commit 7b69641) -> "Observation spaces do not match". Removed the ObsTerm in `soccer_push.py` and `cup_pour.py` (soccer then ran).
CupPour: ckpt trained with a critic-only `target_object_pose` obs (`cup_pour:obs_target_object_pose`, 7-dim) that no longer exists and whose definition is lost; added a zeros placeholder in `cup_pour.py` (eval-only, values are made up).
