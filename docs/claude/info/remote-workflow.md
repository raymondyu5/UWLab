# Local <-> klone workflow (agent-driven)

Edit locally, rsync to klone, run via Slurm + Apptainer, pull results back. Scripts in `scripts/cluster/` use `ssh klone`
(ControlMaster socket, no prompts). Settings (host, remote dir, excludes) live in `scripts/cluster/config.sh`.
There is no apptainer locally: **all python runs on klone inside the container**.

| Step | Command |
|---|---|
| Preview sync | `scripts/cluster/push.sh --dry-run` (checksum diff, slow on first run; `--delete` removes remote files missing locally) |
| Sync | `scripts/cluster/push.sh` |
| Sync + sbatch | `scripts/cluster/submit.sh <sbatch file> [sbatch args]` |
| Queue / recent jobs | `scripts/cluster/status.sh` |
| Job output | `scripts/cluster/status.sh --tail JOBID [N]` (reads `logs/slurm/*-JOBID.out`) |
| Fetch results | `scripts/cluster/pull.sh [--no-ckpt] [run-name ...]` (names are dirs under `logs/`) |
| Cancel | `scripts/cluster/cancel.sh JOBID...` |
| Smoke test | `scripts/cluster/submit.sh scripts/cluster/smoke.sbatch` |

Writing an experiment: copy `scripts/cluster/sbatch_template.sbatch`; it calls `bash scripts/cluster/run_in_container.sh "<cmd>"`,
which applies the same binds as `/gscratch/weirdlab/will/uwlab_overlay.sh` (source, scripts, assets, third_party, configs,
logs, isaac caches, outputs, data_storage, per-job /tmp/isaaclab) and runs `<cmd>` in `/workspace/uwlab`. Use `./uwlab.sh -p <script.py>` for python.

Notes
- Local tree is the source of truth. Push is one-way rsync (git is not used on klone). `logs/` is never pushed or deleted remotely.
- Remote `/mmfs1/gscratch/weirdlab/will/UWLab`; image/caches/overlay in `/gscratch/weirdlab/will/UWLab_Docker` (never synced).
- `logs/` lives inside the repo dir on klone (moved from `UWLab_Docker/logs`, which is now a symlink). Container sees it as `/workspace/uwlab/logs`.
- `commands/**` sbatch scripts written by Raymond hardcode `/gscratch/weirdlab/raymond/...`: port to Will's paths / `run_in_container.sh` before use.
- If scripts print "cannot reach klone non-interactively", the user must run `! ssh klone` (Duo).
- Never run compute on login nodes; only sbatch/squeue/sacct/scancel/tail go over ssh.
- `run_in_container.sh` mounts `UWLab_Docker/overlay.img` read-only by default: it holds pip packages missing from the bare sif (e.g. `torchcfm`). `USE_OVERLAY=0` disables. Install new packages interactively with `uwlab_overlay.sh --write` (one writer at a time).
- Partition notes: weirdlab's `gpu-l40` often shows `AssocGrpGRES` (group quota full). UWLab needs L40/L40S (not A40); `ckpt-all` also has L40/L40S (preemptible).
