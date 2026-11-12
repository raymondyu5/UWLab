Research project: Isaac Lab (UWLab) sim-to-real for dexterous manipulation. Package notes are in `.claude/notes/` (read the relevant ones before diving into code; see `.claude/CLAUDE.MD`).

IMPORTANT (overrides `.claude/CLAUDE.MD`'s "cannot run python" for the klone side): you work locally in ~/research/UWLab.
- Locally you can only edit, `python -m py_compile`, and use git. The isaaclab python lives in an Apptainer image on klone, so run everything via `scripts/cluster/` (push, submit, status, pull). See docs/claude/info/remote-workflow.md.
- Local tree is the source of truth. Never use git on klone for syncing. Commit locally; do NOT push to origin (raymondyu5/UWLab) without asking.
- Keep notes in docs/claude/{experiments,handoffs,ideas,implementations,info}. After each experiment, write a brief note in experiments/ (what was run, results, pointers to logs/checkpoints).
- Keep outputs in logs/ (gitignored). Keep the repo clean.
- Cluster: prefer gpu-l40 (then l40s/a40) with account weirdlab, then ckpt.
