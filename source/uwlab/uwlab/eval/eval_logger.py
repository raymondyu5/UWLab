from typing import Dict, List, Optional, Tuple
import os
import json
import numpy as np
import matplotlib.pyplot as plt


class EvalLogger:
    """
    Collects per-step and per-episode data during BC policy evaluation,
    then writes results, trajectory plots, and optional success heatmaps.

    Usage:
        logger = EvalLogger(output_dir, record_video=False, record_plots=True)
        for each episode:
            logger.begin_episode(spawn_name, spawn_pose)
            for each step:
                logger.record_step(ee_pose, object_pose, action)
            logger.end_episode(success)
        logger.finalize()
    """

    def __init__(
        self,
        output_dir: str,
        record_video: bool = False,
        record_plots: bool = True,
    ):
        self.output_dir = output_dir
        self.record_video = record_video
        self.record_plots = record_plots

        os.makedirs(output_dir, exist_ok=True)
        if record_video:
            os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)

        self._episodes: List[dict] = []
        self._current: Optional[dict] = None

    def begin_episode(self, spawn_name: Optional[str] = None, spawn_pose: Optional[dict] = None):
        self._current = {
            "spawn_name": spawn_name,
            "spawn_pose": spawn_pose,
            "ee_poses": [],
            "object_poses": [],
            "actions": [],
            "frames": [],
            "success": False,
        }

    def record_step(
        self,
        ee_pose: np.ndarray,
        object_pose: np.ndarray,
        action: np.ndarray,
        frame: Optional[np.ndarray] = None,
    ):
        assert self._current is not None, "Call begin_episode first"
        self._current["ee_poses"].append(ee_pose.copy() if isinstance(ee_pose, np.ndarray) else np.array(ee_pose))
        self._current["object_poses"].append(object_pose.copy() if isinstance(object_pose, np.ndarray) else np.array(object_pose))
        self._current["actions"].append(action.copy() if isinstance(action, np.ndarray) else np.array(action))
        if frame is not None and self.record_video:
            self._current["frames"].append(frame)

    def end_episode(self, success: bool):
        assert self._current is not None, "Call begin_episode first"
        self._current["success"] = success
        if self.record_video:
            self._write_episode_video(len(self._episodes), self._current)
            self._current["frames"] = []  # free memory
        self._episodes.append(self._current)
        self._current = None

    def finalize(self) -> dict:
        results = self._write_results()
        if self.record_plots:
            self._write_heatmap()
        return results

    def _write_results(self) -> dict:
        n_total = len(self._episodes)
        assert n_total > 0, "finalize() called with no episodes recorded"
        n_success = sum(e["success"] for e in self._episodes)
        success_rate = n_success / n_total

        records = []
        for i, ep in enumerate(self._episodes):
            records.append({
                "episode": i,
                "success": ep["success"],
                "spawn_name": ep["spawn_name"],
                "spawn_pose": ep["spawn_pose"],
            })

        summary = {
            "n_episodes": n_total,
            "n_success": n_success,
            "success_rate": success_rate,
            "episodes": records,
        }

        out_path = os.path.join(self.output_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[EvalLogger] {n_success}/{n_total} success ({100*success_rate:.1f}%) -> {out_path}")
        return summary

    def _write_trajectory_plot(self):
        fig, ax = plt.subplots(figsize=(8, 4))

        for ep in self._episodes:
            obj_poses = np.stack(ep["object_poses"], axis=0)  # (T, 3)
            color = "green" if ep["success"] else "red"
            ax.plot(obj_poses[:, 2], color=color, alpha=0.5, linewidth=0.8)

        ax.set_title("Object height over episode")
        ax.set_xlabel("step")
        ax.set_ylabel("z (m)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "trajectories.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"[EvalLogger] trajectory plot -> {out_path}")

    def _write_heatmap(self):
        named_episodes = [e for e in self._episodes if e["spawn_name"] is not None]
        if not named_episodes:
            return

        # Group by spawn name, collect unique x/y positions.
        spawn_results: Dict[str, List[bool]] = {}
        spawn_xy: Dict[str, Tuple[float, float]] = {}
        for ep in named_episodes:
            name = ep["spawn_name"]
            if name not in spawn_results:
                spawn_results[name] = []
                pose = ep["spawn_pose"] or {}
                spawn_xy[name] = (pose.get("x", 0.0), pose.get("y", 0.0))
            spawn_results[name].append(ep["success"])

        # Build sorted unique axes so grid is aligned.
        xs = sorted(set(v[0] for v in spawn_xy.values()))
        ys = sorted(set(v[1] for v in spawn_xy.values()))
        grid = np.full((len(xs), len(ys)), np.nan)
        labels = [["" for _ in ys] for _ in xs]

        for name, successes in spawn_results.items():
            x, y = spawn_xy[name]
            r = np.mean(successes)
            i, j = xs.index(x), ys.index(y)
            grid[i, j] = r
            labels[i][j] = f"{r:.0%}\n({sum(successes)}/{len(successes)})"

        fig, ax = plt.subplots(figsize=(max(4, len(ys) * 1.5), max(3, len(xs) * 1.2)))
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

        for i in range(len(xs)):
            for j in range(len(ys)):
                if labels[i][j]:
                    ax.text(j, i, labels[i][j], ha="center", va="center",
                            fontsize=10, fontweight="bold")

        ax.set_xticks(range(len(ys)))
        ax.set_xticklabels([f"y={v:.3f}" for v in ys])
        ax.set_yticks(range(len(xs)))
        ax.set_yticklabels([f"x={v:.3f}" for v in xs])
        ax.set_title("Success rate by spawn position")
        plt.colorbar(im, ax=ax, label="success rate")

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "heatmap.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[EvalLogger] heatmap -> {out_path}")

    def _write_episode_video(self, episode_idx: int, ep: dict):
        import imageio
        frames = ep.get("frames", [])
        if not frames:
            return
        tag = "success" if ep["success"] else "fail"
        out_path = os.path.join(self.output_dir, "videos", f"episode_{episode_idx:03d}_{tag}.mp4")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"[EvalLogger] video -> {out_path}")