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
        logger = EvalLogger(output_dir, record_video=False, record_plots=True, video_fps=10.0)
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
        video_fps: float = 10.0,
    ):
        self.output_dir = output_dir
        self.record_video = record_video
        self.record_plots = record_plots
        self.video_fps = video_fps

        os.makedirs(output_dir, exist_ok=True)
        if record_video:
            os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)

        self._episodes: List[dict] = []
        self._current: Optional[dict] = None
        self._scatter_points: List[dict] = []  # [{x, y, success}]

    def begin_episode(self, spawn_name: Optional[str] = None, spawn_pose: Optional[dict] = None):
        self._current = {
            "spawn_name": spawn_name,
            "spawn_pose": spawn_pose,
            "ee_poses": [],
            "object_poses": [],
            "actions": [],
            "frames": [],
            "fixed_frames": [],
            "success": False,
        }

    def record_step(
        self,
        ee_pose: np.ndarray,
        object_pose: np.ndarray,
        action: np.ndarray,
        frame: Optional[np.ndarray] = None,
        fixed_frame: Optional[np.ndarray] = None,
    ):
        assert self._current is not None, "Call begin_episode first"
        self._current["ee_poses"].append(ee_pose.copy() if isinstance(ee_pose, np.ndarray) else np.array(ee_pose))
        self._current["object_poses"].append(object_pose.copy() if isinstance(object_pose, np.ndarray) else np.array(object_pose))
        self._current["actions"].append(action.copy() if isinstance(action, np.ndarray) else np.array(action))
        if frame is not None and self.record_video:
            self._current["frames"].append(frame)
        if fixed_frame is not None and self.record_video:
            self._current["fixed_frames"].append(fixed_frame)

    def end_episode(self, success: bool | float, n_success: int = None, n_total: int = None,
                    n_grasped: int = None, n_lifted: int = None, n_near_miss: int = None):
        assert self._current is not None, "Call begin_episode first"
        self._current["success"] = success
        self._current["n_success"] = n_success
        self._current["n_total"] = n_total
        self._current["n_grasped"] = n_grasped
        self._current["n_lifted"] = n_lifted
        self._current["n_near_miss"] = n_near_miss
        if self.record_video:
            self._write_episode_video(len(self._episodes), self._current)
            self._current["frames"] = []  # free memory
            self._current["fixed_frames"] = []  # free memory
        self._episodes.append(self._current)
        self._current = None

    def record_scatter_points(self, xs, ys, successes):
        """Record per-env (x, y, success) results for scatter plot visualization."""
        for x, y, s in zip(xs, ys, successes):
            self._scatter_points.append({"x": float(x), "y": float(y), "success": bool(s)})

    def finalize(self) -> dict:
        results = self._write_results()
        if self.record_plots:
            if self._scatter_points:
                self._write_scatter_plot()
            else:
                self._write_heatmap()
            for key, title, fname in [
                ("n_grasped", "Grasped rate by spawn position", "heatmap_grasped.png"),
                ("n_lifted", "Lifted rate by spawn position", "heatmap_lifted.png"),
                ("n_near_miss", "Near miss rate by spawn position", "heatmap_near_miss.png"),
            ]:
                if any(e.get(key) is not None for e in self._episodes):
                    self._write_metric_heatmap(key, title, fname)
        return results

    def _write_results(self) -> dict:
        n_episodes = len(self._episodes)
        assert n_episodes > 0, "finalize() called with no episodes recorded"
        n_success = sum(
            (e["n_success"] if e.get("n_success") is not None else (1 if e["success"] else 0))
            for e in self._episodes
        )
        n_total = sum(
            (e["n_total"] if e.get("n_total") is not None else 1) for e in self._episodes
        )
        success_rate = n_success / n_total if n_total > 0 else 0.0

        records = []
        for i, ep in enumerate(self._episodes):
            rec = {
                "episode": i,
                "success": ep["success"],
                "spawn_name": ep["spawn_name"],
                "spawn_pose": ep["spawn_pose"],
            }
            if ep.get("n_success") is not None and ep.get("n_total") is not None:
                rec["n_success"] = ep["n_success"]
                rec["n_total"] = ep["n_total"]
            for key in ("n_grasped", "n_lifted", "n_near_miss"):
                if ep.get(key) is not None:
                    rec[key] = ep[key]
            records.append(rec)

        def _sum_metric(key):
            eps = [e for e in self._episodes if e.get(key) is not None]
            n_sum = sum(e[key] for e in eps)
            n_trials = sum(e["n_total"] for e in eps if e.get("n_total") is not None)
            rate = n_sum / n_trials if n_trials > 0 else 0.0
            return n_sum, n_trials, rate

        n_grasped, _, grasped_rate = _sum_metric("n_grasped")
        n_lifted, _, lifted_rate = _sum_metric("n_lifted")
        n_near_miss, n_near_miss_trials, near_miss_rate = _sum_metric("n_near_miss")

        summary = {
            "n_episodes": n_episodes,
            "n_success": n_success,
            "n_total": n_total,
            "success_rate": success_rate,
            "n_near_miss": n_near_miss,
            "near_miss_rate": near_miss_rate,
            "n_lifted": n_lifted,
            "lifted_rate": lifted_rate,
            "n_grasped": n_grasped,
            "grasped_rate": grasped_rate,
            "episodes": records,
        }

        out_path = os.path.join(self.output_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        msg = f"[EvalLogger] {n_success}/{n_total} success ({100*success_rate:.1f}%)"
        msg += f", {n_near_miss}/{n_total} near_miss ({100*near_miss_rate:.1f}%)"
        msg += f", {n_lifted}/{n_total} lifted ({100*lifted_rate:.1f}%)"
        msg += f", {n_grasped}/{n_total} grasped ({100*grasped_rate:.1f}%)"
        print(f"{msg} -> {out_path}")
        return summary

    def _write_trajectory_plot(self):
        fig, ax = plt.subplots(figsize=(8, 4))

        for ep in self._episodes:
            obj_poses = np.stack(ep["object_poses"], axis=0)  # (T, 3)
            success_val = ep["success"]
            color = "green" if (success_val if isinstance(success_val, bool) else success_val >= 0.5) else "red"
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

        # Group by spawn name, collect n_success/n_total or legacy bool successes.
        spawn_results: Dict[str, List[Tuple[int, int] | bool]] = {}
        spawn_xy: Dict[str, Tuple[float, float]] = {}
        for ep in named_episodes:
            name = ep["spawn_name"]
            if name not in spawn_results:
                spawn_results[name] = []
                pose = ep["spawn_pose"] or {}
                spawn_xy[name] = (pose.get("x", 0.0), pose.get("y", 0.0))
            if ep.get("n_success") is not None and ep.get("n_total") is not None:
                spawn_results[name].append((ep["n_success"], ep["n_total"]))
            else:
                spawn_results[name].append(1 if ep["success"] else 0)

        # Build sorted unique axes so grid is aligned.
        xs = sorted(set(v[0] for v in spawn_xy.values()))
        ys = sorted(set(v[1] for v in spawn_xy.values()))
        grid = np.full((len(xs), len(ys)), np.nan)
        labels = [["" for _ in ys] for _ in xs]

        for name, results in spawn_results.items():
            x, y = spawn_xy[name]
            i, j = xs.index(x), ys.index(y)
            if results and isinstance(results[0], tuple):
                n_succ = sum(r[0] for r in results)
                n_tot = sum(r[1] for r in results)
                r = n_succ / n_tot if n_tot > 0 else 0.0
                labels[i][j] = f"{r:.0%}\n({n_succ}/{n_tot})"
            else:
                successes = [int(r) for r in results]
                r = np.mean(successes)
                labels[i][j] = f"{r:.0%}\n({sum(successes)}/{len(successes)})"
            grid[i, j] = r

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

    def _write_metric_heatmap(self, n_key: str, title: str, filename: str):
        """Grid plot of a count-based metric rate (n_key / n_total) by spawn position."""
        named_episodes = [
            e for e in self._episodes
            if e["spawn_name"] is not None and e.get(n_key) is not None and e.get("n_total") is not None
        ]
        if not named_episodes:
            return

        spawn_results: Dict[str, List[Tuple[int, int]]] = {}
        spawn_xy: Dict[str, Tuple[float, float]] = {}
        for ep in named_episodes:
            name = ep["spawn_name"]
            if name not in spawn_results:
                spawn_results[name] = []
                pose = ep["spawn_pose"] or {}
                spawn_xy[name] = (pose.get("x", 0.0), pose.get("y", 0.0))
            spawn_results[name].append((ep[n_key], ep["n_total"]))

        xs = sorted(set(v[0] for v in spawn_xy.values()))
        ys = sorted(set(v[1] for v in spawn_xy.values()))
        grid = np.full((len(xs), len(ys)), np.nan)
        labels = [["" for _ in ys] for _ in xs]

        for name, results in spawn_results.items():
            x, y = spawn_xy[name]
            i, j = xs.index(x), ys.index(y)
            n_met = sum(r[0] for r in results)
            n_tot = sum(r[1] for r in results)
            r = n_met / n_tot if n_tot > 0 else 0.0
            labels[i][j] = f"{r:.0%}\n({n_met}/{n_tot})"
            grid[i, j] = r

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
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[EvalLogger] {filename} -> {out_path}")

    def _write_scatter_plot(self):
        points = self._scatter_points
        xs = np.array([p["x"] for p in points])
        ys = np.array([p["y"] for p in points])
        successes = np.array([p["success"] for p in points])

        fig, ax = plt.subplots(figsize=(7, 6))
        for mask, color, label in [
            (~successes, "red",   "fail"),
            ( successes, "green", "success"),
        ]:
            if mask.any():
                ax.scatter(xs[mask], ys[mask], c=color, marker="x", s=40,
                           linewidths=1.2, label=label, alpha=0.7)

        n_succ = int(successes.sum())
        n_tot = len(successes)
        ax.set_title(f"Spawn outcomes  {n_succ}/{n_tot} success ({100*n_succ/n_tot:.1f}%)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "scatter.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[EvalLogger] scatter plot -> {out_path}")

    def _write_episode_video(self, episode_idx: int, ep: dict):
        import imageio
        frames = ep.get("frames", [])
        success_val = ep["success"]
        tag = "success" if (success_val if isinstance(success_val, bool) else success_val >= 0.5) else "fail"
        if frames:
            out_path = os.path.join(self.output_dir, "videos", f"episode_{episode_idx:03d}_{tag}.mp4")
            imageio.mimsave(out_path, frames, fps=self.video_fps)
            print(f"[EvalLogger] video -> {out_path}")

        fixed_frames = ep.get("fixed_frames", [])
        if fixed_frames:
            fixed_out_path = os.path.join(self.output_dir, "videos", f"episode_{episode_idx:03d}_{tag}_fixed.mp4")
            imageio.mimsave(fixed_out_path, fixed_frames, fps=self.video_fps)
            print(f"[EvalLogger] fixed video -> {fixed_out_path}")