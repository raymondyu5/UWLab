from typing import List, Optional
import os
import json
import numpy as np
import matplotlib.pyplot as plt


class EvalLogger:
    """
    Collects per-step and per-episode data during BC policy evaluation,
    then writes results, scatter plots, and optional videos.

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
        self._scatter_points: List[dict] = []  # [{x, y, success, lifted}]

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
                    n_success_ever: int = None,
                    extra_metrics: dict[str, int] | None = None):
        assert self._current is not None, "Call begin_episode first"
        self._current["success"] = success
        self._current["n_success"] = n_success
        self._current["n_total"] = n_total
        self._current["n_success_ever"] = n_success_ever
        self._current["extra_metrics"] = dict(extra_metrics) if extra_metrics else {}
        if self.record_video:
            self._write_episode_video(len(self._episodes), self._current)
            self._current["frames"] = []  # free memory
            self._current["fixed_frames"] = []  # free memory
        self._episodes.append(self._current)
        self._current = None

    def record_scatter_points(self, xs, ys, successes, secondary=None, secondary_name: str = "secondary"):
        """Record per-env (x, y, success, optional secondary metric) for scatter visualization."""
        for i, (x, y, s) in enumerate(zip(xs, ys, successes)):
            pt = {"x": float(x), "y": float(y), "success": bool(s)}
            if secondary is not None:
                pt[secondary_name] = bool(secondary[i])
            self._scatter_points.append(pt)

    def finalize(self) -> dict:
        results = self._write_results()
        if self.record_plots:
            if self._scatter_points:
                self._write_scatter_plots()
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

        # Collect all extra_metric keys across episodes
        all_extra_keys: list[str] = []
        seen = set()
        for e in self._episodes:
            for k in e.get("extra_metrics", {}).keys():
                if k not in seen:
                    all_extra_keys.append(k)
                    seen.add(k)

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
            for k in all_extra_keys:
                v = ep.get("extra_metrics", {}).get(k)
                if v is not None:
                    rec[k] = v
            records.append(rec)

        def _sum_extra(key):
            eps = [e for e in self._episodes if e.get("extra_metrics", {}).get(key) is not None]
            n_sum = sum(e["extra_metrics"][key] for e in eps)
            n_trials = sum(e["n_total"] for e in self._episodes if e.get("n_total") is not None)
            rate = n_sum / n_trials if n_trials > 0 else 0.0
            return n_sum, rate

        n_success_ever_sum = sum(
            e.get("extra_metrics", {}).get("n_success_ever", 0) for e in self._episodes
        )
        n_success_ever_rate = n_success_ever_sum / n_total if n_total > 0 else 0.0
        # n_success_ever stored directly on episode for backward compat
        n_success_ever_direct = sum(
            e["n_success_ever"] for e in self._episodes if e.get("n_success_ever") is not None
        )
        n_success_ever_total = n_success_ever_direct or n_success_ever_sum
        success_rate_ever = n_success_ever_total / n_total if n_total > 0 else 0.0

        extra_rates: dict[str, dict] = {}
        for k in all_extra_keys:
            n, rate = _sum_extra(k)
            extra_rates[k] = {"n": n, "rate": rate}

        summary = {
            "n_episodes": n_episodes,
            "n_success": n_success,
            "n_total": n_total,
            "success_rate": success_rate,
            "n_success_ever": n_success_ever_total,
            "success_rate_ever": success_rate_ever,
            "extra_metric_rates": {k: v["rate"] for k, v in extra_rates.items()},
            "episodes": records,
        }

        out_path = os.path.join(self.output_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        msg = f"[EvalLogger] {n_success}/{n_total} success_end ({100*success_rate:.1f}%)"
        msg += f", {n_success_ever_total}/{n_total} success_ever ({100*success_rate_ever:.1f}%)"
        for k, v in extra_rates.items():
            msg += f", {v['n']}/{n_total} {k} ({100*v['rate']:.1f}%)"
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

    def _write_scatter_plot(
        self,
        metric: np.ndarray,
        positive_label: str,
        title_metric_name: str,
        filename: str,
    ):
        points = self._scatter_points
        xs = np.array([p["x"] for p in points])
        ys = np.array([p["y"] for p in points])
        metric = np.asarray(metric, dtype=bool)

        fig, ax = plt.subplots(figsize=(7, 6))
        for mask, color, label in [
            (~metric, "red", "fail"),
            ( metric, "green", positive_label),
        ]:
            if mask.any():
                ax.scatter(xs[mask], ys[mask], c=color, marker="x", s=40,
                           linewidths=1.2, label=label, alpha=0.7)

        n_pos = int(metric.sum())
        n_tot = len(metric)
        ax.set_title(f"Spawn outcomes  {n_pos}/{n_tot} {title_metric_name} ({100*n_pos/n_tot:.1f}%)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[EvalLogger] scatter plot -> {out_path}")

    def _write_scatter_plots(self):
        points = self._scatter_points
        successes = np.array([p["success"] for p in points])
        self._write_scatter_plot(
            metric=successes,
            positive_label="success",
            title_metric_name="success",
            filename="scatter_success.png",
        )
        # Write a scatter for every secondary metric recorded in scatter points.
        extra_keys = [k for k in points[0].keys() if k not in ("x", "y", "success")] if points else []
        for key in extra_keys:
            vals = np.array([p.get(key, False) for p in points])
            self._write_scatter_plot(
                metric=vals,
                positive_label=key,
                title_metric_name=key,
                filename=f"scatter_{key}.png",
            )

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