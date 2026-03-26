import os
import sys
from pathlib import Path


def _repo_root_with_third_party() -> Path:
    """Resolve UWLab repo root (directory containing third_party/)."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "third_party").is_dir():
            return p
    return Path("/workspace/uwlab")


def setup_third_party_paths():
    """Add third-party packages to sys.path.

    Isaac Sim's python.sh resets PYTHONPATH, so packages in third_party/ must
    be inserted manually. Call this at the top of any entry-point script before
    importing diffusion_policy or pip_packages.
    """
    third_party = _repo_root_with_third_party() / "third_party"
    entries = [
        (str(third_party / "pip_packages"), True),   # dill, imageio, etc.
        (str(third_party / "diffusion_policy"), False),  # EMAModel, dict_apply, etc.
    ]
    for path, insert_front in entries:
        if os.path.isdir(path) and path not in sys.path:
            if insert_front:
                sys.path.insert(0, path)
            else:
                sys.path.append(path)


def setup_maniflow_path():
    """Append ManiFlow package (third_party/ManiFlow_Policy/ManiFlow) to sys.path.

    Call after setup_third_party_paths() if importing maniflow.* (e.g. UWLab
    ManiFlow point-cloud policy). Import verification is expected in Docker.
    """
    root = _repo_root_with_third_party()
    mf = root / "third_party" / "ManiFlow_Policy" / "ManiFlow"
    if mf.is_dir():
        path = str(mf)
        if path not in sys.path:
            sys.path.append(path)
