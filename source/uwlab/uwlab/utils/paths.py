import os
import sys


def setup_third_party_paths():
    """Add third-party packages to sys.path.

    Isaac Sim's python.sh resets PYTHONPATH, so packages in third_party/ must
    be inserted manually. Call this at the top of any entry-point script before
    importing diffusion_policy or pip_packages.
    """
    third_party = "/workspace/uwlab/third_party"
    entries = [
        (os.path.join(third_party, "pip_packages"), True),   # dill, imageio, etc.
        (os.path.join(third_party, "diffusion_policy"), False),  # EMAModel, dict_apply, etc.
    ]
    for path, insert_front in entries:
        if os.path.isdir(path) and path not in sys.path:
            if insert_front:
                sys.path.insert(0, path)
            else:
                sys.path.append(path)
