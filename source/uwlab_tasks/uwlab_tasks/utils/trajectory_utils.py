import os
import numpy as np
from glob import glob

def load_real_episode(episode_path: str):
    """Load real robot episode data."""
    episode_name = os.path.basename(episode_path)

    if episode_path.endswith('.npy'):
        episode_file = episode_path
        episode_id = episode_name.split('_')[1].split('.')[0]
    else:
        episode_id = episode_name.split('_')[1]
        episode_file = os.path.join(episode_path, f"episode_{episode_id}.npy")

    print(f"\nLoading episode: {episode_file}")
    data = np.load(episode_file, allow_pickle=True).item()

    obs_list = data['obs']
    actions = data.get('actions', [])

    pcd_files = sorted(glob(os.path.join(episode_path, "CL*.npy")))
    pointclouds = [np.load(f) for f in pcd_files] if pcd_files else None

    print(f"  Observations: {len(obs_list)}")
    print(f"  Actions: {len(actions)}")
    if pointclouds:
        print(f"  Pointclouds: {len(pointclouds)}")

    if obs_list:
        print(f"\n  Observation keys: {list(obs_list[0].keys())}")

    if actions:
        a0 = np.array(actions[0])
        print(f"  Action shape: {a0.shape}")

    return {
        'obs': obs_list,
        'actions': actions,
        'pointclouds': pointclouds,
        'episode_id': episode_id,
    }

    