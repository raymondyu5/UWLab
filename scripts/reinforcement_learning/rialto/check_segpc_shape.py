import glob, json, os, collections

dirs = [
    '/gscratch/scrubbed/wph52/rialto_rl/credit_card_grasp_rl_privileged'
]

for data_dir in dirs:
    meta_paths = sorted(glob.glob(os.path.join(data_dir, 'episode_*/episode_*.zarr/data/seg_pc/zarr.json')))
    counts = collections.Counter()
    for p in meta_paths:
        shape = tuple(json.load(open(p))['shape'])
        counts[shape] += 1
    print(f'{os.path.basename(data_dir)}: {len(meta_paths)} episodes')
    for shape, n in sorted(counts.items()):
        T, d1, d2 = shape
        if d2 == 3 and d1 != 3:
            fmt = 'CORRECT (T,N,3)'
        elif d1 == 3 and d2 != 3:
            fmt = 'NEEDS FIX (T,3,N)'
        else:
            fmt = 'AMBIGUOUS'
        print(f'  {shape} x{n}  {fmt}')
    print()