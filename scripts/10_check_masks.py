'''
scripts/10_check_masks.py

Visual and numeric check of src/xai/masks.py on a sample of test images, stratified by lighting.

Images are prepared exactly like the experiment runner prepares them (stretched to 256 x 256
with nearest-neighbour resampling, float32 in [0, 1]), so the masks checked here are the masks
the runner will use.

Writes to --out-dir:
  mask_overlays.png   overlay grid (pupil red, iris green, sclera blue, periocular uncoloured)
  mask_qc.csv         one row per image, with the circle that was used and whether the
                      geometric fallback was needed
  mask_sample.csv     relpath of every sampled image (use the same images for hand labelling)

Prints QC statistics per lighting condition. Pupil share is capped by construction
(masks.PUPIL_MAX_SHARE), so a plausible share does NOT show that the pupil is correct;
only IoU against hand-labelled masks can do that.
'''
import argparse
import inspect
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)
from src.xai.masks import get_concept_masks, masks_to_grid  # noqa: E402

HAS_INFO = 'return_info' in inspect.signature(get_concept_masks).parameters


def load_like_runner(path):
    """(256, 256, 3) float32 in [0, 1], loaded like flow_from_directory does (nearest)."""
    img = Image.open(path).convert('RGB').resize((256, 256), Image.NEAREST)
    return np.asarray(img, dtype=np.float32) / 255.0


def overlay(img, m):
    out = img.copy()
    for name, col in (('sclera', (0.0, 0.4, 1.0)), ('iris', (0.0, 1.0, 0.0)), ('pupil', (1.0, 0.0, 0.0))):
        mk = m[name]
        out[mk] = 0.55 * out[mk] + 0.45 * np.array(col, np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/kaggle/working/manifest_split.csv')
    ap.add_argument('--n-per-lighting', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default='/kaggle/working/mask_check')
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    m = m[m.split == 'test']
    df = pd.concat([g.sample(n=min(args.n_per_lighting, len(g)), random_state=args.seed)
                    for _, g in m.groupby('lighting')]).reset_index(drop=True)

    rows, shown = [], []
    for r in df.itertuples():
        img = load_like_runner(r.link_path)
        if HAS_INFO:
            masks, info = get_concept_masks(img, return_info=True)
        else:
            masks, info = get_concept_masks(img), {}
        shown.append((overlay(img, masks), f'{r.lighting} | {r.device}'
                      + (' | FALLBACK' if info.get('fallback_used') else '')))

        names = ('pupil', 'iris', 'sclera', 'periocular')
        stack = np.stack([masks[k] for k in names]).astype(int)
        grid = masks_to_grid(masks)
        area = {k: float(masks[k].mean()) for k in names}
        ip = area['pupil'] + area['iris']
        rows.append({'relpath': r.relpath, 'lighting': r.lighting, 'device': r.device,
                     'partition_ok': bool((stack.sum(0) == 1).all()),
                     'grid_sums_to_1': bool(np.allclose(grid.sum(0), 1.0, atol=1e-5)),
                     'pupil_found': area['pupil'] > 0, 'iris_found': ip > 0,
                     'pupil_share_of_iris_pupil': area['pupil'] / ip if ip > 0 else np.nan,
                     **{f'area_{k}': v for k, v in area.items()},
                     **{k: info[k] for k in ('cx', 'cy', 'radius', 'fallback_used', 'n_candidates',
                                              'circle_score', 'centre_in_opening') if k in info}})

    qc = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print('images checked:', len(qc))
    print('partition ok (masks mutually exclusive and exhaustive):', bool(qc.partition_ok.all()))
    print('masks_to_grid channels sum to 1 in every cell:', bool(qc.grid_sums_to_1.all()))
    print('\nper lighting (means; pupil_share = pupil / (pupil + iris); it is capped by construction):')
    print(qc.groupby('lighting')[['pupil_found', 'iris_found', 'pupil_share_of_iris_pupil',
                                  'area_pupil', 'area_iris', 'area_sclera', 'area_periocular']].mean().round(3).to_string())

    if 'fallback_used' in qc:
        print('\niris circle (Hough candidate vs geometric fallback):')
        print(qc.groupby('lighting').agg(
            fallback_rate=('fallback_used', 'mean'),
            centre_in_opening=('centre_in_opening', 'mean'),
            radius_mean=('radius', 'mean'), radius_std=('radius', 'std'),
            radius_min=('radius', 'min'), radius_max=('radius', 'max')).round(2).to_string())
        rate = float(qc.fallback_used.mean())
        print(f'\noverall fallback rate: {rate:.0%}'
              + ('  <- Hough fails often; the iris mask is a fixed circle for those images' if rate > 0.3 else ''))
        print(f"distinct radii: {qc.radius.nunique()} of {len(qc)} images "
              f"(many repeats of one value suggest the fallback radius is doing the work)")
    else:
        print('\n(this masks.py has no return_info: fallback and circle statistics are not available)')

    os.makedirs(args.out_dir, exist_ok=True)
    qc.to_csv(os.path.join(args.out_dir, 'mask_qc.csv'), index=False)
    df[['relpath', 'lighting', 'device', 'eye', 'phase', 'condition']].to_csv(
        os.path.join(args.out_dir, 'mask_sample.csv'), index=False)
    cols = 6
    nrows = (len(shown) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(2.4 * cols, 2.5 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (im, title) in zip(axes.ravel(), shown):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'mask_overlays.png')
    plt.savefig(path, dpi=100)
    print('saved', path)


if __name__ == '__main__':
    main()