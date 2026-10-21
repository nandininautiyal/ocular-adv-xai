"""
scripts/10_check_masks.py

Visual and numeric check of src/xai/masks.py on a sample of test images, stratified by lighting.
Writes an overlay grid (pupil red, iris green, sclera blue, periocular uncoloured) and
prints QC statistics per lighting condition.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.xai.masks import get_concept_masks, masks_to_grid  # noqa: E402


def overlay(img, m):
    out = img.copy()
    for name, col in (("sclera", (0.0, 0.4, 1.0)), ("iris", (0.0, 1.0, 0.0)), ("pupil", (1.0, 0.0, 0.0))):
        mk = m[name]
        out[mk] = 0.55 * out[mk] + 0.45 * np.array(col, np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/kaggle/working/manifest_split.csv")
    ap.add_argument("--n-per-lighting", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/kaggle/working/mask_check")
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    m = m[m.split == "test"]
    df = pd.concat([g.sample(n=min(args.n_per_lighting, len(g)), random_state=args.seed)
                    for _, g in m.groupby("lighting")]).reset_index(drop=True)

    rows, shown = [], []
    for r in df.itertuples():
        raw = np.asarray(Image.open(r.link_path).convert("RGB"), dtype=np.uint8)
        masks = get_concept_masks(raw)
        base = np.asarray(Image.fromarray(raw).resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
        shown.append((overlay(base, masks), f"{r.lighting} | {r.device}"))

        names = ("pupil", "iris", "sclera", "periocular")
        stack = np.stack([masks[k] for k in names]).astype(int)
        grid = masks_to_grid(masks)
        area = {k: float(masks[k].mean()) for k in names}
        ip = area["pupil"] + area["iris"]
        rows.append({"lighting": r.lighting, "device": r.device,
                     "partition_ok": bool((stack.sum(0) == 1).all()),
                     "grid_sums_to_1": bool(np.allclose(grid.sum(0), 1.0, atol=1e-5)),
                     "pupil_found": area["pupil"] > 0, "iris_found": ip > 0,
                     "pupil_share_of_iris_pupil": area["pupil"] / ip if ip > 0 else np.nan,
                     **{f"area_{k}": v for k, v in area.items()}})

    qc = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("images checked:", len(qc))
    print("partition ok (masks mutually exclusive and exhaustive):", bool(qc.partition_ok.all()))
    print("masks_to_grid channels sum to 1 in every cell:", bool(qc.grid_sums_to_1.all()))
    print("\nper lighting (means; pupil_share = pupil / (pupil + iris), a real pupil is roughly 0.1-0.35):")
    print(qc.groupby("lighting")[["pupil_found", "iris_found", "pupil_share_of_iris_pupil",
                                  "area_pupil", "area_iris", "area_sclera", "area_periocular"]].mean().round(3).to_string())

    os.makedirs(args.out_dir, exist_ok=True)
    qc.to_csv(os.path.join(args.out_dir, "mask_qc.csv"), index=False)
    cols = 6
    nrows = (len(shown) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(2.4 * cols, 2.5 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (im, title) in zip(axes.ravel(), shown):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    path = os.path.join(args.out_dir, "mask_overlays.png")
    plt.savefig(path, dpi=100)
    print("saved", path)


if __name__ == "__main__":
    main()