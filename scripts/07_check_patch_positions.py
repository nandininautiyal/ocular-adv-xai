"""Render the paper's Fig. 7 patch positions on one image to check the (row, col) convention."""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.attacks.blackbox import patch_occlusion  # noqa: E402

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/working/patch_positions.png"
img = np.asarray(Image.open(path).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0

positions = [(64, 96), (96, 64), (128, 96), (64, 64), (96, 96), (32, 96)]
fig, axes = plt.subplots(1, len(positions), figsize=(3 * len(positions), 3.2))
for ax, pos in zip(axes, positions):
    ax.imshow(patch_occlusion(img, 64, pos))
    ax.set_title(str(pos))
    ax.axis("off")
plt.tight_layout()
plt.savefig(out, dpi=100)
print("saved", out)