"""
scripts/11_run_detection.py

Reference-free, training-free detection of adversarial examples by explanation
consistency (src/detect/consistency.py), against prediction-only baselines
(Feature Squeezing; logit cosine) and a supervised detector evaluated on
attacks it has not seen.

Run it on Kaggle next to the checkpoint, after 04_eval_clean.py has written the
correct-image list. Per class, disjoint images are drawn for

  calibration   clean only: sets thresholds and robust-z fusion (no attacks)
  evaluation    clean, plus every attack x epsilon applied to it

Only successful adversarial examples count as positives.

Writes to --out-dir:
  features_<tag>.csv     one row per image variant, every score
  detection_<tag>.csv    AUROC and TPR at 5 % FPR per attack@epsilon x score
  summary_<tag>.json     headline AUROCs, attack success, leave-one-attack-out,
                         attack clustering (ARI)

Usage (Kaggle):
  python scripts/11_run_detection.py \
      --ckpt /kaggle/input/<checkpoint-dataset>/Xception_ArcFace_full.keras \
      --correct-csv /kaggle/working/eval_clean/correct_test_full.csv --tag full
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; set before importing tensorflow
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.attacks.blackbox import monocle_blending, patch_occlusion  # noqa: E402
from src.attacks.whitebox import ATTACK_REGISTRY  # noqa: E402
from src.detect.consistency import RobustZFusion, detection_features, score_columns  # noqa: E402
from src.detect.evaluate import attack_clustering, auroc, leave_one_attack_out, tpr_at_fpr  # noqa: E402
from src.models.arcface_inference import load_checkpoint  # noqa: E402

BLACKBOX = ("patch", "monocle")
NO_BUDGET = ("cw",)
FPR = 0.05


def load_image(path):
    """(256, 256, 3) float32 in [0, 1], loaded like flow_from_directory does (nearest)."""
    img = tf.keras.utils.load_img(path, target_size=(256, 256))
    return np.asarray(img, dtype=np.float32) / 255.0


def resolve_path(row, data_root):
    if isinstance(row.link_path, str) and os.path.exists(row.link_path):
        return row.link_path
    if data_root and isinstance(getattr(row, "relpath", None), str):
        path = os.path.join(data_root, row.relpath)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{row.link_path} not found; rebuild the split tree (01, 02) "
                            f"or pass --data-root to resolve the manifest relpath")


def split_per_class(df, calib_per_class, per_class, seed):
    """Disjoint calibration and evaluation images, drawn per class."""
    shuffled = df.sample(frac=1.0, random_state=seed)
    rank = shuffled.groupby("true_idx").cumcount()
    calib = shuffled[rank < calib_per_class]
    evaluation = shuffled[(rank >= calib_per_class) & (rank < calib_per_class + per_class)]
    return calib.reset_index(drop=True), evaluation.reset_index(drop=True)


def run_attack(name, eps, clean, labels, sides, model, seed):
    if name == "patch":
        return patch_occlusion(clean, patch_size=64, position=(96, 96))
    if name == "monocle":
        return np.stack([monocle_blending(img, side=side) for img, side in zip(clean, sides)])
    kwargs = {"label": labels}
    if name == "pgd":
        kwargs["seed"] = seed
    if name not in NO_BUDGET:
        kwargs["epsilon"] = eps
    return np.asarray(ATTACK_REGISTRY[name](clean, model, **kwargs))


def batches(df, size, data_root):
    for start in range(0, len(df), size):
        batch = df.iloc[start:start + size]
        images = np.stack([load_image(resolve_path(r, data_root)) for r in batch.itertuples()])
        sides = [e if e in ("left", "right") else None for e in batch.get("eye", [None] * len(batch))]
        yield start, batch, images, batch.true_idx.to_numpy(), sides


def evaluate(rows, names, calib_mask, clean_mask, adv):
    """AUROC / TPR@FPR per attack@epsilon (and pooled) for every score in *names*."""
    groups = [("all", adv)] + [(key, g) for key, g in adv.groupby("variant")]
    out = []
    for group, g in groups:
        for name in names:
            op = tpr_at_fpr(rows.loc[calib_mask, name], rows.loc[clean_mask, name], g[name], FPR)
            out.append({"group": group, "score": name, "n_adv": len(g),
                        "auroc": auroc(rows.loc[clean_mask, name], g[name]),
                        "tpr_at_5fpr": op["tpr"], "fpr_test": op["fpr"]})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--correct-csv", required=True, help="correct_<split>_<tag>.csv from 04_eval_clean.py")
    ap.add_argument("--data-root", default=None, help="VISOB root, used if link_path no longer exists")
    ap.add_argument("--per-class", type=int, default=1, help="evaluation images per class")
    ap.add_argument("--calib-per-class", type=int, default=1, help="clean calibration images per class")
    ap.add_argument("--max-images", type=int, default=None, help="debug: cap evaluation and calibration sizes")
    ap.add_argument("--attacks", default="fgsm,bim,pgd,mim,cw,patch,monocle")
    ap.add_argument("--epsilons", default="0.001,0.0019608,0.0039216,0.0078431",
                    help="L-inf budgets: the paper's 0.001, then 0.5/255, 1/255, 2/255")
    ap.add_argument("--transforms", default="jpeg,bit_depth")
    ap.add_argument("--xai", default="layercam,eigencam")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/kaggle/working/detection")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    epsilons = [float(e) for e in args.epsilons.split(",")]
    transforms = [t.strip() for t in args.transforms.split(",") if t.strip()]
    xai_names = [x.strip() for x in args.xai.split(",") if x.strip()]
    for name in attacks:
        assert name in ATTACK_REGISTRY or name in BLACKBOX, f"unknown attack {name}"

    model = load_checkpoint(args.ckpt)
    calib, evaluation = split_per_class(pd.read_csv(args.correct_csv), args.calib_per_class,
                                        args.per_class, args.seed)
    if args.max_images:
        calib, evaluation = calib.head(args.max_images), evaluation.head(args.max_images)
    print(f"calibration: {len(calib)} clean | evaluation: {len(evaluation)} images | attacks: {attacks} "
          f"| epsilons: {epsilons} | transforms: {transforms} | xai: {xai_names}")

    t0 = time.time()
    rows = []

    def record(batch, kind, feats, extra):
        for i, r in enumerate(batch.itertuples()):
            row = {"link_path": r.link_path, "true_idx": int(r.true_idx), "kind": kind, **extra(i)}
            row.update({k: (int(v[i]) if k == "pred" else float(v[i])) for k, v in feats.items()})
            rows.append(row)

    for _, batch, images, labels, _ in batches(calib, args.batch_size, args.data_root):
        feats = detection_features(model, images, xai_names, transforms)
        record(batch, "calibration", feats, lambda i: {})

    for start, batch, clean, labels, sides in batches(evaluation, args.batch_size, args.data_root):
        feats = detection_features(model, clean, xai_names, transforms)
        record(batch, "clean", feats, lambda i: {})
        for name in attacks:
            budgets = [None] if name in BLACKBOX + NO_BUDGET else epsilons
            for eps in budgets:
                adv = run_attack(name, eps, clean, labels, sides, model, args.seed + start)
                feats = detection_features(model, adv, xai_names, transforms)
                success = feats["pred"] != labels
                linf = np.abs(adv - clean).reshape(len(adv), -1).max(axis=1)
                variant = name if eps is None else f"{name}@{eps:.6g}"
                record(batch, "adversarial", feats,
                       lambda i: {"attack": name, "epsilon": eps, "variant": variant,
                                  "success": bool(success[i]), "linf": float(linf[i])})
        print(f"  {start + len(batch)}/{len(evaluation)} images | {(time.time() - t0) / 60:.1f} min", flush=True)

    rows = pd.DataFrame(rows)
    families = score_columns(rows.columns)
    singles = [c for cols in families.values() for c in cols]
    calib_mask, clean_mask = rows.kind == "calibration", rows.kind == "clean"

    fusion = RobustZFusion({c: rows.loc[calib_mask, c].to_numpy() for c in singles})
    scores = {c: rows[c].to_numpy() for c in singles}
    fused = {f"fused_{family}": fusion(scores, cols) for family, cols in families.items() if cols}
    fused["fused_all"] = fusion(scores, singles)
    if families["prediction"] and families["explanation"]:
        fused["fused_prediction_explanation"] = fusion(scores, families["prediction"] + families["explanation"])
    for name, values in fused.items():
        rows[name] = values

    adv_all = rows[rows.kind == "adversarial"]
    adv = adv_all[adv_all.success]
    table = evaluate(rows, singles + list(fused), calib_mask, clean_mask, adv)

    loao = leave_one_attack_out(rows.loc[calib_mask, singles].to_numpy(),
                                rows.loc[clean_mask, singles].to_numpy(),
                                adv[singles].to_numpy(), adv.attack.to_numpy())
    training_free = {a: auroc(rows.loc[clean_mask, "fused_all"], g["fused_all"])
                     for a, g in adv.groupby("attack")}
    pooled = table[table.group == "all"].set_index("score")

    summary = {
        "tag": args.tag, "ckpt": args.ckpt, "n_calibration": int(calib_mask.sum()),
        "n_clean_eval": int(clean_mask.sum()), "n_adversarial_successful": int(len(adv)),
        "attacks": attacks, "epsilons": epsilons, "transforms": transforms, "xai": xai_names,
        "attack_success_rate": {k: float(g.success.mean()) for k, g in adv_all.groupby("variant")},
        "pooled_auroc": {name: float(pooled.loc[name, "auroc"]) for name in fused},
        "pooled_tpr_at_5fpr": {name: float(pooled.loc[name, "tpr_at_5fpr"]) for name in fused},
        "best_single_score": {
            family: max(cols, key=lambda c: pooled.loc[c, "auroc"]) for family, cols in families.items() if cols},
        "unseen_attack_auroc": {"supervised_logistic": loao, "training_free_fused_all": training_free},
        "attack_clustering_ari": {
            "explanation_scores": attack_clustering(adv[families["explanation"]].to_numpy(), adv.attack),
            "all_scores": attack_clustering(adv[singles].to_numpy(), adv.attack)},
        "minutes": round((time.time() - t0) / 60, 1),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    rows.to_csv(os.path.join(args.out_dir, f"features_{args.tag}.csv"), index=False)
    table.to_csv(os.path.join(args.out_dir, f"detection_{args.tag}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("pooled_auroc", "pooled_tpr_at_5fpr", "minutes")}, indent=2))


if __name__ == "__main__":
    main()
