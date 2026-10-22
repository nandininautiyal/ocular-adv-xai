"""
src/detect/evaluate.py

Detection metrics and the two label-dependent comparisons.

auroc, tpr_at_fpr
    Threshold-free and operating-point metrics. The threshold for
    ``tpr_at_fpr`` comes from clean calibration images only.
leave_one_attack_out
    A supervised detector (logistic regression) trained on clean images plus
    every attack but one, tested on the held-out attack. This is how
    label-dependent detectors such as ABRobOcular's are limited: they must have
    seen the attack.
attack_clustering
    Unsupervised attack attribution: do adversarial examples cluster by attack
    family in score space? Reported as adjusted Rand index (0 = chance).
"""

import numpy as np
from scipy.stats import rankdata


def auroc(negative, positive):
    """Area under the ROC curve (Mann-Whitney U, ties count half); NaN if a side is empty."""
    neg, pos = np.asarray(negative, np.float64), np.asarray(positive, np.float64)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([neg, pos]))
    u = ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(neg) * len(pos)))


def tpr_at_fpr(calibration, negative, positive, fpr=0.05):
    """Detection rate at a threshold set on clean calibration scores.

    Returns
    -------
    dict with ``threshold``, ``tpr`` (on *positive*) and ``fpr`` (measured on
    the held-out clean *negative*, which should be close to the target)
    """
    cal = np.asarray(calibration, np.float64)
    threshold = float(np.quantile(cal, 1.0 - fpr, method="higher"))
    pos, neg = np.asarray(positive, np.float64), np.asarray(negative, np.float64)
    return {"threshold": threshold,
            "tpr": float((pos > threshold).mean()) if len(pos) else float("nan"),
            "fpr": float((neg > threshold).mean()) if len(neg) else float("nan")}


def _logistic(seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))


def leave_one_attack_out(clean_train, clean_test, adversarial, attacks):
    """AUROC of a supervised detector on an attack it was not trained on.

    Parameters
    ----------
    clean_train, clean_test : ndarray (n, d) — features of disjoint clean images
    adversarial : ndarray (m, d) — features of successful adversarial examples
    attacks : ndarray (m,) — attack family of each adversarial row

    Returns
    -------
    dict held-out attack -> AUROC (NaN if no other attack to train on)
    """
    attacks = np.asarray(attacks)
    out = {}
    for held in np.unique(attacks):
        seen = adversarial[attacks != held]
        if len(seen) == 0:
            out[str(held)] = float("nan")
            continue
        clf = _logistic().fit(np.vstack([clean_train, seen]),
                              np.r_[np.zeros(len(clean_train)), np.ones(len(seen))])
        out[str(held)] = auroc(clf.decision_function(clean_test),
                               clf.decision_function(adversarial[attacks == held]))
    return out


def attack_clustering(adversarial, attacks, seed=0):
    """Adjusted Rand index between k-means clusters (k = number of attacks) and attack labels."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    attacks = np.asarray(attacks)
    k = len(np.unique(attacks))
    if k < 2 or len(attacks) <= k:
        return float("nan")
    x = StandardScaler().fit_transform(adversarial)
    clusters = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
    return float(adjusted_rand_score(attacks, clusters))
