"""
src/models/recognition_model.py

Thin wrapper around the ABRobOcular reference model
(third_party/ABRobOcular/train.py). No re-implementation: the architecture
comes straight from their create_model(). Keep the vendored folder unmodified.
"""
import os
import sys
import yaml

_THIRD_PARTY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "ABRobOcular")
)
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

from train import ArcFace, create_model   # noqa: E402  (vendored, unmodified)
from dataset_loader import CustomDataGen   # noqa: E402  (vendored, unmodified)

def load_cfg(path="configs/baseline.yml"):
    """Load OUR config (not the vendored config.yml)."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    """Build the paper's Xception + ArcFace model from a config dict."""
    return create_model(cfg)