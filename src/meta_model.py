from __future__ import annotations

from pathlib import Path

import numpy as np

_ADJ = None
_ADJ_FEATURES = None
_ADJ_RUNTIME_CONFIG = None
_DQ = None
_DQ_FEATURES = None
_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNTIME_CONFIG = {
    "dq_deny_threshold": 0.4,
    "dq_approve_ceiling": 0.4,
    "approve_threshold": 0.58,
    "observed_approve_threshold": 0.7,
    "deny_threshold": 0.5,
    "review_threshold": 0.6,
}


def load_adjudication_model():
    global _ADJ, _ADJ_FEATURES, _ADJ_RUNTIME_CONFIG
    if _ADJ is not None:
        return _ADJ, _ADJ_FEATURES
    path = _ROOT / "models" / "adjudication.joblib"
    if not path.exists():
        return None, None
    import joblib

    blob = joblib.load(path)
    _ADJ = blob["model"]
    _ADJ_FEATURES = blob["features"]
    supplied = blob.get("runtime_config") or {}
    _ADJ_RUNTIME_CONFIG = {
        key: float(supplied.get(key, value))
        for key, value in DEFAULT_RUNTIME_CONFIG.items()
    }
    return _ADJ, _ADJ_FEATURES


def adjudication_runtime_config() -> dict[str, float]:
    load_adjudication_model()
    if _ADJ_RUNTIME_CONFIG is None:
        return dict(DEFAULT_RUNTIME_CONFIG)
    return dict(_ADJ_RUNTIME_CONFIG)


def load_dq_model():
    global _DQ, _DQ_FEATURES
    if _DQ is not None:
        return _DQ, _DQ_FEATURES
    path = _ROOT / "models" / "has_dq.joblib"
    if not path.exists():
        return None, None
    import joblib

    blob = joblib.load(path)
    _DQ = blob["model"]
    _DQ_FEATURES = blob["features"]
    return _DQ, _DQ_FEATURES


def predict_adjudication(feat: dict) -> tuple[str, float] | None:
    model, features = load_adjudication_model()
    if model is None:
        return None
    x = np.array([[float(feat.get(name, 0)) for name in features]], dtype=float)
    label = str(model.predict(x)[0])
    proba = model.predict_proba(x)[0]
    classes = list(model.named_steps["gb"].classes_)
    conf = float(proba[classes.index(label)])
    return label, conf


def predict_has_dq(feat: dict) -> float:
    model, features = load_dq_model()
    if model is None:
        return 0.0
    x = np.array([[float(feat.get(name, 0)) for name in features]], dtype=float)
    return float(model.predict_proba(x)[0, 1])
