#!/usr/bin/env python3
"""
Pick and save a decision threshold from the VALIDATION set (no retrain).

- Recreates the train/val split used in train_model.py (same seed, stratify, val_size)
- Chooses the threshold that maximizes F_beta **subject to BOTH floors**:
    recall >= --min-recall  AND  precision >= --min-precision
- If no threshold satisfies both floors, falls back to global max F_beta and
  prints how close you got.
- Writes:
    - models/decision_threshold.npy  (the threshold as a single float)
    - models/decision_threshold.json (metadata about the choice)
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix

import tensorflow as tf
from tensorflow import keras


# ---------- helpers ----------
def load_model(model_path: Path):
    """Load Keras/AutoKeras model with AK custom objects if needed."""
    # Prefer AutoKeras loader
    try:
        import autokeras as ak

        return ak.load_model(model_path)
    except Exception:
        pass

    custom_objects = {}
    try:
        from autokeras.keras_layers import CastToFloat32

        custom_objects["CastToFloat32"] = CastToFloat32
    except Exception:
        pass
    try:
        from autokeras.preprocessors.postprocessors import SigmoidPostprocessor

        custom_objects["SigmoidPostprocessor"] = SigmoidPostprocessor
    except Exception:
        pass

    return keras.models.load_model(
        model_path, compile=False, custom_objects=custom_objects
    )


def f_beta(precision, recall, beta: float):
    """Vectorized F_beta."""
    b2 = beta * beta
    denom = b2 * precision + recall
    return (
        (1.0 + b2)
        * precision
        * recall
        / np.where(denom == 0, np.finfo(float).eps, denom)
    )


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-pickle",
        type=Path,
        default=Path("combined_train_features_labels.pickle"),
        help="Pickle with (train_features, train_labels, input_shape)",
    )
    ap.add_argument(
        "--model-path", type=Path, required=True, help="Path to .keras model"
    )
    ap.add_argument(
        "--out-file", type=Path, default=Path("models/decision_threshold.npy")
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=1024)

    # Floors and objective
    ap.add_argument("--min-recall", type=float, default=0.95)
    ap.add_argument(
        "--min-precision",
        type=float,
        default=0.90,
        help="New: enforce a precision floor together with recall",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Maximize F_beta among thresholds that meet both floors",
    )
    args = ap.parse_args()

    # Match your runtime (avoid grabbing all GPU memory during eval)
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # 1) Load data and reconstruct the train/val split you used in training
    X_all, y_all, _ = pickle.load(open(args.train_pickle, "rb"))
    X_all = np.asarray(X_all)
    y_all = np.asarray(y_all).astype(int).ravel()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all,
        y_all,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_all,
    )

    # 2) Load model and get validation probabilities
    model = load_model(args.model_path)
    y_prob = model.predict(X_val, batch_size=args.batch_size, verbose=0).ravel()

    # 3) Precision–Recall curve
    # Note: precision_recall_curve returns p,r length N+1 and thresholds length N.
    p, r, t = precision_recall_curve(y_val, y_prob)
    p_thr, r_thr, thr = p[:-1], r[:-1], t

    # 4) Constrained selection: meet BOTH floors
    feasible = (r_thr >= args.min_recall) & (p_thr >= args.min_precision)

    if feasible.any():
        scores = f_beta(p_thr[feasible], r_thr[feasible], args.beta)
        # tie-breaker: prefer higher recall inside feasible set
        tie_break = r_thr[feasible] * 1e-6
        idx_local = np.argmax(scores + tie_break)
        chosen_thr = float(thr[feasible][idx_local])
        prec, rec = float(p_thr[feasible][idx_local]), float(r_thr[feasible][idx_local])
        reason = (
            f"max F{args.beta:g} with precision≥{args.min_precision:g} "
            f"and recall≥{args.min_recall:g}"
        )
        floors_satisfied = True
    else:
        # No threshold meets both floors -> report how close we got
        max_recall_at_pfloor = (
            r_thr[p_thr >= args.min_precision].max()
            if (p_thr >= args.min_precision).any()
            else 0.0
        )
        max_precision_at_rfloor = (
            p_thr[r_thr >= args.min_recall].max()
            if (r_thr >= args.min_recall).any()
            else 0.0
        )

        # Fallback: global max F_beta (default F1)
        scores = f_beta(p_thr, r_thr, args.beta)
        idx = int(np.argmax(scores))
        chosen_thr = float(thr[idx])
        prec, rec = float(p_thr[idx]), float(r_thr[idx])
        reason = (
            f"fallback to max F{args.beta:g} (no threshold met both floors); "
            f"max recall at P≥{args.min_precision:g}: {max_recall_at_pfloor:.3f}; "
            f"max precision at R≥{args.min_recall:g}: {max_precision_at_rfloor:.3f}"
        )
        floors_satisfied = False

    # 5) Confusion matrix on VAL at the chosen threshold
    y_hat = (y_prob >= chosen_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_hat).ravel()

    # 6) Save threshold + sidecar JSON
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_file, np.array([chosen_thr]))
    print(f"Saved threshold -> {args.out_file} = {chosen_thr:.4f}")
    print(f"VAL @thr: P={prec:.3f}, R={rec:.3f}, TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Reason: {reason}")

    summary = {
        "threshold": chosen_thr,
        "floors_satisfied": bool(floors_satisfied),
        "min_precision": float(args.min_precision),
        "min_recall": float(args.min_recall),
        "beta": float(args.beta),
        "val_precision": float(prec),
        "val_recall": float(rec),
        "val_tp": int(tp),
        "val_fp": int(fp),
        "val_tn": int(tn),
        "val_fn": int(fn),
        "seed": int(args.seed),
        "val_size": float(args.val_size),
        "model_path": str(args.model_path),
        "train_pickle": str(args.train_pickle),
        "note": "Threshold chosen on VALIDATION; evaluate on TEST with this fixed threshold.",
    }
    with open(args.out_file.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
