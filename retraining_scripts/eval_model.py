#!/usr/bin/env python3
"""
Evaluate a saved Keras/AutoKeras model on test data.

Features:
- Loads model (.keras) and test pickle (features, labels, input_shape)
- Predicts probabilities on the test set
- Writes PR-curve CSV across thresholds + PNG plot
- If --threshold-file is provided, evaluates ONLY at that fixed threshold (no test-time tuning)
- Otherwise: finds an operating point by maximizing F_beta subject to recall/precision floors

Example (with fixed threshold from validation):
  uv run python eval_model.py \
    --model-path models/retrained_best_model_combined.keras \
    --test-pickle test_features_labels_2.pickle \
    --threshold-file models/decision_threshold.npy \
    --out-dir eval_out --batch-size 1024
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

import tensorflow as tf
from tensorflow import keras


def load_test_data(pickle_path: Path):
    with open(pickle_path, "rb") as f:
        test_features, test_labels, input_shape = pickle.load(f)
    test_features = np.asarray(test_features)
    test_labels = np.asarray(test_labels).astype(int).ravel()
    return test_features, test_labels, input_shape


def load_model(model_path: Path):
    # Prefer AutoKeras loader when available
    try:
        import autokeras as ak

        return ak.load_model(model_path)
    except Exception:
        pass

    # Fallback: standard Keras loader with AK custom objects if needed
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


def compute_pr(y_true: np.ndarray, y_prob: np.ndarray):
    # sklearn returns precision, recall of length N+1 and thresholds length N
    p, r, t = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    df = pd.DataFrame(
        {
            "threshold": t,
            "precision": p[:-1],
            "recall": r[:-1],
        }
    )
    denom = (df["precision"] + df["recall"]).replace(0, np.finfo(float).eps)
    df["f1"] = 2.0 * df["precision"] * df["recall"] / denom
    return df, p, r, t, ap


def confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_hat = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, np.finfo(float).eps)
    return dict(
        threshold=float(thr),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
    )


def f_beta(prec, rec, beta: float):
    b2 = beta * beta
    denom = b2 * prec + rec
    return (1 + b2) * prec * rec / np.where(denom == 0, np.finfo(float).eps, denom)


def annotate_best(ax, row, label_prefix="Chosen"):
    ax.scatter([row["recall"]], [row["precision"]])
    ax.annotate(
        f"{label_prefix}\nth={row['threshold']:.3f}\nP={row['precision']:.3f}, R={row['recall']:.3f}",
        (row["recall"], row["precision"]),
        textcoords="offset points",
        xytext=(10, -20),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to .keras model (e.g., models/retrained_best_model_combined.keras)",
    )
    parser.add_argument(
        "--test-pickle",
        type=Path,
        required=True,
        help="Pickle with (test_features, test_labels, input_shape)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("eval_out4"))
    parser.add_argument("--batch-size", type=int, default=1024)

    # New: fixed-threshold and selection policy
    parser.add_argument(
        "--threshold-file",
        type=Path,
        default=None,
        help="If provided, load this .npy threshold and evaluate ONLY at that value.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.95,
        help="When scanning thresholds (no threshold-file), enforce recall >= this.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.0,
        help="Optional precision floor during scan.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="F_beta used to pick threshold when scanning (default F1).",
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "pr_thresholds.csv"
    png_path = args.out_dir / "pr_curve.png"
    summary_path = args.out_dir / "summary.json"

    # Make TF quieter + avoid GPU OOM surprises during eval
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # 1) Load
    print(f"Loading test data from {args.test_pickle} ...")
    X, y, input_shape = load_test_data(args.test_pickle)
    print(f"Test samples: {len(y):,}, input_shape: {input_shape}")

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path)

    # 2) Predict
    print("Predicting probabilities ...")
    y_prob = model.predict(X, batch_size=args.batch_size, verbose=0).ravel()

    # 3) PR curve + AP (for reporting/plot regardless)
    df_thr, p_all, r_all, t_all, ap = compute_pr(y, y_prob)
    auc = roc_auc_score(y, y_prob)
    df_thr.to_csv(csv_path, index=False)
    print(f"Average precision (AP): {ap:.4f} | ROC AUC: {auc:.4f}")
    print(f"Wrote threshold sweep CSV -> {csv_path}")

    # 4) Choose evaluation point
    fixed_threshold = None
    status_line = ""
    chosen_row = None

    if args.threshold_file and args.threshold_file.exists():
        # Use fixed threshold (recommended: from validation)
        fixed_threshold = float(np.load(args.threshold_file)[0])
        chosen_row = confusion_at_threshold(y, y_prob, fixed_threshold)
        status_line = (
            f"Using FIXED threshold {fixed_threshold:.4f} -> "
            f"P={chosen_row['precision']:.3f}, R={chosen_row['recall']:.3f}, "
            f"TP={chosen_row['tp']}, FP={chosen_row['fp']}, "
            f"TN={chosen_row['tn']}, FN={chosen_row['fn']}"
        )
        label_prefix = "Fixed"
    else:
        # Scan thresholds on TEST (ok for analysis, not for final reporting)
        mask = (df_thr["recall"] >= args.min_recall) & (
            df_thr["precision"] >= args.min_precision
        )
        if mask.any():
            cand = df_thr.loc[mask].copy()
            cand["f_beta"] = f_beta(
                cand["precision"].values, cand["recall"].values, args.beta
            )
            best_idx = cand["f_beta"].idxmax()
            chosen_row = df_thr.loc[best_idx].to_dict()
            status_line = (
                f"Selected by F{args.beta:.1f} with recall≥{args.min_recall}, "
                f"precision≥{args.min_precision}: th={chosen_row['threshold']:.4f} "
                f"-> P={chosen_row['precision']:.3f}, R={chosen_row['recall']:.3f}"
            )
            label_prefix = "Chosen"
        else:
            # Fallback: global max F1
            best_idx = df_thr["f1"].idxmax()
            chosen_row = df_thr.loc[best_idx].to_dict()
            status_line = (
                f"No threshold met constraints; using global max F1: th={chosen_row['threshold']:.4f} "
                f"-> P={chosen_row['precision']:.3f}, R={chosen_row['recall']:.3f}"
            )
            label_prefix = "Best F1"

    print(status_line)

    # 5) Plot PR curve
    plt.figure(figsize=(7, 6))
    plt.plot(r_all, p_all, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (AP={ap:.3f})\n{status_line}")
    annotate_best(plt.gca(), chosen_row, label_prefix=label_prefix)
    # guide lines for 0.90 operating point
    plt.axhline(0.90, linestyle="--", linewidth=1)
    plt.axvline(0.90, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    print(f"Wrote PR curve PNG -> {png_path}")

    # 6) Also dump a tiny summary JSON for easy ingestion
    # If we scanned, compute confusion at that chosen threshold too
    if fixed_threshold is None:
        chosen_conf = confusion_at_threshold(y, y_prob, float(chosen_row["threshold"]))
        chosen_threshold = float(chosen_row["threshold"])
    else:
        chosen_conf = chosen_row
        chosen_threshold = fixed_threshold

    summary = {
        "model_path": str(args.model_path),
        "test_pickle": str(args.test_pickle),
        "fixed_threshold_used": fixed_threshold is not None,
        "threshold": chosen_threshold,
        "precision": chosen_conf["precision"],
        "recall": chosen_conf["recall"],
        "tp": chosen_conf["tp"],
        "fp": chosen_conf["fp"],
        "tn": chosen_conf["tn"],
        "fn": chosen_conf["fn"],
        "average_precision": float(ap),
        "roc_auc": float(auc),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
