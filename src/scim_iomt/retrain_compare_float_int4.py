#!/usr/bin/env python3
"""Retrain the released compact model and compare float and effective-INT4 views.

This workflow mirrors the released compact-model pipeline:

1. train the released compact float model
2. fine-tune the fixed-range QAT model
3. evaluate two model views on the held-out split from the same training XML

The public comparison is intentionally defined as:

- `float`: native Keras float checkpoint
- `int4_with_scale`: effective signed quantized weights in the range `[-7, 7]`

The effective-INT4 view uses INT8-quantized inputs and activations
throughout the released QAT evaluation path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import metrics  # noqa: E402
from scim_iomt import fixed_quant_search_runner as fsr  # noqa: E402
from scim_iomt import compare_utils  # noqa: E402

RUN_NAME = "compact_l32_d128_d64"


def build_dataset_bundle(patient_id: str):
    """Load the same 0.6/0.2/0.2 split used in the released compact training flow."""
    train_xml = (PROJECT_ROOT / "data" / "OhioT1DM" / "2018" / "train" / f"{patient_id}-ws-training.xml").resolve()
    if not train_xml.exists():
        raise FileNotFoundError(f"Training XML not found: {train_xml}")

    base_cfg = fsr.load_base_cfg()
    base_cfg["dataset"]["xml_path"] = str(train_xml)
    base_cfg["dataset"]["train_fraction"] = 0.6
    base_cfg["dataset"]["valid_fraction"] = 0.2
    base_cfg["dataset"]["test_fraction"] = 0.2
    base_cfg["dataset"]["nb_future_steps"] = 1

    dataset_module = fsr.load_module(base_cfg["dataset"]["script_path"])
    dataset_bundle = dataset_module.load_dataset(base_cfg["dataset"])
    return base_cfg, dataset_bundle


def run_training(results_root: Path, patient_id: str, args) -> tuple[dict, Path]:
    """Train float plus fixed-QAT compact model into an isolated results root."""
    base_cfg, dataset_bundle = build_dataset_bundle(patient_id)
    specs = [spec for spec in fsr.build_experiment_grid() if spec.run_name == RUN_NAME]
    if not specs:
        raise RuntimeError(f"Could not find {RUN_NAME} in the experiment grid.")
    spec = specs[0]

    results_root.mkdir(parents=True, exist_ok=True)
    old_results_root = fsr.RESULTS_ROOT
    fsr.RESULTS_ROOT = results_root
    try:
        payload = fsr.run_one_experiment(spec, base_cfg, dataset_bundle, args)
        fsr.write_results([payload])
        fsr.compute_pareto(
            [
                {
                    "run_name": payload["config"]["run_name"],
                    "nb_lstm_states": payload["config"]["nb_lstm_states"],
                    "dense_units_1": payload["config"]["dense_units_1"],
                    "dense_units_2": payload["config"]["dense_units_2"],
                    "param_count": payload["config"]["param_count"],
                    "float_rmse": payload["metrics"]["float"]["rmse"],
                    "qat_rmse": payload["metrics"]["fixed_quant_qat"]["rmse"],
                    "qat_seg_mean": payload["metrics"]["fixed_quant_qat"]["seg_mean"],
                    "delta_rmse": payload["metrics"]["delta_qat_vs_float"]["rmse"],
                    "hidden_abs": payload["calibration"]["hidden_abs"],
                    "relu1_max": payload["calibration"]["relu1_max"],
                    "relu2_max": payload["calibration"]["relu2_max"],
                    "total_seconds": payload["training"]["total_seconds"],
                }
            ]
        )
        metrics_path = results_root / "runs" / spec.run_name / "metrics.json"
    finally:
        fsr.RESULTS_ROOT = old_results_root

    return payload, metrics_path


def summarize_predictions(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> dict:
    seg_mean, seg_std = metrics.surveillance_error(y_true, y_mean)
    abs_err = np.abs(y_mean - y_true)
    return {
        "rmse_mgdl": float(metrics.root_mean_squared_error(y_true, y_mean)),
        "mae_mgdl": float(np.mean(abs_err)),
        "seg_mean": float(seg_mean),
        "seg_std": float(seg_std),
        "mean_predicted_std_mgdl": float(np.mean(y_std)),
    }


def evaluate_float_checkpoint(metrics_payload: dict, x_test: np.ndarray, y_test: np.ndarray, scale: float) -> dict:
    """Evaluate the float checkpoint with the native Keras model implementation."""
    base_cfg = fsr.load_base_cfg()
    base_cfg["model"]["nb_lstm_states"] = int(metrics_payload["config"]["nb_lstm_states"])
    base_cfg["model"]["dense_units_1"] = int(metrics_payload["config"]["dense_units_1"])
    base_cfg["model"]["dense_units_2"] = int(metrics_payload["config"]["dense_units_2"])
    base_cfg["model"]["dropout_1"] = 0.2
    base_cfg["model"]["dropout_2"] = 0.3

    model_module = fsr.load_module(base_cfg["model"]["script_path"])
    float_model = model_module.load(x_test.shape[1:], 2, base_cfg["model"])
    float_model.load_weights(metrics_payload["paths"]["float_weights"])

    y_pred = float_model.predict(x_test, verbose=0)
    y_std = np.asarray(y_pred[:, 0]).reshape(-1) / scale
    y_mean = np.asarray(y_pred[:, 1]).reshape(-1) / scale
    y_true = np.asarray(y_test).reshape(-1) / scale
    return summarize_predictions(y_true, y_mean, y_std)


def evaluate_two_views(metrics_payload: dict) -> dict:
    """Evaluate float and effective-INT4 views."""
    _, _, _, _, x_test, y_test = compare_utils.load_dataset_from_metrics(metrics_payload)

    scale = 0.01
    calib = {
        "input_min": float(metrics_payload["calibration"]["input_min"]),
        "input_max": float(metrics_payload["calibration"]["input_max"]),
        "hidden_abs": float(metrics_payload["calibration"]["hidden_abs"]),
        "relu1_max": float(metrics_payload["calibration"]["relu1_max"]),
        "relu2_max": float(metrics_payload["calibration"]["relu2_max"]),
    }
    hidden_units = int(metrics_payload["config"]["nb_lstm_states"])

    int4_weights, _ = compare_utils.build_qat_quantized_weights(Path(metrics_payload["paths"]["qat_weights"]))
    y_true = np.asarray(y_test).reshape(-1) / scale

    int4_mu, int4_std = compare_utils.forward_quantized(x_test.astype(np.float32), int4_weights, calib, hidden_units)

    int4_mu = int4_mu / scale
    int4_std = int4_std / scale

    views = {
        "float": evaluate_float_checkpoint(metrics_payload, x_test.astype(np.float32), y_test, scale),
        "int4_with_scale": summarize_predictions(y_true, int4_mu, int4_std),
    }
    views["delta_int4_vs_float"] = {
        "rmse_mgdl": float(views["int4_with_scale"]["rmse_mgdl"] - views["float"]["rmse_mgdl"]),
        "mae_mgdl": float(views["int4_with_scale"]["mae_mgdl"] - views["float"]["mae_mgdl"]),
        "seg_mean": float(views["int4_with_scale"]["seg_mean"] - views["float"]["seg_mean"]),
    }
    views["quantization_definition"] = {
        "effective_int4_signed_range": [-7, 7],
        "scale_type": "per-output-channel",
    }
    views["evaluation_scope"] = {
        "dataset": "held-out split from training XML",
        "split": [0.6, 0.2, 0.2],
    }
    return views


def write_outputs(results_root: Path, comparison: dict) -> tuple[Path, Path, Path]:
    json_path = results_root / "compact_float_int4_comparison.json"
    csv_path = results_root / "compact_float_int4_comparison.csv"
    md_path = results_root / "compact_float_int4_comparison.md"

    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for key in ["float", "int4_with_scale"]:
        row = {"view": key}
        row.update(comparison[key])
        rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Compact Model Comparison",
        "",
        "Views compared:",
        "- `float`: native float checkpoint evaluated with the original Keras model",
        "- `int4_with_scale`: effective signed quantized weights reconstructed from the QAT checkpoint with per-output-channel scale",
        "- the effective-INT4 view uses INT8-quantized inputs and activations",
        "",
        "Evaluation scope:",
        "- held-out split from `563-ws-training.xml` using the released `0.6 / 0.2 / 0.2` split",
        "",
        "| view | rmse_mgdl | mae_mgdl | seg_mean | seg_std | mean_predicted_std_mgdl |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ["float", "int4_with_scale"]:
        m = comparison[key]
        lines.append(
            f"| {key} | {m['rmse_mgdl']:.6f} | {m['mae_mgdl']:.6f} | {m['seg_mean']:.6f} | {m['seg_std']:.6f} | {m['mean_predicted_std_mgdl']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Deltas vs float:",
            f"- INT4 vs float RMSE delta: {comparison['delta_int4_vs_float']['rmse_mgdl']:.6f} mg/dL",
            "",
            "Quantization definition:",
            f"- effective signed range: {comparison['quantization_definition']['effective_int4_signed_range']}",
            f"- scale type: {comparison['quantization_definition']['scale_type']}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", default="563")
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "outputs" / "compact_qat_patient_563_compare"),
        help="Directory for the new training run and comparison outputs.",
    )
    parser.add_argument("--float-epochs", type=int, default=None)
    parser.add_argument("--float-patience", type=int, default=None)
    parser.add_argument("--qat-epochs", type=int, default=300)
    parser.add_argument("--qat-patience", type=int, default=50)
    parser.add_argument("--qat-learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    resolved_device = fsr.configure_tensorflow_device(args.device)

    class RunnerArgs:
        pass

    runner_args = RunnerArgs()
    runner_args.float_epochs = args.float_epochs
    runner_args.float_patience = args.float_patience
    runner_args.qat_epochs = args.qat_epochs
    runner_args.qat_patience = args.qat_patience
    runner_args.qat_learning_rate = args.qat_learning_rate
    runner_args.skip_existing = args.skip_existing

    payload, metrics_path = run_training(results_root, str(args.patient_id), runner_args)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    comparison = {
        "run_name": RUN_NAME,
        "patient_id": str(args.patient_id),
        "results_root": str(results_root),
        "metrics_json": str(metrics_path),
        "device": resolved_device,
        "training_summary": payload["training"],
        "comparison": evaluate_two_views(metrics_payload),
    }
    json_path, csv_path, md_path = write_outputs(results_root, comparison["comparison"])

    print(f"results_root={results_root}")
    print(f"metrics_json={metrics_path}")
    print(f"comparison_json={json_path}")
    print(f"comparison_csv={csv_path}")
    print(f"comparison_md={md_path}")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
