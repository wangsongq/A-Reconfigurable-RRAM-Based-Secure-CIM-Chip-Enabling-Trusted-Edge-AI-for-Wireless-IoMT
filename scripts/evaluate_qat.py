#!/usr/bin/env python3
"""Evaluate a trained fixed-QAT model and report RMSE / SEG metrics on a test XML."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import metrics
from scim_iomt.fixed_quant_search_runner import configure_tensorflow_device, load_base_cfg


def load_module(script_path: str):
    """Safely import a repo-local Python module referenced by the config."""
    resolved_path = Path(script_path).resolve()
    try:
        resolved_path.relative_to(SRC_ROOT)
    except ValueError as exc:
        raise ValueError(f"Refusing to import script outside src/: {resolved_path}") from exc
    spec = importlib.util.spec_from_file_location("module.name", resolved_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_mu_std(y_pred):
    """Split probabilistic output into mean and standard deviation arrays."""
    y_pred = np.asarray(y_pred)
    return y_pred[:, 1].reshape(-1), y_pred[:, 0].reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory containing metrics.json and qat_model.weights.h5")
    parser.add_argument(
        "--test-xml",
        required=True,
        help="Path to a testing XML file, e.g. data/OhioT1DM/2018/test/563-ws-testing.xml",
    )
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()

    resolved_device = configure_tensorflow_device(args.device)

    run_dir = Path(args.run_dir).resolve()
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {run_dir}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = load_base_cfg()
    cfg["dataset"]["xml_path"] = str(Path(args.test_xml).resolve())
    cfg["dataset"]["train_fraction"] = 0.0
    cfg["dataset"]["valid_fraction"] = 0.0
    cfg["dataset"]["test_fraction"] = 1.0
    cfg["dataset"]["nb_future_steps"] = 1
    cfg["model"]["script_path"] = str(SRC_ROOT / "scim_iomt" / "fixed_quant_qat_model.py")
    cfg["model"]["nb_lstm_states"] = int(payload["config"]["nb_lstm_states"])
    cfg["model"]["dense_units_1"] = int(payload["config"]["dense_units_1"])
    cfg["model"]["dense_units_2"] = int(payload["config"]["dense_units_2"])
    cfg["model"]["dropout_1"] = 0.2
    cfg["model"]["dropout_2"] = 0.3
    cfg["model"]["quant_input_min"] = float(payload["calibration"]["input_min"])
    cfg["model"]["quant_input_max"] = float(payload["calibration"]["input_max"])
    cfg["model"]["quant_hidden_abs"] = float(payload["calibration"]["hidden_abs"])
    cfg["model"]["quant_relu1_max"] = float(payload["calibration"]["relu1_max"])
    cfg["model"]["quant_relu2_max"] = float(payload["calibration"]["relu2_max"])
    cfg["model"]["quant_activation_bits"] = 8
    cfg["model"]["quant_weight_bits"] = 3

    dataset_module = load_module(cfg["dataset"]["script_path"])
    _, _, _, _, x_test, y_test = dataset_module.load_dataset(cfg["dataset"])
    model_module = load_module(cfg["model"]["script_path"])
    model = model_module.load(x_test.shape[1:], 2, cfg["model"])
    model.load_weights(str(run_dir / "qat_model.weights.h5"))

    y_pred = model.predict(x_test, verbose=0)
    scale = float(cfg["dataset"]["scale"])
    y_true = np.asarray(y_test).reshape(-1) / scale
    y_mu, y_std = get_mu_std(y_pred)
    y_mu = y_mu / scale
    y_std = y_std / scale

    seg_mean, seg_std = metrics.surveillance_error(y_true, y_mu)
    result = {
        "rmse_mgdl": float(metrics.root_mean_squared_error(y_true, y_mu)),
        "seg": {
            "mean": float(seg_mean),
            "std": float(seg_std),
        },
        "mean_predicted_std_mgdl": float(np.mean(y_std)),
        "num_samples": int(len(y_true)),
    }
    # Save a small machine-readable summary alongside the trained checkpoint.
    out_path = run_dir / "evaluation_metrics.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"device={resolved_device}")
    print(f"saved_metrics={out_path}")


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    main()
