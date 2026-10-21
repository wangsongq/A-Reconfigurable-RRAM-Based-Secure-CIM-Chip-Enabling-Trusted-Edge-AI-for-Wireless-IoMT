#!/usr/bin/env python3
"""Search compact network sizes under fixed-range QAT deployment settings."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
MODULE_ROOT = SRC_ROOT / "scim_iomt"
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "fixed_quant_search_563"
BASE_YAML = PROJECT_ROOT / "configs" / "563_final_experiment.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import metrics  # noqa: E402
from scim_iomt.fixed_quant_qat_model import copy_weights_from_float_model  # noqa: E402


@dataclass(frozen=True)
class ExperimentSpec:
    index: int
    run_name: str
    nb_lstm_states: int
    dense_units_1: int
    dense_units_2: int


def load_module(script_path: str):
    """Safely import only modules that live inside this repository's src tree."""
    resolved_path = Path(script_path).resolve()
    try:
        resolved_path.relative_to(SRC_ROOT)
    except ValueError as exc:
        raise ValueError(f"Refusing to import script outside src/: {resolved_path}") from exc
    spec = importlib.util.spec_from_file_location("module.name", resolved_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_paths_absolute(dir_, cfg):
    """Resolve all *_path config entries relative to the base YAML location."""
    if not isinstance(cfg, dict):
        return cfg
    for key in list(cfg.keys()):
        value = cfg[key]
        if isinstance(value, str) and key.endswith("_path"):
            cfg[key] = os.path.abspath(os.path.join(dir_, value))
        elif isinstance(value, dict):
            cfg[key] = make_paths_absolute(dir_, value)
    return cfg


def load_base_cfg():
    """Load the baseline experiment config used by compact float/QAT search."""
    with open(BASE_YAML, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg = make_paths_absolute(str(BASE_YAML.parent), cfg)
    if "param_nb_future_steps" in cfg["dataset"]:
        cfg["dataset"]["nb_future_steps"] = int(cfg["dataset"]["param_nb_future_steps"][0])
        del cfg["dataset"]["param_nb_future_steps"]
    cfg["model"]["script_path"] = str(MODULE_ROOT / "compact_lstm_model.py")
    cfg["loss_function"]["script_path"] = str(MODULE_ROOT / "local_nll_keras.py")
    return cfg


def set_global_seed(seed: int):
    """Set Python, NumPy, and TensorFlow seeds for reproducible search runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)


def configure_tensorflow_device(device: str = "gpu") -> str:
    """Configure TensorFlow to prefer GPU or CPU before model construction."""
    requested = str(device).lower()
    if requested not in {"gpu", "cpu"}:
        raise ValueError(f"Unsupported device mode: {device}")

    if requested == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return "cpu"

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[warn] no GPU detected, falling back to CPU")
        return "cpu"
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    return "gpu"


def build_experiment_grid() -> list[ExperimentSpec]:
    """Enumerate the compact model candidates evaluated by the QAT search."""
    lstm_sizes = [256, 192, 160, 128, 96, 64, 48, 32]
    head_sizes = [(512, 256), (384, 192), (256, 128), (128, 64)]
    specs = []
    index = 0
    for dense_units_1, dense_units_2 in head_sizes:
        for nb_lstm_states in lstm_sizes:
            run_name = f"compact_l{nb_lstm_states}_d{dense_units_1}_d{dense_units_2}"
            specs.append(
                ExperimentSpec(
                    index=index,
                    run_name=run_name,
                    nb_lstm_states=nb_lstm_states,
                    dense_units_1=dense_units_1,
                    dense_units_2=dense_units_2,
                )
            )
            index += 1
    return specs


def clone_cfg(base_cfg, spec: ExperimentSpec, artifacts_path: Path):
    """Clone the base config and inject one architecture choice plus output path."""
    cfg = json.loads(json.dumps(base_cfg))
    cfg["model"]["nb_lstm_states"] = spec.nb_lstm_states
    cfg["model"]["dense_units_1"] = spec.dense_units_1
    cfg["model"]["dense_units_2"] = spec.dense_units_2
    cfg["model"]["dropout_1"] = 0.2
    cfg["model"]["dropout_2"] = 0.3
    cfg["train"]["artifacts_path"] = str(artifacts_path)
    cfg["train"]["seed"] = 0
    return cfg


def fit_model(
    model,
    x_train,
    y_train,
    x_valid,
    y_valid,
    *,
    optimizer,
    loss_function,
    epochs: int,
    batch_size: int,
    patience: int,
    shuffle: bool,
    csv_path: Path,
):
    model.compile(optimizer=optimizer, loss=loss_function)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(10, patience // 4),
            min_lr=1e-6,
            verbose=0,
            mode="min",
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path)),
    ]
    start = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        callbacks=callbacks,
        verbose=0,
    )
    elapsed = time.time() - start
    return history, elapsed


def get_mu_and_std(y_pred):
    y_pred = np.asarray(y_pred)
    y_std = y_pred[:, 0].reshape(-1)
    y_mu = y_pred[:, 1].reshape(-1)
    return y_mu, y_std


def evaluate_predictions(y_true, y_pred, scale):
    y_mu, y_std = get_mu_and_std(y_pred)
    y_true = np.asarray(y_true).reshape(-1) / scale
    y_mu = y_mu / scale
    y_std = y_std / scale
    seg = metrics.surveillance_error(y_true, y_mu)
    return {
        "rmse": float(metrics.root_mean_squared_error(y_true, y_mu)),
        "seg_mean": float(seg[0]),
        "seg_std": float(seg[1]),
        "mean_std": float(np.mean(y_std)),
    }


def evaluate_model(model, x_test, y_test, scale):
    y_pred = model.predict(x_test, verbose=0)
    return evaluate_predictions(y_test, y_pred, scale), y_pred


def round_up(value: float, step: float, minimum: float | None = None, cap: float | None = None):
    rounded = np.ceil(float(value) / step) * step
    if minimum is not None:
        rounded = max(rounded, minimum)
    if cap is not None:
        rounded = min(rounded, cap)
    return float(rounded)


def calibrate_ranges(float_model, x_cal):
    lstm_layer = float_model.get_layer("lstm")

    seq_inputs = tf.keras.layers.Input(shape=x_cal.shape[1:])
    seq_outputs = tf.keras.layers.LSTM(
        units=int(lstm_layer.units),
        unit_forget_bias=True,
        return_sequences=True,
    )(seq_inputs)
    hidden_seq_model = tf.keras.Model(inputs=seq_inputs, outputs=seq_outputs)
    hidden_seq_model.layers[1].set_weights(lstm_layer.get_weights())

    relu1_model = tf.keras.Model(inputs=float_model.input, outputs=float_model.get_layer("activation").output)
    relu2_model = tf.keras.Model(inputs=float_model.input, outputs=float_model.get_layer("activation_1").output)

    hidden_seq = hidden_seq_model.predict(x_cal, batch_size=1024, verbose=0).reshape(-1)
    relu1 = relu1_model.predict(x_cal, batch_size=1024, verbose=0).reshape(-1)
    relu2 = relu2_model.predict(x_cal, batch_size=1024, verbose=0).reshape(-1)

    hidden_abs_p99_9 = float(np.percentile(np.abs(hidden_seq), 99.9))
    relu1_p99_9 = float(np.percentile(relu1, 99.9))
    relu2_p99_9 = float(np.percentile(relu2, 99.9))

    hidden_abs = round_up(hidden_abs_p99_9 * 1.05, 0.05, minimum=0.25, cap=1.0)
    relu1_max = round_up(relu1_p99_9 * 1.05, 0.05, minimum=0.10)
    relu2_max = round_up(relu2_p99_9 * 1.05, 0.05, minimum=0.10)

    return {
        "input_min": 0.5,
        "input_max": 3.5,
        "hidden_abs": hidden_abs,
        "relu1_max": relu1_max,
        "relu2_max": relu2_max,
        "raw": {
            "hidden_abs_p99_9": hidden_abs_p99_9,
            "relu1_p99_9": relu1_p99_9,
            "relu2_p99_9": relu2_p99_9,
        },
    }


def build_qat_model(cfg, input_shape, ranges):
    qat_cfg = json.loads(json.dumps(cfg))
    qat_cfg["model"]["script_path"] = str(MODULE_ROOT / "fixed_quant_qat_model.py")
    qat_cfg["model"]["quant_input_min"] = ranges["input_min"]
    qat_cfg["model"]["quant_input_max"] = ranges["input_max"]
    qat_cfg["model"]["quant_hidden_abs"] = ranges["hidden_abs"]
    qat_cfg["model"]["quant_relu1_max"] = ranges["relu1_max"]
    qat_cfg["model"]["quant_relu2_max"] = ranges["relu2_max"]
    qat_cfg["model"]["quant_activation_bits"] = 8
    qat_cfg["model"]["quant_weight_bits"] = 3

    model_module = load_module(qat_cfg["model"]["script_path"])
    model = model_module.load(input_shape, 2, qat_cfg["model"])
    return model, qat_cfg


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_model(spec: ExperimentSpec, model):
    return {
        "run_name": spec.run_name,
        "nb_lstm_states": spec.nb_lstm_states,
        "dense_units_1": spec.dense_units_1,
        "dense_units_2": spec.dense_units_2,
        "param_count": int(model.count_params()),
    }


def run_one_experiment(spec: ExperimentSpec, base_cfg, dataset_bundle, args):
    run_dir = RESULTS_ROOT / "runs" / spec.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    cfg = clone_cfg(base_cfg, spec, run_dir)
    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset_bundle
    scale = float(cfg["dataset"].get("scale", 1.0))

    optimizer_module = load_module(cfg["optimizer"]["script_path"])
    loss_module = load_module(cfg["loss_function"]["script_path"])
    float_model_module = load_module(cfg["model"]["script_path"])

    set_global_seed(int(cfg["train"]["seed"]))
    tf.keras.backend.clear_session()

    float_model = float_model_module.load(x_train.shape[1:], 2, cfg["model"])
    float_history, float_seconds = fit_model(
        float_model,
        x_train,
        y_train,
        x_valid,
        y_valid,
        optimizer=optimizer_module.load(cfg["optimizer"]),
        loss_function=loss_module.load(),
        epochs=int(args.float_epochs or cfg["train"]["epochs"]),
        batch_size=int(cfg["train"]["batch_size"]),
        patience=int(args.float_patience or cfg["train"]["patience"]),
        shuffle=bool(cfg["train"]["shuffle"]),
        csv_path=run_dir / "float_history.csv",
    )
    float_model.save_weights(str(run_dir / "float_model.weights.h5"))

    float_metrics, _ = evaluate_model(float_model, x_test, y_test, scale)
    ranges = calibrate_ranges(float_model, x_train)
    save_json(run_dir / "calibration_ranges.json", ranges)

    qat_model, qat_cfg = build_qat_model(cfg, x_train.shape[1:], ranges)
    copy_weights_from_float_model(float_model, qat_model)

    qat_optimizer_cfg = dict(cfg["optimizer"])
    qat_optimizer_cfg["learning_rate"] = args.qat_learning_rate
    qat_history, qat_seconds = fit_model(
        qat_model,
        x_train,
        y_train,
        x_valid,
        y_valid,
        optimizer=optimizer_module.load(qat_optimizer_cfg),
        loss_function=loss_module.load(),
        epochs=args.qat_epochs,
        batch_size=int(cfg["train"]["batch_size"]),
        patience=args.qat_patience,
        shuffle=bool(cfg["train"]["shuffle"]),
        csv_path=run_dir / "qat_history.csv",
    )
    qat_model.save_weights(str(run_dir / "qat_model.weights.h5"))

    quant_metrics, _ = evaluate_model(qat_model, x_test, y_test, scale)
    architecture_summary = summarize_model(spec, qat_model)

    result = {
        "status": "complete",
        "config": architecture_summary,
        "dataset": {
            "xml_path": cfg["dataset"]["xml_path"],
            "split": {
                "train_fraction": float(cfg["dataset"]["train_fraction"]),
                "valid_fraction": float(cfg["dataset"]["valid_fraction"]),
                "test_fraction": float(cfg["dataset"]["test_fraction"]),
            },
        },
        "calibration": ranges,
        "training": {
            "float_epochs_ran": len(float_history.epoch),
            "qat_epochs_ran": len(qat_history.epoch),
            "float_seconds": float(float_seconds),
            "qat_seconds": float(qat_seconds),
            "total_seconds": float(float_seconds + qat_seconds),
        },
        "metrics": {
            "float": float_metrics,
            "fixed_quant_qat": quant_metrics,
            "delta_qat_vs_float": {
                "rmse": float(quant_metrics["rmse"] - float_metrics["rmse"]),
                "seg_mean": float(quant_metrics["seg_mean"] - float_metrics["seg_mean"]),
                "mean_std": float(quant_metrics["mean_std"] - float_metrics["mean_std"]),
            },
        },
        "paths": {
            "float_weights": str(run_dir / "float_model.weights.h5"),
            "qat_weights": str(run_dir / "qat_model.weights.h5"),
            "calibration_ranges": str(run_dir / "calibration_ranges.json"),
        },
    }
    save_json(metrics_path, result)
    print(
        "[done] {} params={} float_rmse={:.4f} qat_rmse={:.4f}".format(
            spec.run_name,
            architecture_summary["param_count"],
            float_metrics["rmse"],
            quant_metrics["rmse"],
        )
    )
    return result


def write_results(results):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_ROOT / "results.csv"
    rows = []
    for payload in results:
        row = {
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
        rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, rows


def compute_pareto(rows):
    rows = sorted(rows, key=lambda r: (r["param_count"], r["qat_rmse"]))
    frontier = []
    best_rmse = float("inf")
    for row in rows:
        if row["qat_rmse"] < best_rmse - 1e-9:
            frontier.append(row)
            best_rmse = row["qat_rmse"]
    return frontier


def write_report(rows, frontier):
    best = min(rows, key=lambda r: r["qat_rmse"])
    smallest = min(rows, key=lambda r: r["param_count"])

    lines = [
        "# Fixed-Quant Search Report",
        "",
        "Search setup:",
        "- 32 architectures: 8 LSTM widths x 4 dense-head settings.",
        "- Each model trained in two stages: float pretrain, then fixed-range QAT fine-tune.",
        "- Deployment target: 8-bit VMM input quantization with effective signed weight levels [-7, 7].",
        "",
        "Best fixed-QAT model:",
        f"- `{best['run_name']}` with `{best['param_count']}` params, fixed-QAT RMSE `{best['qat_rmse']:.4f}`.",
        "",
        "Smallest model:",
        f"- `{smallest['run_name']}` with `{smallest['param_count']}` params, fixed-QAT RMSE `{smallest['qat_rmse']:.4f}`.",
        "",
        "Pareto front (smaller is better for both params and RMSE):",
    ]
    for row in frontier:
        lines.append(
            f"- `{row['run_name']}`: params `{row['param_count']}`, fixed-QAT RMSE `{row['qat_rmse']:.4f}`, delta vs float `{row['delta_rmse']:.4f}`"
        )

    report_path = RESULTS_ROOT / "search_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def load_all_completed_rows():
    payloads = []
    runs_root = RESULTS_ROOT / "runs"
    if not runs_root.exists():
        return [], []
    for metrics_path in sorted(runs_root.glob("*/metrics.json")):
        payloads.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    if not payloads:
        return [], []
    _, rows = write_results(payloads)
    return payloads, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--weight-bits", type=int, default=3)
    parser.add_argument("--float-epochs", type=int, default=None)
    parser.add_argument("--float-patience", type=int, default=None)
    parser.add_argument("--qat-epochs", type=int, default=300)
    parser.add_argument("--qat-patience", type=int, default=50)
    parser.add_argument("--qat-learning-rate", type=float, default=2e-4)
    parser.add_argument("--only-run", type=str, default=None)
    parser.add_argument("--index-mod", type=int, default=None)
    parser.add_argument("--index-rem", type=int, default=None)
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        _, rows = load_all_completed_rows()
        if not rows:
            raise RuntimeError("No completed runs found to aggregate.")
        frontier = compute_pareto(rows)
        pareto_path = RESULTS_ROOT / "pareto_front.csv"
        with pareto_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(frontier[0].keys()))
            writer.writeheader()
            writer.writerows(frontier)
        report_path = write_report(rows, frontier)
        print(f"saved_pareto={pareto_path}")
        print(f"saved_report={report_path}")
        return

    base_cfg = load_base_cfg()
    dataset_module = load_module(base_cfg["dataset"]["script_path"])
    dataset_bundle = dataset_module.load_dataset(base_cfg["dataset"])

    payloads = []
    specs = build_experiment_grid()
    if args.only_run:
        specs = [spec for spec in specs if spec.run_name == args.only_run]
        if not specs:
            raise ValueError(f"Unknown run name: {args.only_run}")
    if args.index_mod is not None:
        if args.index_mod <= 0:
            raise ValueError("--index-mod must be positive")
        if args.index_rem is None:
            raise ValueError("--index-rem is required when --index-mod is set")
        specs = [spec for spec in specs if spec.index % args.index_mod == args.index_rem]

    for spec in specs:
        payloads.append(run_one_experiment(spec, base_cfg, dataset_bundle, args))

    if args.skip_summary:
        return

    csv_path, rows = write_results(payloads)
    frontier = compute_pareto(rows)
    pareto_path = RESULTS_ROOT / "pareto_front.csv"
    with pareto_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier[0].keys()))
        writer.writeheader()
        writer.writerows(frontier)
    report_path = write_report(rows, frontier)

    print(f"saved_results={csv_path}")
    print(f"saved_pareto={pareto_path}")
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    configure_tensorflow_device("gpu")
    main()
