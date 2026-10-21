#!/usr/bin/env python3
"""Train the original probabilistic LSTM model from a YAML experiment config."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(script_path: str):
    """Safely import a repo-local Python module from an experiment config."""
    resolved_path = Path(script_path).resolve()
    try:
        resolved_path.relative_to(SRC_ROOT)
    except ValueError as exc:
        raise ValueError(f"Refusing to import script outside src/: {resolved_path}") from exc
    spec = importlib.util.spec_from_file_location("module.name", resolved_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_paths_absolute(base_dir: Path, cfg):
    """Resolve all *_path entries relative to the YAML file location."""
    if not isinstance(cfg, dict):
        return cfg
    for key, value in list(cfg.items()):
        if isinstance(value, str) and key.endswith("_path"):
            cfg[key] = str((base_dir / value).resolve())
        elif isinstance(value, dict):
            cfg[key] = make_paths_absolute(base_dir, value)
    return cfg


def load_cfg(cfg_path: Path):
    """Load the experiment YAML and normalize list-valued singleton fields."""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = make_paths_absolute(cfg_path.parent, cfg)
    if "param_nb_future_steps" in cfg["dataset"]:
        cfg["dataset"]["nb_future_steps"] = int(cfg["dataset"]["param_nb_future_steps"][0])
        del cfg["dataset"]["param_nb_future_steps"]
    if "param_seed" in cfg["train"]:
        cfg["train"]["seed"] = int(cfg["train"]["param_seed"][0])
        del cfg["train"]["param_seed"]
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "563_final_experiment.yaml"),
        help="Path to the YAML experiment config.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)

    np.random.seed(int(cfg["train"]["seed"]))

    dataset_module = load_module(cfg["dataset"]["script_path"])
    model_module = load_module(cfg["model"]["script_path"])
    optimizer_module = load_module(cfg["optimizer"]["script_path"])
    loss_module = load_module(cfg["loss_function"]["script_path"])
    train_module = load_module(cfg["train"]["script_path"])

    # Load all splits up front so the saved summary records exact tensor shapes.
    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset_module.load_dataset(cfg["dataset"])
    model = model_module.load(x_train.shape[1:], 2, cfg["model"])
    model.compile(
        optimizer=optimizer_module.load(cfg["optimizer"]),
        loss=loss_module.load(),
    )
    train_module.train(
        model,
        x_train,
        y_train,
        x_valid,
        y_valid,
        batch_size=int(cfg["train"]["batch_size"]),
        epochs=int(cfg["train"]["epochs"]),
        patience=int(cfg["train"]["patience"]),
        shuffle=bool(cfg["train"]["shuffle"]),
        artifacts_path=cfg["train"]["artifacts_path"],
    )

    summary = {
        "config_path": str(cfg_path),
        "artifacts_path": cfg["train"]["artifacts_path"],
        "train_shape": list(x_train.shape),
        "valid_shape": list(x_valid.shape),
        "test_shape": list(x_test.shape),
    }
    summary_path = Path(cfg["train"]["artifacts_path"]) / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
