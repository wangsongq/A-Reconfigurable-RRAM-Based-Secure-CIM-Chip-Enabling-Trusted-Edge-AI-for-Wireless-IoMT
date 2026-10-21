#!/usr/bin/env python3
"""Shared utilities for float/INT4 comparison of the released compact model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QMAX = 7.0


def load_module(script_path: str):
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset_from_metrics(metrics_payload: dict):
    """Load the same held-out split used by the released compact run."""
    ds_mod = load_module(str(PROJECT_ROOT / "src" / "datasets" / "ohio.py"))
    cfg = {
        "xml_path": metrics_payload["dataset"]["xml_path"],
        "nb_past_steps": 12,
        "nb_future_steps": 1,
        "train_fraction": metrics_payload["dataset"]["split"]["train_fraction"],
        "valid_fraction": metrics_payload["dataset"]["split"]["valid_fraction"],
        "test_fraction": metrics_payload["dataset"]["split"]["test_fraction"],
        "scale": 0.01,
    }
    return ds_mod.load_dataset(cfg)


def quantize_unsigned_dequant(x: np.ndarray, min_value: float, max_value: float, num_bits: int = 8) -> np.ndarray:
    qmax = float((1 << num_bits) - 1)
    scale = max((max_value - min_value) / qmax, 1e-8)
    clipped = np.clip(x.astype(np.float32), min_value, max_value)
    q = np.rint((clipped - min_value) / scale)
    q = np.clip(q, 0.0, qmax)
    return (q * scale + min_value).astype(np.float32)


def quantize_signed_dequant(x: np.ndarray, range_abs: float, num_bits: int = 8) -> np.ndarray:
    qmax = float((1 << (num_bits - 1)) - 1)
    scale = max(range_abs / qmax, 1e-8)
    clipped = np.clip(x.astype(np.float32), -range_abs, range_abs)
    q = np.rint(clipped / scale)
    q = np.clip(q, -qmax, qmax)
    return (q * scale).astype(np.float32)


def quantize_weight_per_output_channel(weights: np.ndarray):
    """Map weights to the effective signed range [-7, 7] with per-output-channel scale."""
    weights = weights.astype(np.float32)
    max_abs = np.max(np.abs(weights), axis=0, keepdims=True)
    max_abs = np.where(max_abs == 0.0, 1.0, max_abs).astype(np.float32)
    scale = max_abs / QMAX
    q_signed = np.rint(np.clip(weights, -max_abs, max_abs) / scale).astype(np.int16)
    q_signed = np.clip(q_signed, -QMAX, QMAX).astype(np.int8)
    w_dequant = (q_signed.astype(np.float32) * scale).astype(np.float32)
    return q_signed, scale.reshape(-1).astype(np.float32), w_dequant


def _read_first_existing(h5_file: h5py.File, candidates: list[str]) -> np.ndarray:
    for candidate in candidates:
        if candidate in h5_file:
            return h5_file[candidate][()].astype(np.float32)
    raise KeyError(f"None of the candidate datasets were found: {candidates}")


def build_qat_quantized_weights(qat_weights_path: Path):
    """Load a QAT checkpoint and rebuild the effective signed quantized weights."""
    with h5py.File(qat_weights_path, "r") as f:
        raw = {
            "lstm_kernel": _read_first_existing(
                f,
                ["/qat_lstm/qat_lstm/qat_lstm_cell/kernel:0", "/layers/rnn/cell/vars/0"],
            ),
            "lstm_recurrent": _read_first_existing(
                f,
                ["/qat_lstm/qat_lstm/qat_lstm_cell/recurrent_kernel:0", "/layers/rnn/cell/vars/1"],
            ),
            "lstm_bias": _read_first_existing(
                f,
                ["/qat_lstm/qat_lstm/qat_lstm_cell/bias:0", "/layers/rnn/cell/vars/2"],
            ),
            "dense1_kernel": _read_first_existing(
                f,
                ["/qat_dense1/qat_dense1/kernel:0", "/layers/fixed_quant_dense/vars/0"],
            ),
            "dense1_bias": _read_first_existing(
                f,
                ["/qat_dense1/qat_dense1/bias:0", "/layers/fixed_quant_dense/vars/1"],
            ),
            "dense2_kernel": _read_first_existing(
                f,
                ["/qat_dense2/qat_dense2/kernel:0", "/layers/fixed_quant_dense_1/vars/0"],
            ),
            "dense2_bias": _read_first_existing(
                f,
                ["/qat_dense2/qat_dense2/bias:0", "/layers/fixed_quant_dense_1/vars/1"],
            ),
            "mu_kernel": _read_first_existing(
                f,
                ["/qat_mu_head/qat_mu_head/kernel:0", "/layers/fixed_quant_dense_3/vars/0"],
            ),
            "mu_bias": _read_first_existing(
                f,
                ["/qat_mu_head/qat_mu_head/bias:0", "/layers/fixed_quant_dense_3/vars/1"],
            ),
            "std_kernel": _read_first_existing(
                f,
                ["/qat_std_head/qat_std_head/kernel:0", "/layers/fixed_quant_dense_2/vars/0"],
            ),
            "std_bias": _read_first_existing(
                f,
                ["/qat_std_head/qat_std_head/bias:0", "/layers/fixed_quant_dense_2/vars/1"],
            ),
        }

    quant = {}
    metadata = {}
    for key in ["lstm_kernel", "lstm_recurrent", "dense1_kernel", "dense2_kernel", "mu_kernel", "std_kernel"]:
        q_signed, scale, w_dequant = quantize_weight_per_output_channel(raw[key])
        quant[key] = w_dequant
        metadata[key] = {"q_signed": q_signed, "scale": scale}
    for key in ["lstm_bias", "dense1_bias", "dense2_bias", "mu_bias", "std_bias"]:
        quant[key] = raw[key]
    return quant, metadata


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def forward_quantized(x: np.ndarray, weights: dict, calib: dict, hidden_units: int) -> tuple[np.ndarray, np.ndarray]:
    batch = x.shape[0]
    h = np.zeros((batch, hidden_units), dtype=np.float32)
    c = np.zeros((batch, hidden_units), dtype=np.float32)
    for t in range(x.shape[1]):
        xt = quantize_unsigned_dequant(x[:, t, :], calib["input_min"], calib["input_max"])
        h_q = quantize_signed_dequant(h, calib["hidden_abs"])
        z = xt @ weights["lstm_kernel"] + h_q @ weights["lstm_recurrent"] + weights["lstm_bias"]
        i, f, g, o = np.split(z, 4, axis=1)
        i = sigmoid(i)
        f = sigmoid(f)
        g = np.tanh(g).astype(np.float32)
        o = sigmoid(o)
        c = f * c + i * g
        h = quantize_signed_dequant(o * np.tanh(c).astype(np.float32), calib["hidden_abs"])
    x1 = np.maximum(h @ weights["dense1_kernel"] + weights["dense1_bias"], 0.0).astype(np.float32)
    x1_q = quantize_unsigned_dequant(x1, 0.0, calib["relu1_max"])
    x2 = np.maximum(x1_q @ weights["dense2_kernel"] + weights["dense2_bias"], 0.0).astype(np.float32)
    x2_q = quantize_unsigned_dequant(x2, 0.0, calib["relu2_max"])
    mu = (x2_q @ weights["mu_kernel"] + weights["mu_bias"]).reshape(-1).astype(np.float32)
    std = np.exp(x2_q @ weights["std_kernel"] + weights["std_bias"]).reshape(-1).astype(np.float32)
    return mu, std
