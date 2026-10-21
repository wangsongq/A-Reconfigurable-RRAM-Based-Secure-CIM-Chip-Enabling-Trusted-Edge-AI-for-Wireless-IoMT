#!/usr/bin/env python3
"""Train the released compact fixed-QAT model for a specified Ohio patient."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scim_iomt import fixed_quant_search_runner as fsr


def main():
    """Train the fixed-QAT compact model for one OhioT1DM patient."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--float-epochs", type=int, default=None)
    parser.add_argument("--float-patience", type=int, default=None)
    parser.add_argument("--qat-epochs", type=int, default=300)
    parser.add_argument("--qat-patience", type=int, default=50)
    parser.add_argument("--qat-learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    resolved_device = fsr.configure_tensorflow_device(args.device)

    patient_id = str(args.patient_id)
    train_xml = (
        fsr.PROJECT_ROOT / "data" / "OhioT1DM" / "2018" / "train" / f"{patient_id}-ws-training.xml"
    ).resolve()
    if not train_xml.exists():
        raise FileNotFoundError(f"Training XML not found: {train_xml}")

    results_root = fsr.PROJECT_ROOT / "outputs" / f"compact_qat_patient_{patient_id}"
    results_root.mkdir(parents=True, exist_ok=True)
    fsr.RESULTS_ROOT = results_root

    base_cfg = fsr.load_base_cfg()
    base_cfg["dataset"]["xml_path"] = str(train_xml)
    base_cfg["dataset"]["train_fraction"] = 0.6
    base_cfg["dataset"]["valid_fraction"] = 0.2
    base_cfg["dataset"]["test_fraction"] = 0.2
    base_cfg["dataset"]["nb_future_steps"] = 1

    dataset_module = fsr.load_module(base_cfg["dataset"]["script_path"])
    dataset_bundle = dataset_module.load_dataset(base_cfg["dataset"])

    # This release exposes the exact compact configuration used in the paper
    # rather than the whole architecture search matrix.
    specs = [spec for spec in fsr.build_experiment_grid() if spec.run_name == "compact_l32_d128_d64"]
    if not specs:
        raise RuntimeError("Could not find the released compact spec in the search grid.")
    spec = specs[0]

    class Args:
        pass

    runner_args = Args()
    runner_args.float_epochs = args.float_epochs
    runner_args.float_patience = args.float_patience
    runner_args.qat_epochs = args.qat_epochs
    runner_args.qat_patience = args.qat_patience
    runner_args.qat_learning_rate = args.qat_learning_rate
    runner_args.skip_existing = args.skip_existing

    payload = fsr.run_one_experiment(spec, base_cfg, dataset_bundle, runner_args)
    csv_path, rows = fsr.write_results([payload])
    frontier = fsr.compute_pareto(rows)
    pareto_path = results_root / "pareto_front.csv"
    import csv as _csv

    with pareto_path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(frontier[0].keys()))
        writer.writeheader()
        writer.writerows(frontier)
    report_path = fsr.write_report(rows, frontier)

    print(f"results_root={results_root}")
    print(f"device={resolved_device}")
    print(f"metrics={results_root / 'runs' / spec.run_name / 'metrics.json'}")
    print(f"results_csv={csv_path}")
    print(f"pareto_csv={pareto_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
