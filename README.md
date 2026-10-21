# Blood Glucose Prediction for SCIM-IoMT

This repository is a cleaned, publication-oriented snapshot of the blood-glucose prediction code used in the SCIM-IoMT project. It contains the core training, quantization-aware training (QAT), and evaluation pipelines needed to reproduce the glucose-prediction part of the workflow.

## What is included

- the original probabilistic LSTM training pipeline
- the compact float model used before QAT
- the fixed-range QAT model used for deployment-oriented experiments
- a two-way comparison workflow for float and effective-INT4 views
- model evaluation code for RMSE and SEG metrics

## Repository layout

- `configs/563_final_experiment.yaml`
  Baseline float-model experiment config.
- `data/OhioT1DM/`
  Expected location of the OhioT1DM XML files (not included, see [Data](#data)).
- `src/datasets/`
  Dataset loading and preprocessing code.
- `src/models/`
  Original probabilistic LSTM model.
- `src/loss_functions/`
  Gaussian negative log-likelihood loss for mean/std prediction.
- `src/optimizers/`
  Optimizer loader.
- `src/train/`
  Training callbacks and checkpoint logic.
- `src/scim_iomt/compact_lstm_model.py`
  Compact float model used before QAT.
- `src/scim_iomt/fixed_quant_qat_model.py`
  Fixed-range activation and effective signed INT4-style weight QAT model.
- `src/scim_iomt/fixed_quant_search_runner.py`
  Shared utilities for compact float pretraining and QAT fine-tuning.
- `src/scim_iomt/compare_utils.py`
  Utility functions for effective signed INT4-style quantization and inference.
- `src/scim_iomt/retrain_compare_float_int4.py`
  Retrain the released compact model and compare float and effective-INT4 views.
- `src/scim_iomt/train_compact_qat_patient.py`
  Fixed-QAT training entry point for the released compact model.
- `scripts/bootstrap.sh`
  One-command environment setup.
- `scripts/train_float.py`
  Train the original float model from a YAML config.
- `scripts/train_qat.py`
  Train the released fixed-QAT compact model.
- `scripts/evaluate_qat.py`
  Evaluate a trained QAT model on a test XML.
- `scripts/retrain_compare.py`
  Retrain the released compact model and export float/INT4 comparison metrics.
- `scripts/run_full_pipeline.sh`
  One-command end-to-end run for the released compact model.

## Security and release audit

This GitHub folder was reviewed before packaging. The current release:

- does not contain API keys, passwords, tokens, or cloud credentials
- does not contain shell injection paths or `subprocess(..., shell=True)` usage
- does not download code from the network at runtime
- does not unpickle arbitrary external objects
- restricts config-driven Python imports to files under the repository `src/` directory

Known trust boundary:

- The scripts intentionally execute local Python modules referenced by the shipped configs. This is normal for experiment repositories, but you should still avoid running modified configs from untrusted sources.

## Data

The OhioT1DM dataset is not redistributed in this repository. It is available on request from the dataset maintainers:
https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html

According to the dataset page:

1. Send an email from your institutional email account to `razvan.bunescu@charlotte.edu` with the subject line `OhioT1DM Request`. Personal email addresses are not accepted.
2. In the body, include your full name, job title, institutional email address, institution name, street address, city, state/province, postal/zip code, and country.
3. Approved requests are answered with a link to a compressed, encrypted copy of the dataset and a password for decompressing it. Processing may take up to one week.

After obtaining the dataset, place the patient-563 files at the paths used by the configs and scripts:

```text
data/OhioT1DM/2018/train/563-ws-training.xml
data/OhioT1DM/2018/test/563-ws-testing.xml
```

Publications that use OhioT1DM data should cite:

C. Marling and R. Bunescu, "The OhioT1DM dataset for blood glucose level prediction: Update 2020," in Proc. 6th International Workshop on Knowledge Discovery from Healthcare Data (KDH), 2020.

## Environment and compatibility

This repository is tested with:

- Python `3.8` or `3.9`
- TensorFlow `2.4.1`

Runtime note:

- the public scripts default to `--device gpu` and will use GPU when TensorFlow can see one
- you can force CPU execution with `--device cpu`
- in newer GPU/CUDA environments, TensorFlow `2.4.1` may still require legacy CUDA/cuDNN compatibility to use the GPU successfully

The dependency list is provided in [requirements.txt](./requirements.txt).

## One-command setup

Create a virtual environment and install dependencies:

```bash
./scripts/bootstrap.sh
```

If you need a specific Python binary, set it explicitly:

```bash
PYTHON_BIN=python3.9 ./scripts/bootstrap.sh
```

After setup:

```bash
source .venv/bin/activate
```

If you prefer not to activate the environment manually, the end-to-end script
below uses `.venv/bin/python` directly after `bootstrap.sh` finishes.

## One-command end-to-end run

For the released patient-specific compact QAT pipeline, after placing the patient-563 XML files as described in [Data](#data):

```bash
./scripts/bootstrap.sh
./scripts/run_full_pipeline.sh 563
```

This runs:

1. fixed-QAT training
2. metric evaluation

Outputs are written under:

```text
outputs/compact_qat_patient_563/
```

The one-command pipeline is scoped to patient 563 in this public release.

## Float and effective-INT4 comparison

The public release also includes a local comparison workflow for two views of the same released compact run:

- `float`: the native float checkpoint evaluated with the original Keras model
- `int4_with_scale`: effective signed quantized weights in the range `[-7, 7]` with per-output-channel scale
- the effective-INT4 view uses INT8-quantized inputs and activations

Important evaluation scope:

- this comparison uses the released `0.6 / 0.2 / 0.2` held-out split from `563-ws-training.xml`
- it is intended for deployment-view comparison inside the released compact pipeline
- it is not the same as evaluating on the external `563-ws-testing.xml`

To retrain and export the comparison metrics:

```bash
python scripts/retrain_compare.py --patient-id 563 --device gpu
```

This writes outputs under:

```text
outputs/compact_qat_patient_563_compare/
```

The main comparison files are:

- `compact_float_int4_comparison.json`
- `compact_float_int4_comparison.csv`
- `compact_float_int4_comparison.md`

## Step-by-step usage

### 1. Train the original float model

```bash
python scripts/train_float.py --config configs/563_final_experiment.yaml
```

This writes float-model artifacts to:

```text
outputs/563_final_experiment/
```

### 2. Train the fixed-QAT compact model

```bash
python scripts/train_qat.py --patient-id 563 --device gpu
```

This writes the released compact-run outputs to:

```text
outputs/compact_qat_patient_563/
```

The main files are:

- `runs/compact_l32_d128_d64/float_model.weights.h5`
- `runs/compact_l32_d128_d64/qat_model.weights.h5`
- `runs/compact_l32_d128_d64/calibration_ranges.json`
- `runs/compact_l32_d128_d64/metrics.json`

### 3. Evaluate QAT model performance

```bash
python scripts/evaluate_qat.py \
  --run-dir outputs/compact_qat_patient_563/runs/compact_l32_d128_d64 \
  --test-xml data/OhioT1DM/2018/test/563-ws-testing.xml \
  --device gpu
```

This writes `evaluation_metrics.json` into the run directory and reports:

- RMSE
- SEG mean / std
- mean predicted standard deviation

## Notes on the released model

- The fixed-QAT path is tied to the released compact configuration `compact_l32_d128_d64`.
- The QAT model uses 8-bit activation quantization and effective signed weights in the range `[-7, 7]`.
- The comparison workflow reconstructs these quantized weights from the QAT checkpoint with per-output-channel scale.
- The `int4_with_scale` view uses INT8-quantized inputs and activations during inference.
- SEG scoring uses the lookup table in `seg.csv`.
- The release focuses on reproducibility of the final compact path, not the entire historical notebook/artifact tree from development.

## License

This packaged release carries the GPL-3.0 license from the original codebase. See [LICENSE](./LICENSE).
