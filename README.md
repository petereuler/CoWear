# CoWear

CoWear is a Python implementation for multi-device inertial localization. It
provides shared data loading, preprocessing, model training, inference, and
numerical evaluation for phone, watch, and glasses IMU streams.

The dataset is distributed separately at
[Hugging Face](https://huggingface.co/datasets/zyshe/CoWear). This repository
contains no sensor data, checkpoints, caches, trajectory exports, or figures.

## Install

Use Python 3.10 or newer:

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

Install a CUDA-enabled PyTorch build appropriate for the target GPU before
running training or full evaluation.

## Commands

All runtime paths are explicit:

```bash
python -m cowear validate \
  --data-root /path/to/data \
  --split-index splits/paper_v1_session_split.csv \
  --config configs/paper_v1.toml \
  --output-dir /path/to/output
```

```bash
python -m cowear train \
  --data-root /path/to/data/processed \
  --split-index splits/paper_v1_session_split.csv \
  --target-mode phone \
  --device cuda \
  --output-dir /path/to/output
```

```bash
python -m cowear evaluate \
  --data-root /path/to/data/processed \
  --split-index splits/paper_v1_session_split.csv \
  --weights-root /path/to/weights \
  --device cuda \
  --output-dir /path/to/output
```

Evaluation produces numerical CSV and JSON metrics only.

For published-weight evaluation, extract all three `v1.0.1` Release archives
into the same `--weights-root`. This supplies the CoWear checkpoints, baseline
parameters, and published per-session baseline metrics required by the command.

## Layout

```text
src/cowear/
  cli.py               command-line entry points
  protocol/            data, geometry, metrics, and checkpoint contracts
  models/              neural model definitions
  baselines/           PDR, RIDI, RoNIN, and TLIO implementations
  evaluation/          common numerical evaluation
  training/            shared optimization loop
  _internal/           reusable preprocessing and data adapters
configs/               runtime configuration
splits/                session split index
scripts/               command wrappers and release helpers
tests/                 contract and mathematical tests
```

## License

Code is Apache-2.0. Dataset access and terms are defined by its Hugging Face
dataset card.
