# Conformer-AMC: Automatic Modulation Classification

作者：张治柯 | 北京交通大学  
Author: Zhike Zhang | Beijing Jiaotong University

This project trains an automatic modulation classification model on the RadioML 2018 HDF5 dataset. The current pipeline uses a stronger CNN-Conformer style backbone, SNR curriculum learning, validation-driven checkpointing, AMP, and dual-GPU DistributedDataParallel (DDP).

## Key Features

- Strong AMC model: multi-scale 1D CNN stem, residual SE blocks, Conformer-style attention/convolution blocks, and attention pooling.
- Dual-GPU DDP training: launch with `torchrun --nproc_per_node=2 train_smooth.py`.
- Config-driven curriculum: SNR thresholds are defined in `configs/train_config.yaml`.
- Reliable checkpoints: saves `latest_model.pth` every evaluated epoch and `best_model.pth` by validation accuracy.
- Unified evaluation: `test.py`, `utils/visualize.py`, and `export_onnx.py` read the same experiment/checkpoint settings.
- Production export: ONNX export keeps dynamic batch axes for deployment.

## Project Structure

```text
RadioML/
├── configs/
│   └── train_config.yaml      # Data, model, training, curriculum, checkpoint config
├── dataloaders/
│   └── amc_dataset.py         # HDF5 lazy-loading PyTorch Dataset
├── dataset/
│   └── split_data.py          # Generates train/val split indices
├── models/
│   ├── cnn_transformer.py     # ConformerAMC model
│   └── loss.py                # CE / Focal loss factory
├── tests/
│   └── test_smoke.py          # Lightweight shape/import tests
├── utils/
│   └── visualize.py           # Confusion matrix and Accuracy-vs-SNR plots
├── train_smooth.py            # Main DDP training entry
├── test.py                    # Evaluation on val/test split
└── export_onnx.py             # Standalone ONNX export
```

## Configuration

Edit `configs/train_config.yaml` before running:

- `experiment.name`: controls output folders under `checkpoints/`, `results/`, and `values/`.
- `data.file_path`: RadioML HDF5 file path.
- `data.split_dir`: directory containing `train_indices.npy`, `train_snrs.npy`, and `val_indices.npy`.
- `data.batch_size_per_gpu`: per-process DDP batch size. With two GPUs, the effective batch size is `2 * batch_size_per_gpu`.
- `model.*`: model width/depth and class count.
- `curriculum.schedule`: epoch-to-minimum-SNR schedule.
- `train.save_dir`: checkpoint output directory.

## Quick Start

### 1. Generate Splits

```bash
python dataset/split_data.py
```

This creates `data_splits/train_indices.npy`, `data_splits/train_snrs.npy`, and `data_splits/val_indices.npy`.

### 2. Train With Two GPUs

```bash
torchrun --nproc_per_node=2 train_smooth.py
```

For a single-process smoke run, you can still use:

```bash
python train_smooth.py
```

Checkpoints are saved to `train.save_dir`, defaulting to `checkpoints/ddp_amc_v2`.

### 3. Evaluate

```bash
python test.py
```

The script loads `best_model.pth` first, falls back to `latest_model.pth`, and saves arrays to `results/<experiment.name>/`.

### 4. Visualize

```bash
python utils/visualize.py
```

Plots are saved to `values/<experiment.name>/visualizations/`.

### 5. Export ONNX

```bash
python export_onnx.py
```

The exported model is saved as `conformer_amc.onnx` in the configured checkpoint directory.

## Tests

Run the lightweight smoke tests:

```bash
python -m pytest tests/test_smoke.py
```

These tests verify model forward shape, mini HDF5 dataset loading, and safe import of the DDP training module.

## Repository Hygiene

The repository keeps source code, configuration, tests, and documentation under version control. Large or generated artifacts are ignored by `.gitignore`, including:

- RadioML HDF5 data and generated `data_splits/`
- checkpoints, ONNX exports, and model weights
- test predictions under `results/`
- generated plots/logs under `values/`
- Python caches and pytest caches

Before committing, check the working tree with:

```bash
git status --short
```

Commit only source, config, tests, and docs. Keep trained checkpoints and generated visualizations outside git.

## Dependencies

- Python 3.8+
- PyTorch 2.0+ with CUDA for DDP training
- NumPy, h5py, PyYAML, tqdm
- Matplotlib, Seaborn, scikit-learn
- pytest for smoke tests
