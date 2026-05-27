import importlib
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.amc_dataset import RadioMLDataset
from models.cnn_transformer import ConformerAMC


def test_model_forward_shape():
    model = ConformerAMC(num_classes=24, d_model=32, nhead=4, num_layers=1, dropout=0.1)
    x = torch.randn(2, 2, 1024)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 24)


def test_dataset_output_shape(tmp_path):
    h5_path = tmp_path / "mini_radioml.hdf5"
    y = np.zeros((3, 24), dtype=np.float32)
    y[np.arange(3), [0, 1, 2]] = 1.0
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=np.random.randn(3, 1024, 2).astype(np.float32))
        f.create_dataset("Y", data=y)
        f.create_dataset("Z", data=np.array([[-20.0], [0.0], [20.0]], dtype=np.float32))

    dataset = RadioMLDataset(str(h5_path))
    signal, label, snr = dataset[0]

    assert signal.shape == (2, 1024)
    assert label.dtype == torch.long
    assert snr.dtype == torch.float32


def test_train_module_imports_without_ddp_init():
    module = importlib.import_module("train_smooth")
    assert module.setup_distributed()[0] is False
