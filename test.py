import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloaders.amc_dataset import RadioMLDataset
from models.cnn_transformer import ConformerAMC


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path="configs/train_config.yaml"):
    config_full_path = PROJECT_ROOT / config_path
    if not config_full_path.exists():
        print(f"Config file not found: {config_full_path}")
        sys.exit(1)

    with config_full_path.open("r") as f:
        return yaml.safe_load(f)


def build_model(config):
    model_cfg = config["model"]
    data_cfg = config["data"]
    return ConformerAMC(
        num_classes=model_cfg["num_classes"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        input_channels=model_cfg.get("input_channels", 2),
        input_length=data_cfg.get("input_length", 1024),
        dropout=model_cfg.get("dropout", 0.15),
    )


def checkpoint_dir(config):
    save_dir = config["train"].get("save_dir")
    if save_dir:
        return PROJECT_ROOT / save_dir

    experiment_name = config.get("experiment", {}).get("name", "ddp_amc_v2")
    return PROJECT_ROOT / "checkpoints" / experiment_name


def load_model_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value
    model.load_state_dict(cleaned_state_dict)
    return checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}


def test():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    split_dir = PROJECT_ROOT / config["data"].get("split_dir", "data_splits")
    test_idx_path = split_dir / "val_indices.npy"
    if not test_idx_path.exists():
        print(f"Test index file not found: {test_idx_path}")
        return

    test_idx = np.load(test_idx_path)
    full_dataset = RadioMLDataset(config["data"]["file_path"])
    test_subset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(
        test_subset,
        batch_size=config["data"].get("eval_batch_size", config["data"].get("batch_size", 4096)),
        shuffle=False,
        num_workers=config["data"].get("num_workers", 8),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["data"].get("num_workers", 8) > 0,
    )

    model = build_model(config).to(device)
    ckpt_dir = checkpoint_dir(config)
    checkpoint_path = ckpt_dir / "best_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = ckpt_dir / "latest_model.pth"
    if not checkpoint_path.exists():
        print(f"No checkpoint found in: {ckpt_dir}")
        return

    metrics = load_model_weights(model, checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if metrics:
        print(f"Checkpoint metrics: {metrics}")
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_snrs = []
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"

    with torch.no_grad():
        for signals, labels, snrs in tqdm(test_loader, desc="Testing"):
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(signals)

            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
            all_snrs.append(snrs.cpu())

    acc = 100.0 * correct / max(total, 1)
    print("\n" + "=" * 50)
    print(f"Overall validation/test accuracy: {acc:.2f}%")
    print("=" * 50 + "\n")

    experiment_name = config.get("experiment", {}).get("name", "ddp_amc_v2")
    result_dir = PROJECT_ROOT / "results" / experiment_name
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "test_preds.npy", torch.cat(all_preds).numpy())
    np.save(result_dir / "test_labels.npy", torch.cat(all_labels).numpy())
    np.save(result_dir / "test_snrs.npy", torch.cat(all_snrs).numpy())
    print(f"Prediction arrays saved to: {result_dir}")


if __name__ == "__main__":
    test()
