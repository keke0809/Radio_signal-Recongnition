import sys
from collections import OrderedDict
from pathlib import Path

import torch
import yaml

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


def export_to_onnx():
    config = load_config()
    device = torch.device("cpu")
    model = build_model(config).to(device)

    ckpt_dir = checkpoint_dir(config)
    checkpoint_path = ckpt_dir / "best_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = ckpt_dir / "latest_model.pth"
    if not checkpoint_path.exists():
        print(f"Export failed: no checkpoint found in {ckpt_dir}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    load_model_weights(model, checkpoint_path, device)
    model.eval()

    dummy_input = torch.randn(
        1,
        config["model"].get("input_channels", 2),
        config["data"].get("input_length", 1024),
        device=device,
    )
    onnx_path = ckpt_dir / "conformer_amc.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=config["train"].get("onnx_opset", 14),
            do_constant_folding=True,
            input_names=["input_signal"],
            output_names=["logits"],
            dynamic_axes={"input_signal": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"ONNX exported to: {onnx_path}")
        print(f"ONNX size: {size_mb:.2f} MB")
    except Exception as exc:
        print(f"ONNX export failed: {exc}")


if __name__ == "__main__":
    export_to_onnx()
