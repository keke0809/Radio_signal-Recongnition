import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from dataloaders.amc_dataset import RadioMLDataset
from models.cnn_transformer import ConformerAMC
from models.loss import build_loss_fn


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path="configs/train_config.yaml"):
    config_full_path = PROJECT_ROOT / config_path
    if not config_full_path.exists():
        print(f"Config file not found: {config_full_path}")
        sys.exit(1)

    with config_full_path.open("r") as f:
        return yaml.safe_load(f)


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA. Run single-process training on CPU instead.")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def seed_everything(seed, rank=0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def current_snr_threshold(config, epoch):
    curriculum = config.get("curriculum", {})
    if not curriculum.get("enabled", True):
        return config["data"].get("snr_threshold", -20)

    schedule = sorted(curriculum.get("schedule", []), key=lambda item: item["epoch"])
    threshold = schedule[0]["min_snr"] if schedule else config["data"].get("snr_threshold", -20)
    for stage in schedule:
        if epoch >= int(stage["epoch"]):
            threshold = stage["min_snr"]
        else:
            break
    return threshold


def make_train_loader(config, full_dataset, train_idx, train_snrs, snr_threshold, is_distributed, rank, world_size):
    data_cfg = config["data"]
    valid_mask = train_snrs >= snr_threshold
    filtered_idx = train_idx[valid_mask]
    subset = Subset(full_dataset, filtered_idx)
    sampler = None
    shuffle = True
    if is_distributed:
        sampler = DistributedSampler(
            subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False

    loader = DataLoader(
        subset,
        batch_size=data_cfg.get("batch_size_per_gpu", data_cfg.get("batch_size", 4096)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=torch.cuda.is_available(),
        drop_last=is_distributed,
        persistent_workers=data_cfg.get("num_workers", 8) > 0,
    )
    return loader, sampler, len(subset)


def make_eval_loader(config, full_dataset, val_idx):
    data_cfg = config["data"]
    subset = Subset(full_dataset, val_idx)
    return DataLoader(
        subset,
        batch_size=data_cfg.get("eval_batch_size", data_cfg.get("batch_size", 4096)),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data_cfg.get("num_workers", 8) > 0,
    )


def reduce_sum(tensor, device, is_distributed):
    value = tensor.detach().to(device)
    if is_distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def train_one_epoch(model, train_loader, train_sampler, criterion, optimizer, scaler, device, config, epoch, rank, is_distributed):
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    grad_clip = config["train"].get("grad_clip", 1.0)
    running_loss = torch.zeros(1, device=device)
    seen = torch.zeros(1, device=device)
    interval_loss = 0.0
    interval_seen = 0
    log_interval = max(1, len(train_loader) // 5)

    for step, (signals, labels, _) in enumerate(train_loader, start=1):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(signals)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.detach() * batch_size
        seen += batch_size
        interval_loss += loss.item() * batch_size
        interval_seen += batch_size
        should_log = step % log_interval == 0 or step == len(train_loader)
        if is_main_process(rank) and should_log:
            current_lr = optimizer.param_groups[0]["lr"]
            avg_interval_loss = interval_loss / max(interval_seen, 1)
            print(
                f"Epoch {epoch + 1} | iter {step}/{len(train_loader)} | "
                f"loss={avg_interval_loss:.4f} | lr={current_lr:.6g}"
            )
            interval_loss = 0.0
            interval_seen = 0

    total_loss = reduce_sum(running_loss, device, is_distributed)
    total_seen = reduce_sum(seen, device, is_distributed).clamp_min(1)
    return (total_loss / total_seen).item()


@torch.no_grad()
def evaluate(model, eval_loader, criterion, device, config):
    model.eval()
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for signals, labels, _ in eval_loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(signals)
            loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += preds.eq(labels).sum().item()
        total_seen += batch_size

    total_seen = max(total_seen, 1)
    return {
        "val_loss": total_loss / total_seen,
        "val_acc": total_correct / total_seen * 100.0,
    }


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
        },
        save_path,
    )


def export_onnx(model, config, device, save_dir):
    if not config["train"].get("export_onnx_after_train", True):
        return

    model_to_export = unwrap_model(model).eval()
    dummy_input = torch.randn(
        1,
        config["model"].get("input_channels", 2),
        config["data"].get("input_length", 1024),
        device=device,
    )
    onnx_path = save_dir / "conformer_amc.onnx"
    try:
        torch.onnx.export(
            model_to_export,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=config["train"].get("onnx_opset", 14),
            do_constant_folding=True,
            input_names=["input_signal"],
            output_names=["logits"],
            dynamic_axes={"input_signal": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        print(f"ONNX exported to: {onnx_path}")
    except Exception as exc:
        print(f"ONNX export failed: {exc}")


def build_scheduler(config, optimizer):
    train_cfg = config["train"]
    epochs = int(train_cfg["epochs"])
    warmup_epochs = min(int(train_cfg.get("warmup_epochs", 5)), max(epochs - 1, 1))
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup_epochs, 1),
        eta_min=float(train_cfg.get("min_lr", 0.0)),
    )
    return optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def load_split_array(split_dir, name):
    path = PROJECT_ROOT / split_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required split file not found: {path}")
    return np.load(path)


def train():
    config = load_config()
    is_distributed, rank, local_rank, world_size = setup_distributed()
    try:
        seed_everything(config.get("experiment", {}).get("seed", 42), rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if is_main_process(rank):
            print(f"Training on {device}; distributed={is_distributed}; world_size={world_size}")

        split_dir = config["data"].get("split_dir", "data_splits")
        train_idx = load_split_array(split_dir, "train_indices.npy")
        train_snrs = load_split_array(split_dir, "train_snrs.npy")
        val_idx = load_split_array(split_dir, "val_indices.npy")
        full_dataset = RadioMLDataset(config["data"]["file_path"])

        model = build_model(config).to(device)
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config["train"]["lr"]),
            weight_decay=float(config["train"].get("weight_decay", 0.0)),
        )
        scheduler = build_scheduler(config, optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["train"].get("amp", True))
        criterion = build_loss_fn(
            loss_type=config.get("loss", {}).get("type", "ce"),
            smoothing=float(config.get("loss", {}).get("smoothing", 0.1)),
        )

        save_dir = checkpoint_dir(config)
        if is_main_process(rank):
            save_dir.mkdir(parents=True, exist_ok=True)
        if is_distributed:
            dist.barrier()

        best_acc = -1.0
        current_threshold = None
        train_loader = None
        train_sampler = None
        eval_loader = make_eval_loader(config, full_dataset, val_idx) if is_main_process(rank) else None

        for epoch in range(int(config["train"]["epochs"])):
            snr_threshold = current_snr_threshold(config, epoch)
            if snr_threshold != current_threshold or train_loader is None:
                current_threshold = snr_threshold
                train_loader, train_sampler, subset_size = make_train_loader(
                    config,
                    full_dataset,
                    train_idx,
                    train_snrs,
                    current_threshold,
                    is_distributed,
                    rank,
                    world_size,
                )
                if is_main_process(rank):
                    print(f"Using train SNR >= {current_threshold} dB; samples={subset_size}")

            train_loss = train_one_epoch(
                model,
                train_loader,
                train_sampler,
                criterion,
                optimizer,
                scaler,
                device,
                config,
                epoch,
                rank,
                is_distributed,
            )
            scheduler.step()

            metrics = {"train_loss": train_loss, "snr_threshold": current_threshold}
            should_eval = (epoch + 1) % int(config["train"].get("eval_interval", 1)) == 0
            if is_main_process(rank) and should_eval:
                metrics.update(evaluate(unwrap_model(model), eval_loader, criterion, device, config))
                print(
                    f"Epoch {epoch + 1}: train_loss={metrics['train_loss']:.4f}, "
                    f"val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_acc']:.2f}%"
                )
                save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_dir / "latest_model.pth")
                if metrics["val_acc"] > best_acc:
                    best_acc = metrics["val_acc"]
                    save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_dir / "best_model.pth")
                    print(f"New best checkpoint saved: val_acc={best_acc:.2f}%")
            elif is_main_process(rank):
                print(f"Epoch {epoch + 1}: train_loss={metrics['train_loss']:.4f}")
                save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_dir / "latest_model.pth")

            if is_distributed:
                dist.barrier()

        if is_main_process(rank):
            export_onnx(model, config, device, save_dir)
            print(f"Training complete. Checkpoints saved in: {save_dir}")
    finally:
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    train()