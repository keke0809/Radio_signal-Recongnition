import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path="configs/train_config.yaml"):
    config_full_path = PROJECT_ROOT / config_path
    if not config_full_path.exists():
        print(f"Config file not found: {config_full_path}")
        sys.exit(1)

    with config_full_path.open("r") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(labels_path, preds_path, save_dir):
    y_true = np.load(labels_path)
    y_pred = np.load(preds_path)
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    save_path = save_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_acc_vs_snr(labels_path, preds_path, snrs_path, save_dir):
    y_true = np.load(labels_path)
    y_pred = np.load(preds_path)
    snrs = np.load(snrs_path)

    total_acc = np.mean(y_true == y_pred) * 100
    print("\n" + "=" * 50)
    print(f"Overall Accuracy: {total_acc:.2f}%")
    print("=" * 50)

    unique_snrs = np.sort(np.unique(snrs))
    accuracies = []
    print(f"{'SNR (dB)':<10} | {'Samples':<10} | {'Accuracy':<10}")
    print("-" * 38)
    for snr in unique_snrs:
        idx = np.where(snrs == snr)[0]
        acc = np.mean(y_true[idx] == y_pred[idx]) * 100
        accuracies.append(acc)
        print(f"{snr:<10} | {len(idx):<10} | {acc:.2f}%")
    print("-" * 38)

    plt.figure(figsize=(10, 6))
    plt.plot(unique_snrs, accuracies, marker="o", linestyle="-", linewidth=2.5, markersize=8)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title("Classification Accuracy vs. Signal-to-Noise Ratio (SNR)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.xticks(unique_snrs)
    if accuracies:
        plt.ylim(max(0, min(accuracies) - 5), min(105, max(accuracies) + 5))
    plt.tight_layout()

    save_path = save_dir / "acc_vs_snr.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Accuracy-vs-SNR curve saved to: {save_path}")


def main():
    config = load_config()
    experiment_name = config.get("experiment", {}).get("name", "ddp_amc_v2")
    result_dir = PROJECT_ROOT / "results" / experiment_name
    save_dir = PROJECT_ROOT / "values" / experiment_name / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)

    labels_file = result_dir / "test_labels.npy"
    preds_file = result_dir / "test_preds.npy"
    snrs_file = result_dir / "test_snrs.npy"
    missing_files = [str(path) for path in (labels_file, preds_file, snrs_file) if not path.exists()]
    if missing_files:
        print("Missing result files:")
        for path in missing_files:
            print(f"- {path}")
        print("Run `python test.py` before visualization.")
        return

    plot_confusion_matrix(labels_file, preds_file, save_dir)
    plot_acc_vs_snr(labels_file, preds_file, snrs_file, save_dir)
    print(f"Visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()
