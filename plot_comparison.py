import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Paths to CSVs
    yolov8s_csv = Path("fruit_detection_runs_backup/train/results.csv")
    yolo26s_csv = Path("fruit_detection_runs_yolo26s/train/results.csv")

    # Load data
    df8 = pd.read_csv(yolov8s_csv)
    df26 = pd.read_csv(yolo26s_csv)

    # Clean column names (strip whitespace)
    df8.columns = df8.columns.str.strip()
    df26.columns = df26.columns.str.strip()

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Precision & Recall
    axs[0, 0].plot(df8["epoch"], df8["metrics/precision(B)"], 'b--', label="YOLOv8s Precision")
    axs[0, 0].plot(df8["epoch"], df8["metrics/recall(B)"], 'b-', label="YOLOv8s Recall")
    axs[0, 0].plot(df26["epoch"], df26["metrics/precision(B)"], 'r--', label="YOLO26s Precision")
    axs[0, 0].plot(df26["epoch"], df26["metrics/recall(B)"], 'r-', label="YOLO26s Recall")
    axs[0, 0].set_title("Precision & Recall Over Epochs")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Score")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Subplot 2: mAP50 & mAP50-95
    axs[0, 1].plot(df8["epoch"], df8["metrics/mAP50(B)"], 'b--', label="YOLOv8s mAP50")
    axs[0, 1].plot(df8["epoch"], df8["metrics/mAP50-95(B)"], 'b-', label="YOLOv8s mAP50-95")
    axs[0, 1].plot(df26["epoch"], df26["metrics/mAP50(B)"], 'r--', label="YOLO26s mAP50")
    axs[0, 1].plot(df26["epoch"], df26["metrics/mAP50-95(B)"], 'r-', label="YOLO26s mAP50-95")
    axs[0, 1].set_title("mAP Accuracy Over Epochs")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("mAP Score")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Subplot 3: Box Loss (Train vs Val)
    axs[1, 0].plot(df8["epoch"], df8["train/box_loss"], 'b--', label="YOLOv8s Train Box Loss")
    axs[1, 0].plot(df8["epoch"], df8["val/box_loss"], 'b-', label="YOLOv8s Val Box Loss")
    axs[1, 0].plot(df26["epoch"], df26["train/box_loss"], 'r--', label="YOLO26s Train Box Loss")
    axs[1, 0].plot(df26["epoch"], df26["val/box_loss"], 'r-', label="YOLO26s Val Box Loss")
    axs[1, 0].set_title("Bounding Box Loss Over Epochs")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Subplot 4: Cls Loss (Train vs Val)
    axs[1, 1].plot(df8["epoch"], df8["train/cls_loss"], 'b--', label="YOLOv8s Train Cls Loss")
    axs[1, 1].plot(df8["epoch"], df8["val/cls_loss"], 'b-', label="YOLOv8s Val Cls Loss")
    axs[1, 1].plot(df26["epoch"], df26["train/cls_loss"], 'r--', label="YOLO26s Train Cls Loss")
    axs[1, 1].plot(df26["epoch"], df26["val/cls_loss"], 'r-', label="YOLO26s Val Cls Loss")
    axs[1, 1].set_title("Classification Loss Over Epochs")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Loss")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("yolov8s_vs_yolo26s_comparison.png", dpi=300)
    print("Comparison plot saved as yolov8s_vs_yolo26s_comparison.png")

if __name__ == "__main__":
    main()
