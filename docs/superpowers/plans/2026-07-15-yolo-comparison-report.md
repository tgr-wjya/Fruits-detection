# YOLOv8s vs YOLO26s Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a comparison plotting script and update the final training report to present a side-by-side performance evaluation of YOLOv8s vs YOLO26s.

**Architecture:** Create `plot_comparison.py` to parse results CSV files for both models and plot overlays of loss, precision, recall, and mAP metrics. Rewrite `final_training_report.md` to display tabular data, architectural comparisons, and embedded plot visualizations.

**Tech Stack:** Python 3.10+, Pandas, Matplotlib, Markdown.

## Global Constraints

*   NO EMOJIS EVER: Do not write or include emojis in the code changes, print messages, comments, configurations, logs, or documentation.
*   Prefix all CLI/shell executions with `rtk` (Rust Token Killer) to minimize context token overhead.

---

### Task 1: Create the Comparison Plotting Script

**Files:**
*   Create: `plot_comparison.py`

**Interfaces:**
*   Consumes:
    *   YOLOv8s: `fruit_detection_runs_backup/train/results.csv`
    *   YOLO26s: `fruit_detection_runs_yolo26s/train/results.csv`
*   Produces:
    *   Image file: `yolov8s_vs_yolo26s_comparison.png`

*   [ ] **Step 1: Write the comparison plotting script**
    Create the [plot_comparison.py](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/plot_comparison.py) script in the workspace root:

    ```python
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
    ```

*   [ ] **Step 2: Execute the plotting script**
    Run the script:
    `rtk python plot_comparison.py`
    Expected output: "Comparison plot saved as yolov8s_vs_yolo26s_comparison.png" and verification that the file `yolov8s_vs_yolo26s_comparison.png` exists in the root directory.

*   [ ] **Step 3: Commit Task 1 changes**
    Notify the user before running git operations:
    ```bash
    rtk git add plot_comparison.py
    rtk git commit -m "feat: add script to generate comparative training metric plots"
    ```

---

### Task 2: Update the Final Training Report

**Files:**
*   Modify: `final_training_report.md`

**Interfaces:**
*   Consumes:
    *   YOLOv8s evaluation metrics.
    *   YOLO26s validation metrics.
    *   Plot image `yolov8s_vs_yolo26s_comparison.png`.
*   Produces:
    *   Updated report `final_training_report.md`.

*   [ ] **Step 1: Update the report**
    Overwrite [final_training_report.md](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/final_training_report.md) with the new comparison structure:

    ```markdown
    # YOLOv8s vs YOLO26s Fruit Detection Model Training Report

    This report documents the final evaluation metrics, architectural trade-offs, and performance analysis comparing the original YOLOv8s model and the upgraded YOLO26s model on the target fruits detection dataset (evaluated on the validation set containing 914 images with 3227 labeled fruit instances).

    ## 1. Executive Summary & Overall Metrics

    The table below compares the overall performance metrics after completing 50 training epochs:

    | Model | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
    | :--- | :---: | :---: | :---: | :---: |
    | **YOLOv8s** | 59.8% | **40.1%** | **42.8%** | **24.6%** |
    | **YOLO26s** | **60.5%** | **40.1%** | 40.6% | 23.2% |

    *Key Takeaway:* YOLO26s delivers a slight improvement in Precision (+0.7%) while maintaining the exact same overall Recall (40.1%). However, its mAP50 is slightly lower (-2.2%) and mAP50-95 is slightly lower (-1.4%) compared to YOLOv8s.

    ---

    ## 2. Model Complexity & Hardware Footprint

    The hardware benchmarks were evaluated on the local host device (NVIDIA GeForce GTX 1650, 4GB VRAM):

    | Metric | YOLOv8s | YOLO26s | Change |
    | :--- | :---: | :---: | :---: |
    | **Parameters** | 11.2M | **9.5M** | -15.2% |
    | **GFLOPs** | 28.6 | **20.5** | -28.3% |
    | **Inference Latency** | **15.7 ms** | 21.6 ms | +37.6% |
    | **Total Frame Processing Time** | **16.9 ms** | 22.3 ms | +32.0% |
    | **Actual Frame Rate** | **59.1 FPS** | 44.8 FPS | -24.2% |

    *Key Takeaway:* YOLO26s is significantly more compact, requiring 15.2% fewer parameters and 28.3% fewer GFLOPs. However, on the target GPU architecture (GTX 1650), YOLOv8s shows better optimized inference speeds (59.1 FPS vs 44.8 FPS).

    ---

    ## 3. Class-by-Class Performance Comparison

    Below is the class-by-class comparison of Precision (P), Recall (R), and mAP50 across the six target fruit categories:

    | Class | YOLOv8s P | YOLO26s P | YOLOv8s R | YOLO26s R | YOLOv8s mAP50 | YOLO26s mAP50 |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
    | **Apple** | 56.7% | **60.2%** | 36.2% | **43.6%** | 41.2% | **45.5%** |
    | **Banana** | 61.4% | **66.1%** | **37.5%** | 36.4% | **40.8%** | 39.2% |
    | **Grape** | 58.4% | **62.5%** | **34.6%** | 31.5% | **36.3%** | 32.5% |
    | **Orange** | **64.6%** | 51.8% | **36.9%** | 31.5% | **40.7%** | 30.5% |
    | **Pineapple** | 53.6% | **60.4%** | **45.5%** | 41.6% | **43.5%** | 40.4% |
    | **Watermelon** | 64.3% | 62.3% | 50.2% | **55.8%** | 54.1% | **55.3%** |

    ---

    ## 4. Key Observations & Analysis

    *   **Apple Performance Boost:** YOLO26s shows a substantial performance increase for Apples, with Precision rising to 60.2% (+3.5%), Recall rising to 43.6% (+7.4%), and mAP50 rising to 45.5% (+4.3%).
    *   **Watermelon Recall Gain:** Watermelon remains highly recognizable, with YOLO26s increasing its Recall to 55.8% (+5.6%) and mAP50 to 55.3% (+1.2%).
    *   **Orange Performance Drop:** YOLO26s shows a noticeable decrease in performance on Oranges, with mAP50 dropping to 30.5% (-10.2%) and Precision dropping to 51.8% (-12.8%).
    *   **Precision Bias:** YOLO26s shows an improvement in Precision for 4 out of 6 classes (Apple, Banana, Grape, Pineapple), reflecting a general design preference for reducing false positive detections.

    ---

    ## 5. Training Convergence Visualization

    Below is the training history overlay visualization comparing both models:

    ![YOLOv8s vs YOLO26s Comparison](/home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/yolov8s_vs_yolo26s_comparison.png)
    ```

*   [ ] **Step 2: Commit Task 2 changes**
    Notify the user before running git operations:
    ```bash
    rtk git add final_training_report.md
    rtk git commit -m "docs: update final training report to compare YOLOv8s and YOLO26s"
    ```
