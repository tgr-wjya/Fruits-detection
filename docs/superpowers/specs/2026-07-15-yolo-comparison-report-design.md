# Spec: YOLOv8s vs YOLO26s Evaluation & Comparison Report

## Objective
Update the legacy [final_training_report.md](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/final_training_report.md) with comprehensive comparative evaluation metrics, model parameter profiles, and visual learning curves comparing the original YOLOv8s model and the upgraded YOLO26s model on the target fruits detection dataset.

## Tech Stack
*   Python 3.10+
*   Pandas (for CSV parsing)
*   Matplotlib (for chart generation)
*   Markdown (for report updates)

## Core Components

### 1. Data Extractor & Comparison Plotter (`plot_comparison.py`)
We will create a new Python utility [plot_comparison.py](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/plot_comparison.py) in the project root. This script will:
*   Load [fruit_detection_runs_backup/train/results.csv](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train/results.csv) (YOLOv8s) and [fruit_detection_runs_yolo26s/train/results.csv](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_yolo26s/train/results.csv) (YOLO26s).
*   Create a 2x2 multi-panel layout using Matplotlib:
    *   **Subplot 1 (Precision & Recall):** Precision and Recall curves for both models.
    *   **Subplot 2 (mAP Accuracy):** mAP50 and mAP50-95 curves for both models.
    *   **Subplot 3 (Box Loss):** Train/Val Bounding Box Loss curves.
    *   **Subplot 4 (Cls Loss):** Train/Val Classification Loss curves.
*   Save the final visualization to `yolov8s_vs_yolo26s_comparison.png` in the repository root.

### 2. Comparative Training Report (`final_training_report.md`)
We will overwrite [final_training_report.md](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/final_training_report.md) to structure the data comparisons:
*   **Metric Comparison Matrix:** Comparison table of overall Precision, Recall, mAP50, and mAP50-95.
*   **Model Profiles:** Compare parameters (11.2M vs 9.5M), GFLOPs (28.6 vs 20.5), and device latency (16.9ms vs 22.3ms).
*   **Class-by-Class Breakdown:** A table detailing Apple, Banana, Grape, Orange, Pineapple, and Watermelon across both models.
*   **Visualizations:** Embed the newly generated comparison plot image.
*   **Observations:** Analyze trade-offs (e.g. YOLO26s size/compute benefits vs validation differences).

## Safety & Style Guidelines
*   **No Emojis:** Do NOT use emojis in code, comments, print statements, or documentation.
*   **RTK CLI Prefix:** Always run commands with `rtk` (e.g., `rtk python plot_comparison.py`).
*   **File Integrity:** Keep all other project files untouched.

## Success Criteria
*   The script [plot_comparison.py](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/plot_comparison.py) executes cleanly and generates the `yolov8s_vs_yolo26s_comparison.png` file.
*   The updated report [final_training_report.md](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/final_training_report.md) displays accurate, verified metrics for both YOLOv8s and YOLO26s.
*   The embedded comparison image loads properly in Markdown viewers.
