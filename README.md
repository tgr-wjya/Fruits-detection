# Real-Time Fruit Detection (YOLOv8s)

This repository contains scripts to fine-tune a YOLOv8s model on a custom fruits detection dataset, manage resource-constrained training workloads incrementally, and run real-time camera inference.

The model is optimized to detect six classes of fruits: **Apple, Banana, Grape, Orange, Pineapple, and Watermelon**.

---

## Key Features

- **Incremental Training Loop:** Prevents laptop system hangs or freezes on consumer hardware. Training can be split into chunks (e.g., 5 epochs per run) with automatic hardware cooldown periods.
- **Resource Optimization:** Pre-configured low-memory defaults (no image RAM caching, reduced worker count, and CUDA cache clearing) to run safely on low-VRAM GPUs like the NVIDIA GeForce GTX 1650 (4GB VRAM).
- **Graceful Callback System:** Custom callbacks manage early stopping inside YOLOv8. Re-entrant monkeypatches protect and restore the model's optimizer and learning rate scheduler states (`last.pt`) during resumes.
- **Interactive Camera Inference:** A dedicated real-time preview script that renders detections with corner indicators, a dashboard with performance metrics (FPS), and class counts.

---

## Dataset Details
The model was fine-tuned and validated on a fruit detection dataset:
- **Validation Split:** 914 images, containing 3,227 labeled fruit instances.
- **Target Classes:** Apple, Banana, Grape, Orange, Pineapple, Watermelon.

---

## Final Training Results (50 Epochs)

Below are the final evaluation metrics of the model after 50 epochs:

- **Overall Precision:** 59.8%
- **Overall Recall:** 40.1%
- **mAP50:** 42.8%
- **mAP50-95:** 24.6%

### Category Performance Breakdown:

| Class | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **Watermelon** | 64.3% | 50.2% | 54.1% | 33.8% |
| **Orange** | 64.6% | 36.9% | 40.7% | 23.3% |
| **Banana** | 61.4% | 37.5% | 40.8% | 21.0% |
| **Apple** | 56.7% | 36.2% | 41.2% | 26.1% |
| **Grape** | 58.4% | 34.6% | 36.3% | 19.9% |
| **Pineapple** | 53.6% | 45.5% | 43.5% | 23.7% |

*Note: Watermelon has the highest accuracy and recall due to its distinct features, while Grapes are more challenging due to clustering and smaller relative bounding box sizes.*

---

## Installation & Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd Fruits-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install opencv-python ultralytics numpy torch
   ```
   *Note: Ensure you do not use `opencv-python-headless` if you intend to run the graphical webcam preview.*

---

## Usage Guide

### 1. Incremental Model Training

To train the model in chunks of 5 epochs with a 30-second hardware cooldown between runs, run:
```bash
python train_fruit_detection.py --mode train --epochs 50 --epochs-per-run 5 --cooldown 30 --batch 8 --workers 2
```

Arguments:
- `--epochs`: Total target epochs (default: 50).
- `--epochs-per-run`: Number of epochs to train before pausing (default: 5).
- `--cooldown`: Cooldown sleep in seconds between runs (default: 30).
- `--batch`: Batch size. For 4GB VRAM cards, 8 or 4 is recommended to avoid CUDA memory fragmentation.
- `--workers`: Number of data loader CPU worker processes (default: 2).

The script automatically detects existing checkpoints in `fruit_detection_runs/train/weights/last.pt` and safely resumes progression without breaking learning curves.

### 2. Real-Time Camera Inference

Ensure your graphical environment (X11/Wayland) is active, and run:
```bash
python realtime_fruit_detection.py
```

Options:
- Run on secondary camera: `python realtime_fruit_detection.py --source 1`
- Run on a video file: `python realtime_fruit_detection.py --source path/to/video.mp4`
- Adjust detection confidence threshold: `python realtime_fruit_detection.py --conf 0.35`

Interactive Controls:
- **Q:** Quit the camera session.
- **P:** Pause/Resume the camera stream.
- **S:** Save a annotated screenshot of the current frame to the directory.
