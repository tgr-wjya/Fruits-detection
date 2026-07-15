# Real-Time Fruit Detection (YOLOv8s & YOLO26s)

This repository contains scripts to fine-tune YOLOv8s and YOLO26s models on a custom fruits detection dataset, manage resource-constrained training workloads incrementally, and run real-time camera inference.

The models are optimized to detect six classes of fruits: **Apple, Banana, Grape, Orange, Pineapple, and Watermelon**.

---

## Key Features

- **Incremental Training Loop:** Prevents laptop system hangs or freezes on consumer hardware. Training can be split into chunks (e.g., 5 epochs per run) with automatic hardware cooldown periods.
- **Resource Optimization:** Pre-configured low-memory defaults (no image RAM caching, reduced worker count, and CUDA cache clearing) to run safely on low-VRAM GPUs like the NVIDIA GeForce GTX 1650 (4GB VRAM).
- **Graceful Callback System:** Custom callbacks manage early stopping inside YOLO. Re-entrant monkeypatches protect and restore the model's optimizer and learning rate scheduler states (`last.pt`) during resumes.
- **Interactive Camera Inference:** A dedicated real-time preview script that renders detections with corner indicators, a dashboard with performance metrics (FPS), and class counts.

---

## Dataset Details
The models were fine-tuned and validated on a fruit detection dataset:
- **Validation Split:** 914 images, containing 3,227 labeled fruit instances.
- **Target Classes:** Apple, Banana, Grape, Orange, Pineapple, Watermelon.

---

## Final Training Results (50 Epochs)

Below is the comparative performance metrics of the models after 50 epochs:

### 1. Overall Performance Comparison

| Model | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8s** | 59.8% | **40.1%** | **42.8%** | **24.6%** |
| **YOLO26s** | **60.5%** | **40.1%** | 40.6% | 23.2% |

### 2. Class-by-Class Performance Breakdown

| Class | YOLOv8s P | YOLO26s P | YOLOv8s R | YOLO26s R | YOLOv8s mAP50 | YOLO26s mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple** | 56.7% | **60.2%** | 36.2% | **43.6%** | 41.2% | **45.5%** |
| **Banana** | 61.4% | **66.1%** | **37.5%** | 36.4% | **40.8%** | 39.2% |
| **Grape** | 58.4% | **62.5%** | **34.6%** | 31.5% | **36.3%** | 32.5% |
| **Orange** | **64.6%** | 51.8% | **36.9%** | 31.5% | **40.7%** | 30.5% |
| **Pineapple** | 53.6% | **60.4%** | **45.5%** | 41.6% | **43.5%** | 40.4% |
| **Watermelon** | 64.3% | 62.3% | 50.2% | **55.8%** | 54.1% | **55.3%** |

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
   pip install opencv-python ultralytics numpy torch pandas matplotlib
   ```
   *Note: Ensure you do not use `opencv-python-headless` if you intend to run the graphical webcam preview.*

---

## Usage Guide

### 1. Incremental Model Training

To train a model in chunks of 5 epochs with a 30-second hardware cooldown between runs, run:
```bash
# YOLO26s (Default)
python train_fruit_detection.py --mode train --model yolo26s.pt --epochs 50 --epochs-per-run 5 --cooldown 30 --batch 8 --workers 2

# YOLOv8s
python train_fruit_detection.py --mode train --model yolov8s.pt --epochs 50 --epochs-per-run 5 --cooldown 30 --batch 8 --workers 2
```

Arguments:
- `--mode`: Operations mode: `train`, `eval`, `predict`, `export`.
- `--model`: Weights file base name to initialize training (default: `yolo26s.pt`).
- `--epochs`: Total target epochs (default: 50).
- `--epochs-per-run`: Number of epochs to train before pausing (default: 5).
- `--cooldown`: Cooldown sleep in seconds between runs (default: 30).
- `--batch`: Batch size. For 4GB VRAM cards, 8 or 4 is recommended to avoid CUDA memory fragmentation.
- `--workers`: Number of data loader CPU worker processes (default: 2).

The script automatically detects existing checkpoints in the model's respective training run folder (e.g. `fruit_detection_runs_yolo26s/train/weights/last.pt`) and safely resumes progression without breaking learning curves.

### 2. Real-Time Camera Inference

Ensure your graphical environment (X11/Wayland) is active, and run:
```bash
# YOLO26s (Default)
python realtime_fruit_detection.py --model yolo26s.pt

# YOLOv8s
python realtime_fruit_detection.py --model yolov8s.pt
```

Options:
- Run on secondary camera: `python realtime_fruit_detection.py --source 1`
- Run on a video file: `python realtime_fruit_detection.py --source path/to/video.mp4`
- Adjust detection confidence threshold: `python realtime_fruit_detection.py --conf 0.35`

Interactive Controls:
- **Q:** Quit the camera session.
- **P:** Pause/Resume the camera stream.
- **S:** Save an annotated screenshot of the current frame to the directory.

---

## Web Application Deployment

A client-side browser version of this fruit detection application is hosted live on **GitHub Pages**:

*   **Live App URL:** [https://tgr-wjya.github.io/Fruits-detection/](https://tgr-wjya.github.io/Fruits-detection/)
*   **Static Assets:** Located inside the [docs/](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/docs) directory of the repository.

### Client-Side Performance Note (GPU vs. CPU Fallback)
Because deep learning models run directly in the client's browser, frame rates are highly dependent on security settings and system configurations:
*   **WebGPU (GPU-accelerated):** On default installations of Google Chrome (v113+), Microsoft Edge, or Apple Safari (v18+) on Windows, macOS, iOS, or Android, the page will initialize using the GPU, achieving high-speed inference.
*   **WebAssembly (CPU Fallback):** On privacy-focused browsers (such as Brave with Shields/Fingerprinting Protection active) or Linux environments without enabled experimental flags, the browser blocks access to the GPU adapter. The application automatically falls back to multithreaded CPU WebAssembly (WASM), resulting in significantly lower frame rates (~1-2 FPS).

### Recommended Testing: Local Python Script
For full native performance (e.g., ~45-60 FPS using GPU CUDA acceleration), we recommend testing the model locally using our dedicated Python OpenCV script:
1.  **Clone the repository locally.**
2.  **Execute the real-time webcam command:**
    ```bash
    python realtime_fruit_detection.py --model yolo26s.pt
    ```
This bypasses browser virtualization overhead entirely to showcase the actual throughput of the custom fine-tuned weights on hardware.

---

## Credits

This project was developed as a **Tugas Besar** for the Computer Vision class (1A4154CA) at Mercu Buana University.

*   **Group:** Group 5
*   **Supporting Lecturer:** Muhaimin Hasanudin, S.T, M.Kom
*   **Group Members:**
    1.  Tegar Wijaya Kusuma (41523010217)
    2.  Muhammad Ihsar Fatiha (41523010202)
    3.  Ahmad Khadavi (41523010215)
