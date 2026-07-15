# Spec: YOLO26s Fruits Detection Upgrade

## Objective
Upgrade the core real-time object detection model of the project from YOLOv8s to YOLO26s. The goal is to evaluate if YOLO26s improves fruit detection metrics (Precision, Recall, mAP50, mAP50-95) and latency (FPS) on the target dataset while maintaining stable resource-constrained training limits on the NVIDIA GeForce GTX 1650 (4GB VRAM) laptop GPU.

## Tech Stack
*   Python 3.10+
*   PyTorch (with CUDA support)
*   Ultralytics YOLO (compatible with YOLO26 family, January 2026 release)
*   OpenCV (for real-time camera inference preview)

## Commands
*   **Show Training Help**:
    ```bash
    python train_fruit_detection.py --help
    ```
*   **Run YOLO26s Training**:
    ```bash
    python train_fruit_detection.py --model yolo26s.pt --epochs 50 --epochs-per-run 5 --cooldown 30
    ```
*   **Run Real-Time Inference**:
    ```bash
    python realtime_fruit_detection.py --model fruit_detection_runs_yolo26s/train/weights/best.pt
    ```

## Project Structure
*   [train_fruit_detection.py](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/train_fruit_detection.py): Python training controller. Will be modified to support dynamic model loading and output routing.
*   [realtime_fruit_detection.py](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/realtime_fruit_detection.py): Real-time inference preview script. Will be modified to support custom model paths.
*   `fruit_detection_runs_yolo26s/`: Target directory for YOLO26s checkpoints, logs, and evaluation reports.
*   `docs/superpowers/specs/2026-07-15-yolo26s-upgrade-design.md`: This specification file.

## Code Style
Keep the existing style:
*   Clear parameter logging blocks at launch.
*   Argument parsing via `argparse`.
*   Indonesian comments and output prints to match the legacy codebase style.
*   No emojis ever in code, comments, print statements, or logs.

Example structure update for argument definition:
```python
parser.add_argument("--model", type=str, default="yolo26s.pt",
                    help="Model weights file to load (default: yolo26s.pt)")
```

## Testing Strategy
1.  **CLI Validation**: Verify `--help` outputs for both updated scripts.
2.  **Dry Run**: Execute a 1-epoch dry-run of YOLO26s to ensure the PyTorch-CUDA pipeline runs without memory fragmentation or OOM errors:
    ```bash
    python train_fruit_detection.py --model yolo26s.pt --epochs 1 --epochs-per-run 1
    ```
3.  **Resume Check**: Verify the checkpoint resume loop loads weights correctly from the new `fruit_detection_runs_yolo26s/train/weights/last.pt` path.
4.  **Inference Run**: Launch the camera preview script using the initial pre-trained model to verify OpenCV display and frame latency before final training.

## Boundaries
*   **Always do**: 
    *   Verify git configurations (`user.name` and `user.email`) before any commit.
    *   Prefix all CLI executions with `rtk`.
    *   Ensure `--cache` defaults to `False` to prevent memory overflows.
*   **Ask first**: Adding new dependencies or updating system-wide CUDA/conda configurations.
*   **Never do**: 
    *   Use emojis anywhere.
    *   Silently proceed if training freezes, hangs, or encounters OOM errors.
    *   Commit checkpoint files (`.pt`) to version control.

## Success Criteria
*   Both scripts execute without crashes when passed `--model yolo26s.pt`.
*   Training output is properly routed to `fruit_detection_runs_yolo26s/` (derived from the model name) instead of overwriting `fruit_detection_runs/`.
*   The incremental resume loop performs correct learning rate callback monkeypatching for YOLO26s.
*   A comparative evaluation report can be generated once the full 50 epochs complete.

## Open Questions
*   None.
