# Task 2 Report: Implement Early Stopping Callback and Automatic Checkpoint Resuming

## What Was Implemented

1. **Automatic Checkpoint Resuming**:
   - Updated `train(args)` to check for an existing checkpoint file (`fruit_detection_runs/train/weights/last.pt`).
   - If `last.pt` exists, training resumes automatically (`resume=True`) using the checkpoint model weights.
   - If it does not exist, training starts fresh from the pretrained model (`yolov8s.pt`).

2. **Early Stopping callback (`stop_at_epoch_limit`)**:
   - Implemented an inner function `stop_at_epoch_limit(trainer)` registered to the YOLO `on_fit_epoch_end` event when `epochs_per_run` is set.
   - Tracks the number of completed epochs in the current chunk and triggers a graceful stop (`trainer.stop = True`) when the chunk limit (`epochs_per_run`) is reached.

3. **Checkpoint Optimizer State Preservation**:
   - Solved the Ultralytics default behavior of stripping the optimizer state from the model checkpoint when a training loop exits (which prevents proper resuming in subsequent chunks).
   - Monkeypatched `trainer.final_eval` in the callback to back up `last.pt` to `last.pt.bak` before it gets stripped, and then restore it after `final_eval` completes, ensuring the full optimizer state remains intact.
   - Added a `try...finally` block around `original_final_eval()` to guarantee that the backup file is always moved back to `last.pt` even if validation or `final_eval` raises an exception.
   - Implemented a re-entrant guard check `trainer._final_eval_monkeypatched` to prevent double monkeypatching and potential infinite recursion if the callback is invoked multiple times.

## What Was Tested and Test Results

- **Chunk 1 Training (Fresh Start)**:
  - Executed: `python train_fruit_detection.py --mode train --epochs 2 --epochs-per-run 1 --device 0 --batch 8 --workers 2`
  - Verified it started fresh using `yolov8s.pt`.
  - Verified it completed epoch 1, stopped gracefully via the early stopping callback, backed up `last.pt`, ran `final_eval`, and successfully restored `last.pt` preserving the optimizer state.
- **Checkpoint Inspection**:
  - Wrote a python script to load the saved `last.pt` checkpoint.
  - Verified that `optimizer` was preserved as a dictionary rather than being `None`, and the epoch was correctly saved as `1` (0-indexed).
- **Chunk 2 Training (Auto-Resume)**:
  - Executed the same training command.
  - Verified it correctly detected the checkpoint, loaded it, and resumed training exactly from epoch 2.
  - Verified training completed the final epoch successfully.

## Files Changed

- train_fruit_detection.py

## Self-Review Findings

- Checked all modifications for emojis: verified that zero emojis are present in the code modifications, output logs, or comments.
- No redundant changes or over-engineering (YAGNI) were introduced.

## Issues or Concerns

- None.
