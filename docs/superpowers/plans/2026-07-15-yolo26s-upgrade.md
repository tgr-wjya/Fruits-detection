# YOLO26s Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the fruits detection codebase to support dynamic model selection (defaulting to YOLO26s) and route output runs to isolated directories to prevent data collision.

**Architecture:** Update argparse logic to accept `--model` in both training and inference scripts. Resolve project run directory names dynamically based on the target model. Replace references to the hardcoded `DEFAULT_PROJECT_NAME` with dynamic properties from parsed arguments.

**Tech Stack:** Python 3.10+, PyTorch, Ultralytics YOLO26s, OpenCV.

## Global Constraints
*   NO EMOJIS EVER: Do not write or include emojis in the code changes, print messages, comments, configurations, logs, or documentation.
*   Prefix all CLI/shell executions with `rtk` (Rust Token Killer) to minimize context token overhead.
*   Keep image memory caching disabled (`--cache False` or equivalent default) to avoid CUDA out-of-memory or system hangs on the 4GB VRAM GPU.

---

### Task 1: Update training script with dynamic model parsing and directory routing

**Files:**
*   Modify: `train_fruit_detection.py`

**Interfaces:**
*   Consumes: Dataset YAML configuration file `data.yaml`.
*   Produces: Command-line script supporting `--model` parameter and routing all training outputs to `fruit_detection_runs_<model_name>`.

*   [ ] **Step 1: Update CLI parser arguments and configure dynamic project routing**
    Add the `--model` argument to `main()` in `train_fruit_detection.py`, set the default value to `"yolo26s.pt"`, and determine the dynamic project name based on the selected model name.

    Modify `main()` from line 308 to 356:
    ```python
    def main():
        parser = argparse.ArgumentParser(
            description="Real-Time Fruit Detection - Training & Evaluation (YOLOv8)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
    Contoh penggunaan:
      python train_fruit_detection.py --mode train --epochs 100
      python train_fruit_detection.py --mode eval
      python train_fruit_detection.py --mode predict --source test/images
      python train_fruit_detection.py --mode export
            """
        )

        parser.add_argument("--mode", type=str, required=True,
                            choices=["train", "eval", "predict", "export"],
                            help="Mode: train | eval | predict | export")
        parser.add_argument("--model", type=str, default="yolo26s.pt",
                            help="Model weights file to load (default: yolo26s.pt)")
        parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                            help=f"Jumlah epoch training (default: {DEFAULT_EPOCHS})")
        parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                            help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
        parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE,
                            help=f"Image size (default: {DEFAULT_IMG_SIZE})")
        parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                            help=f"Device: 0,1,2.. untuk GPU atau cpu (default: {DEFAULT_DEVICE})")
        parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                            help=f"Data loader workers (default: {DEFAULT_WORKERS})")
        parser.add_argument("--weights", type=str, default=None,
                            help="Path ke model weights (default: auto detect dari training)")
        parser.add_argument("--source", type=str, default=None,
                            help="Source gambar/video untuk prediksi")
        parser.add_argument("--conf", type=float, default=0.5,
                            help="Confidence threshold untuk prediksi (default: 0.5)")
        parser.add_argument("--cache", type=str, default=str(DEFAULT_CACHE),
                            help=f"Cache mode: True/ram, disk, atau False (default: {DEFAULT_CACHE})")
        parser.add_argument("--epochs-per-run", type=int, default=None,
                            help="Jumlah epoch per chunk running sebelum cooldown (default: None, jalankan sekaligus)")
        parser.add_argument("--cooldown", type=int, default=30,
                            help="Durasi jeda pendinginan (detik) antar run/chunk (default: 30)")
        parser.add_argument("--is-chunk", action="store_true",
                            help="Internal flag: Menandakan proses dijalankan sebagai sub-proses chunk")

        args = parser.parse_args()

        # Convert string boolean to python boolean/type for Ultralytics
        if args.cache.lower() == 'true':
            args.cache = True
        elif args.cache.lower() == 'false':
            args.cache = False

        # Tentukan project name secara dinamis berdasarkan model
        model_stem = Path(args.model).stem
        if model_stem.startswith("yolov8"):
            args.project_name = "fruit_detection_runs"
        else:
            args.project_name = f"fruit_detection_runs_{model_stem}"
    ```

*   [ ] **Step 2: Update the chunk-training loop subprocess invocation**
    Update the auto-chunk loop in `main()` to load the correct checkpoint name and pass the `--model` parameter to sub-processes.

    Modify the chunk-training block in `main()` from line 357 onwards:
    ```python
        if args.mode == "train":
            # Jika user meminta chunk training dan ini adalah proses utama (bukan subprocess chunk)
            if args.epochs_per_run is not None and args.epochs_per_run > 0 and not args.is_chunk:
                import subprocess
                import time

                total_epochs = args.epochs
                epochs_per_run = args.epochs_per_run
                cooldown = args.cooldown
                last_pt = PROJECT_DIR / args.project_name / "train" / "weights" / "last.pt"

                print("=" * 60)
                print("  AUTO-CHUNK RUNNER DI-AKTIFKAN")
                print("=" * 60)
                print(f"  Target Epochs  : {total_epochs}")
                print(f"  Epochs Per Run : {epochs_per_run}")
                print(f"  Cooldown Pause : {cooldown} seconds")
                print(f"  Project Dir    : {args.project_name}")
                print("=" * 60)

                while True:
                    # Cek progress epoch saat ini dari checkpoint last.pt
                    completed_epochs = 0
                    if last_pt.exists():
                        try:
                            ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
                            if ckpt and "train_results" in ckpt and "epoch" in ckpt["train_results"]:
                                completed_epochs = len(ckpt["train_results"]["epoch"])
                        except Exception as e:
                            print(f"Gagal membaca checkpoint progress: {e}. Melanjutkan run...")

                    print(f"\nProgress Training: {completed_epochs}/{total_epochs} epoch selesai.")

                    if completed_epochs >= total_epochs:
                        print("Semua epoch telah berhasil diselesaikan!")
                        break

                    # Hitung target epoch run berikutnya
                    next_target = min(completed_epochs + epochs_per_run, total_epochs)
                    log_msg = f"Progress: {completed_epochs}/{total_epochs} epochs. Memulai chunk baru: Epoch {completed_epochs + 1} s/d {next_target}"
                    print(log_msg)

                    # Catat progress ke log file
                    log_file = PROJECT_DIR / args.project_name / "incremental_training.log"
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with open(log_file, "a") as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {log_msg}\n")
                    except Exception as e:
                        print(f"Gagal menulis ke log file: {e}")

                    # Jalankan subprocess script ini sendiri dengan flag --is-chunk dan parameter model
                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--mode", "train",
                        "--model", args.model,
                        "--epochs", str(total_epochs),
                        "--epochs-per-run", str(epochs_per_run),
                        "--cooldown", str(cooldown),
                        "--is-chunk",
                        "--batch", str(args.batch),
                        "--imgsz", str(args.imgsz),
                        "--device", str(args.device),
                        "--workers", str(args.workers),
                        "--cache", str(args.cache),
                    ]

                    result = subprocess.run(cmd)
                    if result.returncode != 0:
                        print(f"Subproses training terhenti dengan exit code {result.returncode}. Membatalkan loop.")
                        sys.exit(result.returncode)

                    # Check progress baru
                    new_completed = 0
                    if last_pt.exists():
                        try:
                            ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
                            if ckpt and "train_results" in ckpt and "epoch" in ckpt["train_results"]:
                                new_completed = len(ckpt["train_results"]["epoch"])
                        except Exception:
                            pass

                    # Jika progress tidak bertambah sama sekali, cegah infinite loop
                    if new_completed <= completed_epochs:
                        print("Peringatan: Tidak ada kemajuan epoch terdeteksi pada run ini. Menghentikan untuk mencegah loop tanpa akhir.")
                        break

                    if new_completed >= total_epochs:
                        print("\nSemua epoch telah berhasil diselesaikan!")
                        break

                    print(f"\nMemasuki masa pendinginan hardware selama {cooldown} detik...")
                    time.sleep(cooldown)

                print("=" * 60)
            else:
                # Menjalankan single-chunk training
                train(args)
        elif args.mode == "eval":
            evaluate(args)
        elif args.mode == "predict":
            predict(args)
        elif args.mode == "export":
            export_model(args)
    ```

*   [ ] **Step 3: Update `train`, `evaluate`, `predict`, and `export_model` helper functions**
    Replace all references to `DEFAULT_PROJECT_NAME` and `PRETRAINED_MODEL` with dynamic model parameters parsed into `args`.

    Update `train(args)`:
    ```python
    def train(args):
        from ultralytics import YOLO

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pretrained_model = PROJECT_DIR / args.model

        print("=" * 60)
        print(f"  TRAINING: Real-Time Fruit Detection ({Path(args.model).stem})")
        print("=" * 60)
        print(f"  Dataset     : {DATA_YAML}")
        print(f"  Pretrained  : {pretrained_model}")
        print(f"  Total Epochs: {args.epochs}")
        if args.epochs_per_run:
            print(f"  Chunk Size  : {args.epochs_per_run} epochs per run")
            print(f"  Cooldown    : {args.cooldown} seconds")
        print(f"  Batch Size  : {args.batch}")
        print(f"  Image Size  : {args.imgsz}")
        print(f"  Device      : {args.device}")
        print(f"  Workers     : {args.workers}")
        print(f"  Cache       : {args.cache}")
        print("=" * 60)

        # Cek apakah checkpoint sebelumnya ada untuk di-resume
        last_pt = PROJECT_DIR / args.project_name / "train" / "weights" / "last.pt"
        if last_pt.exists():
            print(f"Checkpoint ditemukan di {last_pt}. Me-resume training...")
            model = YOLO(str(last_pt))
            resume_mode = True

            # Daftarkan callback untuk meng-override epochs target pada saat resume
            def override_resume_epochs(trainer):
                if hasattr(trainer, 'args'):
                    trainer.epochs = args.epochs
                    trainer.args.epochs = args.epochs
                    print(f"Resume epochs target overridden to {args.epochs}")

            model.add_callback("on_pretrain_routine_start", override_resume_epochs)
        else:
            print(f"Tidak ada checkpoint. Memulai training baru menggunakan {pretrained_model}...")
            model = YOLO(str(pretrained_model))
            resume_mode = False

        # Jika diset epochs-per-run, daftarkan callback untuk stop training di akhir limit chunk
        if args.epochs_per_run is not None and args.epochs_per_run > 0:
            session_epochs_completed = 0
            def stop_at_epoch_limit(trainer):
                nonlocal session_epochs_completed
                session_epochs_completed += 1
                if session_epochs_completed >= args.epochs_per_run:
                    print(f"\nLimit epoch per run ({args.epochs_per_run}) tercapai untuk chunk ini.")
                    print("Menyimpan checkpoint dan menghentikan training secara graceful...")
                    trainer.stop = True

                    # Monkeypatch final_eval to preserve the optimizer state in last.pt
                    if not hasattr(trainer, '_final_eval_monkeypatched'):
                        trainer._final_eval_monkeypatched = True
                        original_final_eval = trainer.final_eval
                        def custom_final_eval(*eval_args, **eval_kwargs):
                            import shutil
                            last_pt_path = Path(trainer.last)
                            backup_pt = last_pt_path.with_suffix('.pt.bak')
                            if last_pt_path.exists():
                                shutil.copy(str(last_pt_path), str(backup_pt))
                                print(f"Mengamankan checkpoint dengan optimizer state ke {backup_pt}")

                            try:
                                original_final_eval(*eval_args, **eval_kwargs)
                            finally:
                                if backup_pt.exists():
                                    shutil.move(str(backup_pt), str(last_pt_path))
                                    print(f"Mengembalikan checkpoint dengan optimizer state ke {last_pt_path}")

                        trainer.final_eval = custom_final_eval

            model.add_callback("on_fit_epoch_end", stop_at_epoch_limit)

        # Jalankan training
        results = model.train(
            data=str(DATA_YAML),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
            project=str(PROJECT_DIR / args.project_name),
            name="train",
            exist_ok=True,
            resume=resume_mode,
            # Augmentasi data
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.0,
            mosaic=1.0,
            mixup=0.1,
            # Optimizer
            optimizer="auto",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            # Early stopping
            patience=15,
            # Logging
            verbose=True,
            plots=True,
        )

        best_model_path = PROJECT_DIR / args.project_name / "train" / "weights" / "best.pt"
        print("\n" + "=" * 60)
        print("  SUBPROSES CHUNK SELESAI!")
        print("=" * 60)

        return results
    ```

    Update `evaluate(args)`:
    ```python
    def evaluate(args):
        from ultralytics import YOLO

        # Cari model terbaik
        best_model = args.weights
        if best_model is None:
            best_model = str(PROJECT_DIR / args.project_name / "train" / "weights" / "best.pt")

        if not os.path.exists(best_model):
            print(f"Model tidak ditemukan: {best_model}")
            print("   Jalankan training terlebih dahulu!")
            sys.exit(1)

        print("=" * 60)
        print("  EVALUASI: Fruit Detection Model")
        print("=" * 60)
        print(f"  Model   : {best_model}")
        print(f"  Dataset : {DATA_YAML}")
        print("=" * 60)

        model = YOLO(best_model)

        # Evaluasi pada data validation
        print("\nEvaluasi pada Validation Set:")
        val_results = model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=True,
            project=str(PROJECT_DIR / args.project_name),
            name="val_eval",
            exist_ok=True,
        )

        # Evaluasi pada data test
        print("\nEvaluasi pada Test Set:")
        test_results = model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=True,
            project=str(PROJECT_DIR / args.project_name),
            name="test_eval",
            exist_ok=True,
        )

        print("\n" + "=" * 60)
        print("  EVALUASI SELESAI!")
        print(f"  Val  mAP50    : {val_results.box.map50:.4f}")
        print(f"  Val  mAP50-95 : {val_results.box.map:.4f}")
        print(f"  Test mAP50    : {test_results.box.map50:.4f}")
        print(f"  Test mAP50-95 : {test_results.box.map:.4f}")
        print("=" * 60)
    ```

    Update `predict(args)`:
    ```python
    def predict(args):
        from ultralytics import YOLO

        best_model = args.weights
        if best_model is None:
            best_model = str(PROJECT_DIR / args.project_name / "train" / "weights" / "best.pt")

        if not os.path.exists(best_model):
            print(f"Model tidak ditemukan: {best_model}")
            sys.exit(1)

        if args.source is None:
            print("Tentukan --source (path ke gambar atau folder)")
            sys.exit(1)

        print("=" * 60)
        print("  PREDIKSI: Fruit Detection")
        print("=" * 60)

        model = YOLO(best_model)

        results = model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            save=True,
            project=str(PROJECT_DIR / args.project_name),
            name="predict",
            exist_ok=True,
        )

        print(f"\nPrediksi selesai! Hasil disimpan di folder {args.project_name}/predict/")
    ```

    Update `export_model(args)`:
    ```python
    def export_model(args):
        from ultralytics import YOLO

        best_model = args.weights
        if best_model is None:
            best_model = str(PROJECT_DIR / args.project_name / "train" / "weights" / "best.pt")

        if not os.path.exists(best_model):
            print(f"Model tidak ditemukan: {best_model}")
            sys.exit(1)

        print("=" * 60)
        print("  EXPORT MODEL ke ONNX")
        print("=" * 60)

        model = YOLO(best_model)
        model.export(format="onnx", imgsz=args.imgsz, dynamic=True)
    ```

*   [ ] **Step 4: Verify the help argument outputs**
    Run: `rtk python train_fruit_detection.py --help`
    Expected: Output contains the `--model` flag option defaulting to `yolo26s.pt`.

*   [ ] **Step 5: Run a 1-epoch chunk trial to check auto-routing and compilation**
    Run: `rtk python train_fruit_detection.py --mode train --model yolo26s.pt --epochs 1 --epochs-per-run 1 --batch 4 --workers 2`
    Expected:
    - Automatically downloads the pre-trained weights `yolo26s.pt` to the workspace directory.
    - Creates the `fruit_detection_runs_yolo26s` directory.
    - Successfully processes the train loop for 1 epoch.
    - Saves the checkpoint correctly to `fruit_detection_runs_yolo26s/train/weights/last.pt`.
    - Terminated gracefully and saved the `incremental_training.log` details to the new project directory.

*   [ ] **Step 6: Commit Task 1 Changes**
    Run:
    ```bash
    rtk git add train_fruit_detection.py
    rtk git commit -m "feat: add dynamic model selection and run routing to training script"
    ```

---

### Task 2: Update real-time inference preview script to support dynamic model weighting paths

**Files:**
*   Modify: `realtime_fruit_detection.py`

**Interfaces:**
*   Consumes: Trained weights checkpoint `best.pt` in model specific run folders.
*   Produces: Command-line script accepting `--model` argument, directing weights routing to `fruit_detection_runs_<model_name>/train/weights/best.pt`.

*   [ ] **Step 1: Add `--model` argument to `main()` in `realtime_fruit_detection.py`**
    Update the argument parser in `main()` to accept `--model` (defaulting to `"yolo26s.pt"`).

    Modify `main()` in `realtime_fruit_detection.py` starting at line 363:
    ```python
    def main():
        parser = argparse.ArgumentParser(
            description="Real-Time Fruit Detection menggunakan Webcam/Video (YOLOv8)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
    Contoh penggunaan:
      python realtime_fruit_detection.py                          # Webcam default
      python realtime_fruit_detection.py --source 1               # Kamera ke-2
      python realtime_fruit_detection.py --source video.mp4       # File video
      python realtime_fruit_detection.py --conf 0.7               # Confidence tinggi
      python realtime_fruit_detection.py --save-video             # Simpan hasil
            """
        )

        parser.add_argument("--source", type=str, default=None,
                            help="Video source: 0 untuk webcam (default), atau path ke video")
        parser.add_argument("--model", type=str, default="yolo26s.pt",
                            help="Model weights base name to resolve run dir (default: yolo26s.pt)")
        parser.add_argument("--weights", type=str, default=None,
                            help="Path ke model weights (.pt)")
        parser.add_argument("--imgsz", type=int, default=640,
                            help="Ukuran input image (default: 640)")
        parser.add_argument("--conf", type=float, default=0.5,
                            help="Confidence threshold (default: 0.5)")
        parser.add_argument("--iou", type=float, default=0.45,
                            help="IOU threshold untuk NMS (default: 0.45)")
        parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu",
                            help="Device: 0 untuk GPU, cpu untuk CPU (default: auto-detect)")
        parser.add_argument("--save-video", action="store_true",
                            help="Simpan hasil deteksi ke video file")

        args = parser.parse_args()
        run_detection(args)
    ```

*   [ ] **Step 2: Update model path resolution in `run_detection()`**
    Modify `run_detection(args)` to dynamically compute the target project directory and look for the trained weight files inside `fruit_detection_runs_<model_name>/train/weights/best.pt`.

    Modify lines 185 to 208:
    ```python
    def run_detection(args):
        """
        Loop utama deteksi real-time.
        Mendukung input dari webcam (source=0) atau file video.
        """
        from ultralytics import YOLO

        # Tentukan project name secara dinamis berdasarkan model
        model_stem = Path(args.model).stem
        if model_stem.startswith("yolov8"):
            project_name = "fruit_detection_runs"
        else:
            project_name = f"fruit_detection_runs_{model_stem}"

        # ── Load Model ──────────────────────────────────────────────
        weights_path = args.weights
        if weights_path is None:
            # Coba cari model dari hasil training berdasarkan model target
            weights_path = str(PROJECT_DIR / project_name / "train" / "weights" / "best.pt")

        if not os.path.exists(weights_path):
            print(f"Model tidak ditemukan: {weights_path}")
            print("   Opsi:")
            print(f"   1. Jalankan training dulu: python train_fruit_detection.py --mode train --model {args.model}")
            print("   2. Tentukan path model spesifik: --weights path/to/best.pt")
            print(f"   3. Gunakan pretrained model langsung: --weights {args.model}")
            sys.exit(1)

        print(f"Loading model: {weights_path}")
        model = YOLO(weights_path)
    ```

*   [ ] **Step 3: Verify the help argument outputs**
    Run: `rtk python realtime_fruit_detection.py --help`
    Expected: Output contains the `--model` flag option defaulting to `yolo26s.pt`.

*   [ ] **Step 4: Verify default routing fallback response**
    Run: `rtk python realtime_fruit_detection.py --model yolo26s.pt`
    Expected: Program prints that the weight file `fruit_detection_runs_yolo26s/train/weights/best.pt` is missing, listing the correct command to train the model first.

*   [ ] **Step 5: Commit Task 2 Changes**
    Run:
    ```bash
    rtk git add realtime_fruit_detection.py
    rtk git commit -m "feat: support dynamic model weights resolution in realtime inference script"
    ```
