"""
=============================================================================
  Real-Time Fruit Detection - Webcam / Video
  Model   : YOLOv8s (fine-tuned on Fruits dataset)
  Classes : Apple, Banana, Grape, Orange, Pineapple, Watermelon
=============================================================================

CARA PENGGUNAAN:
  1. Pastikan sudah training model (jalankan train_fruit_detection.py dulu)

  2. Deteksi lewat webcam (real-time):
       python realtime_fruit_detection.py

  3. Deteksi lewat file video:
       python realtime_fruit_detection.py --source video.mp4

  4. Gunakan model custom:
       python realtime_fruit_detection.py --weights path/to/best.pt

  5. Tekan 'Q' untuk keluar dari deteksi real-time.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent

# Warna unik per kelas (BGR format) - warna cerah dan mudah dibedakan
CLASS_COLORS = {
    "Apple":      (50, 50, 230),     # Merah
    "Banana":     (30, 220, 250),    # Kuning
    "Grape":      (180, 60, 150),    # Ungu
    "Orange":     (40, 140, 255),    # Orange
    "Pineapple":  (50, 200, 50),     # Hijau
    "Watermelon": (60, 180, 75),     # Hijau tua
}


def draw_fancy_box(frame, x1, y1, x2, y2, color, label, confidence, thickness=2):
    """
    Menggambar bounding box dengan style modern.
    - Corner markers (bukan full rectangle)
    - Label background dengan rounded effect
    - Semi-transparent overlay
    """
    # Semi-transparent fill di dalam box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    # Bounding box utama
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Corner markers (lebih tebal & lebih panjang di sudut)
    corner_len = min(30, (x2 - x1) // 4, (y2 - y1) // 4)
    corner_thick = thickness + 2

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, corner_thick)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, corner_thick)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, corner_thick)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, corner_thick)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, corner_thick)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, corner_thick)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, corner_thick)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, corner_thick)

    # Label background
    text = f"{label} {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    label_x1 = x1
    label_y1 = y1 - text_h - 12
    label_x2 = x1 + text_w + 12
    label_y2 = y1

    # Pastikan label tidak keluar frame
    if label_y1 < 0:
        label_y1 = y2
        label_y2 = y2 + text_h + 12

    # Background label (filled rounded rectangle)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)

    # Text
    text_x = label_x1 + 6
    text_y = label_y2 - 6
    cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), font_thickness, cv2.LINE_AA)


def draw_dashboard(frame, fps, detections, frame_count):
    """
    Menggambar dashboard info di atas frame:
    - FPS counter
    - Jumlah deteksi per kelas
    - Frame counter
    """
    h, w = frame.shape[:2]

    # Header bar (semi-transparent)
    header_height = 45
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "FRUIT DETECTION", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # FPS (warna berdasarkan performa)
    if fps >= 25:
        fps_color = (0, 255, 0)      # Hijau = lancar
    elif fps >= 15:
        fps_color = (0, 220, 255)    # Kuning = cukup
    else:
        fps_color = (0, 0, 255)      # Merah = lambat

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)

    # Detection summary (sidebar kanan)
    if detections:
        # Hitung jumlah per kelas
        class_counts = {}
        for det in detections:
            cls = det["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        sidebar_w = 200
        sidebar_h = len(class_counts) * 35 + 50
        sidebar_x = w - sidebar_w - 10
        sidebar_y = header_height + 10

        # Background sidebar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (sidebar_x, sidebar_y),
                       (sidebar_x + sidebar_w, sidebar_y + sidebar_h),
                       (30, 30, 30), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

        # Title sidebar
        cv2.putText(frame, f"Detected: {len(detections)}",
                    (sidebar_x + 10, sidebar_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Per-class count
        for i, (cls, count) in enumerate(sorted(class_counts.items())):
            y_pos = sidebar_y + 50 + i * 35
            color = CLASS_COLORS.get(cls, (255, 255, 255))

            # Color dot
            cv2.circle(frame, (sidebar_x + 20, y_pos - 5), 6, color, -1)

            # Text
            cv2.putText(frame, f"{cls}: {count}",
                        (sidebar_x + 35, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Footer - instructions
    footer_y = h - 15
    cv2.putText(frame, "Press 'Q' to quit | 'S' to screenshot | 'P' to pause",
                (10, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def run_detection(args):
    """
    Loop utama deteksi real-time.
    Mendukung input dari webcam (source=0) atau file video.
    """
    from ultralytics import YOLO

    # ── Load Model ──────────────────────────────────────────────
    weights_path = args.weights
    if weights_path is None:
        # Coba cari model dari hasil training
        weights_path = str(PROJECT_DIR / "fruit_detection_runs" / "train" / "weights" / "best.pt")

    if not os.path.exists(weights_path):
        print(f"Model tidak ditemukan: {weights_path}")
        print("   Opsi:")
        print("   1. Jalankan training dulu: python train_fruit_detection.py --mode train")
        print("   2. Tentukan path model: --weights path/to/best.pt")
        print(f"   3. Gunakan pretrained: --weights {PRETRAINED_MODEL}")
        sys.exit(1)

    print(f"Loading model: {weights_path}")
    model = YOLO(weights_path)

    # ── Open Video Source ───────────────────────────────────────
    source = args.source
    is_webcam = False

    if source is None or source == "0":
        source = 0
        is_webcam = True
        print("Membuka webcam...")
    elif source.isdigit():
        source = int(source)
        is_webcam = True
        print(f"Membuka kamera {source}...")
    else:
        print(f"Membuka video: {source}")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Gagal membuka video source!")
        if is_webcam:
            print("   Pastikan webcam terhubung dan tidak dipakai aplikasi lain.")
        sys.exit(1)

    # Set resolusi webcam (jika webcam)
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Resolusi: {actual_w}x{actual_h}")

    # ── Video writer (opsional, untuk save hasil) ───────────────
    out_writer = None
    if args.save_video:
        output_path = str(PROJECT_DIR / "output_detection.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, 30, (actual_w, actual_h))
        print(f"Menyimpan hasil ke: {output_path}")

    # ── Main Detection Loop ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("  REAL-TIME FRUIT DETECTION AKTIF")
    print("  Tekan 'Q' untuk keluar")
    print("  Tekan 'S' untuk screenshot")
    print("  Tekan 'P' untuk pause/resume")
    print("=" * 50 + "\n")

    frame_count = 0
    fps = 0.0
    prev_time = time.time()
    paused = False

    # Nama window
    window_name = "Fruit Detection - Real Time"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(actual_w, 1280), min(actual_h, 720))

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    print("Gagal baca frame dari webcam")
                    break
                else:
                    print("Video selesai diputar")
                    break

            frame_count += 1

            # ── Inference YOLOv8 ────────────────────────────────
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )

            # ── Parse & Draw Detections ─────────────────────────
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Koordinat bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]

                        # Simpan deteksi
                        detections.append({
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": (x1, y1, x2, y2),
                        })

                        # Gambar bounding box
                        color = CLASS_COLORS.get(class_name, (255, 255, 255))
                        draw_fancy_box(frame, x1, y1, x2, y2, color,
                                       class_name, confidence)

            # ── Hitung FPS ──────────────────────────────────────
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # ── Gambar Dashboard ────────────────────────────────
            draw_dashboard(frame, fps, detections, frame_count)

            # ── Save video ──────────────────────────────────────
            if out_writer is not None:
                out_writer.write(frame)

        # ── Tampilkan Frame ─────────────────────────────────────
        cv2.imshow(window_name, frame)

        # ── Keyboard Controls ───────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            print("\nKeluar dari deteksi...")
            break
        elif key == ord("s") or key == ord("S"):
            screenshot_path = str(PROJECT_DIR / f"screenshot_{frame_count}.jpg")
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot disimpan: {screenshot_path}")
        elif key == ord("p") or key == ord("P"):
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(status)

    # ── Cleanup ─────────────────────────────────────────────────
    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()

    print(f"\nTotal frames diproses: {frame_count}")
    print(f"   Rata-rata FPS: {fps:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

# Path ke pretrained (untuk fallback)
PRETRAINED_MODEL = PROJECT_DIR / "yolov8s.pt"


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


if __name__ == "__main__":
    main()
