# Design Specification: Fruit Detection PPT Visualizations and Slides Draft

## 1. Goal
Membuat materi presentasi (PPT) pendeteksian buah real-time YOLOv8s dengan visualisasi hasil training 50 epoch yang diperbarui, analisis performa per kelas buah, dan penjelasan teknis yang mendalam (arsitektur model, optimasi rasio dataset, dan strategi incremental training) dalam Bahasa Indonesia yang lugas dan objektif.

## 2. Rincian Perubahan File & Folder
1. **Penyimpanan Visualisasi 50 Epoch:**
   - Memindahkan folder visualisasi 1-epoch yang sudah usang dari `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train` ke folder baru bernama `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train_1epoch_backup`.
   - Menyalin folder berisi visualisasi 50-epoch lengkap dari `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs/train` ke `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train`.

2. **File Draf Materi PPT:**
   - Membuat file draf presentasi baru dengan nama deskriptif `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_ppt_draft.md`.
   - File ini akan berisi 9 slide terstruktur lengkap dengan metrik evaluasi akhir (Precision, Recall, mAP50, mAP50-95), analisis performa detail untuk tiap kelas buah, perbandingan rasio split dataset (60:27:13), penjelasan model YOLOv8s, dan alur incremental training.

3. **Dokumentasi Briefing:**
   - Mencatat aktivitas pembaruan visualisasi ini pada file briefing baru atau memperbarui file yang ada di `/home/tgrwjya/Documents/Briefing/README.md` (atau file relevan lainnya di direktori briefing).

## 3. Kerangka Slide Presentasi (fruit_detection_ppt_draft.md)
- **Slide 1:** Judul Proyek & Anggota Tim
- **Slide 2:** Latar Belakang & Kendala Hardware (GTX 1650 4GB VRAM)
- **Slide 3:** Spesifikasi Arsitektur Model (YOLOv8s, Anchor-Free, Decoupled Head, imgsz 640, batch 8)
- **Slide 4:** Dataset & Analisis Rasio Split Data (Rasio 60:27:13 setelah pemangkasan data train menjadi 2.000 gambar, dengan set validasi besar ~27% untuk evaluasi objektif)
- **Slide 5:** Algoritma Incremental Training Loop & Cooldown (Metode 5-epoch per chunk, jeda 30 detik untuk GPU cooldown, monkeypatch callback optimizer state)
- **Slide 6:** Hasil Akhir & Metrik Evaluasi Model (Penjelasan hasil mAP dan visualisasi kurva loss menggunakan file `results.png`)
- **Slide 7:** Analisis Performa per Kelas Buah (Mengapa Semangka mendapat akurasi tertinggi dan mengapa Anggur paling rendah berdasarkan grafik `confusion_matrix_normalized.png`)
- **Slide 8:** Implementasi Real-Time Inference (Menggunakan realtime_fruit_detection.py, tampilan FPS, class counter, antarmuka fancy box)
- **Slide 9:** Kesimpulan & Pengembangan Selanjutnya
