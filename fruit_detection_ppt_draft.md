# Draf Materi Slide Presentasi: Deteksi Buah Real-Time berbasis YOLOv8s

Dokumen ini berisi materi slide presentasi yang siap disalin ke Microsoft PowerPoint. File gambar visualisasi dapat ditemukan pada folder `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train/`.

---

## Slide 1: Judul Utama
* **Judul:** Deteksi Buah Real-Time berbasis YOLOv8s
* **Subjudul:** Implementasi Incremental Training, Evaluasi Model, dan Deteksi Menggunakan Kamera Real-Time
* **Poin Slide:**
  - Penerapan computer vision untuk deteksi 6 kelas buah secara otomatis.
  - Penanganan kendala keterbatasan hardware melalui metode training bertahap.
  - Implementasi dashboard antarmuka pada pengujian langsung.

---

## Slide 2: Latar Belakang & Masalah
* **Judul Slide:** Tantangan Deteksi Buah & Batasan Hardware
* **Poin Slide:**
  - **Kelas Target:** Deteksi 6 kelas buah (Apple, Banana, Grape, Orange, Pineapple, Watermelon).
  - **Kendala Hardware:** Training model deep learning membutuhkan daya GPU tinggi. Laptop pengujian menggunakan NVIDIA GeForce GTX 1650 (4GB VRAM) yang rentan kehabisan memori.
  - **Masalah:** Proses training tanpa henti menyebabkan memori CUDA terfragmentasi, suhu naik (overheat), dan sistem laptop hang (beku).
  - **Solusi:** Perlu manajemen alokasi memori yang ketat dan modifikasi pada skenario training.

---

## Slide 3: Spesifikasi Model & Konfigurasi
* **Judul Slide:** Arsitektur Model YOLOv8s
* **Poin Slide:**
  - **Arsitektur Model:** YOLOv8s (versi Small) - terpilih karena memiliki parameter yang lebih ringkas sehingga cepat dan efisien untuk pemrosesan video secara real-time.
  - **Anchor-Free Detection:** YOLOv8 tidak menggunakan kotak acuan statis (anchor-free), melainkan memprediksi pusat objek secara langsung, mengurangi kompleksitas komputasi.
  - **Decoupled Head:** Cabang klasifikasi dan regresi bounding box dipisahkan untuk mempercepat waktu konvergensi model.
  - **Konfigurasi Utama:**
    - Input Ukuran Gambar: 640x640 piksel
    - Ukuran Batch: 8 (dibatasi untuk menghindari kehabisan VRAM)
    - Optimizer: Auto-configured dengan Automatic Mixed Precision (AMP) diaktifkan.

---

## Slide 4: Karakteristik Dataset & Optimasi Rasio Data
* **Judul Slide:** Dataset & Pemangkasan Berimbang (Balanced Split Ratio)
* **Poin Slide:**
  - **Pengurangan Data Training:** Data awal training dipangkas secara berimbang (balanced sampling) menjadi tepat 2.000 gambar menggunakan reduce_dataset.py agar beban RAM laptop saat training berkurang.
  - **Rasio Pembagian Data (Split Ratio):**
    - Training Set: 2.000 gambar (~59.3%)
    - Validation Set: 914 gambar (~27.1%)
    - Test Set: 457 gambar (~13.6%)
    - Total Data Aktif: 3.371 gambar.
  - **Keuntungan Struktur Data:** Set validasi yang relatif besar (~27%) menjamin evaluasi performa model berjalan sangat objektif, ketat, dan bebas dari bias overfitting.
  - **Visualisasi Dataset:**
    ![labels.jpg](/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train/labels.jpg)

---

## Slide 5: Alur Kerja Incremental Training (Solusi GPU Overheat)
* **Judul Slide:** Algoritma Training Bertahap & Cooldown
* **Poin Slide:**
  - **Mekanisme Chunk:** Target total 50 epoch dipecah menjadi sub-proses training modular, masing-masing berjalan selama 5 epoch secara bertahap.
  - **Mekanisme Cooldown:** Sistem secara otomatis menjeda training selama 30 detik setiap kali 1 chunk (5 epoch) selesai untuk menurunkan suhu GPU.
  - **Safe State Recovery:** Modifikasi callback pada script training bertugas menyimpan serta mengamankan optimizer state dan learning rate scheduler (`last.pt`), sehingga saat proses dilanjutkan (resume), kurva pelatihan tidak terganggu.
  - **Optimal Data Loading:** Mengaktifkan worker CPU seminimal mungkin (2 workers) dan menonaktifkan cache gambar di RAM untuk stabilitas laptop.

---

## Slide 6: Analisis Hasil Training Akhir (50 Epochs)
* **Judul Slide:** Performa Evaluasi Model Akhir
* **Poin Slide:**
  - **Hasil Evaluasi Keseluruhan:**
    - Precision (P): 59.8% (Akurasi ketepatan prediksi buah)
    - Recall (R): 40.1% (Persentase buah nyata yang berhasil dideteksi)
    - mAP50: 42.8% (Mean Average Precision dengan batas IoU 0.5)
    - mAP50-95: 24.6% (mAP di seluruh rentang tingkat kecocokan IoU 0.5 hingga 0.95)
  - **Kurva Belajar:** Loss fungsi box_loss, cls_loss, dan dfl_loss pada data latih maupun validasi turun secara konisten tanpa gejala overfitting yang ekstrem.
  - **Visualisasi Kurva Belajar:**
    ![results.png](/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train/results.png)

---

## Slide 7: Analisis Performa Kelas Buah (Evaluasi Detail)
* **Judul Slide:** Analisis Kategori Buah: Kelebihan & Hambatan
* **Poin Slide:**
  - **Performa Terbaik - Semangka (Watermelon):**
    - mAP50: 54.1%, Recall: 50.2%.
    - Alasan: Semangka memiliki karakteristik visual yang sangat unik (bentuk bulat besar dan pola garis hijau-hitam) yang mudah dikenali oleh filter konvolusi.
  - **Performa Terendah - Anggur (Grape):**
    - mAP50: 36.3%, Recall: 34.6%.
    - Alasan: Buah anggur berukuran kecil, sering bergerombol rapat, dan terjadi oklusi (saling menutupi antarbuah atau daun), sehingga model kesulitan mengisolasi bounding box individu.
  - **Visualisasi Matriks Kesalahan & Kurva PR:**
    ![confusion_matrix_normalized.png](/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train/confusion_matrix_normalized.png)
    ![BoxPR_curve.png](/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train/BoxPR_curve.png)

---

## Slide 8: Uji Coba Deteksi Real-Time Kamera
* **Judul Slide:** Deteksi Langsung Kamera & Dashboard Metrik
* **Poin Slide:**
  - **Inference Real-Time:** Diuji langsung menggunakan file video dan live webcam melalui script realtime_fruit_detection.py.
  - **Dashboard Performa:**
    - Indikator FPS dinamis: Mencapai rata-rata ~59 FPS pada kartu grafis GTX 1650, sangat lancar untuk aplikasi waktu nyata.
    - Penghitung Kelas (Class Counter): Menghitung secara dinamis total masing-masing buah yang terlihat di layar.
  - **Antarmuka Fancy Box:** Bounding box modern dengan garis sudut yang tegas (bukan kotak penuh biasa) untuk estetika UI yang bersih.
  - **Kontrol Interaktif:** Shortcut keyboard untuk menyimpan tangkapan layar beranotasi (tombol S), menjeda kamera (tombol P), dan keluar (tombol Q).
  - **Visualisasi Uji Coba:**
    ![val_batch0_pred.jpg](/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train/val_batch0_pred.jpg)

---

## Slide 9: Kesimpulan & Rencana ke Depan
* **Judul Slide:** Kesimpulan & Rencana Tindak Lanjut
* **Poin Slide:**
  - **Kesimpulan:** Model YOLOv8s berhasil dilatih secara aman tanpa kendala hardware beku (system crash). Kecepatan pemrosesan frame (59 FPS) sudah melampaui standar aplikasi real-time umum.
  - **Peluang Optimasi:**
    - Perbaikan dataset anggur (menambah variasi data anggur yang saling bertumpuk).
    - Mencoba teknik konversi model ke format ONNX atau TensorRT untuk meningkatkan efisiensi komputasi pada perangkat edge yang lebih kecil.
