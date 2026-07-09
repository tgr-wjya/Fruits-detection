# Fruit Detection PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menyiapkan file draf presentasi deskriptif (dalam Bahasa Indonesia) beserta pembaruan visualisasi hasil training 50 epoch untuk presentasi PPT proyek pendeteksian buah.

**Architecture:** Mengarsipkan data visualisasi lama, memindahkan data visualisasi 50 epoch ke direktori backup, membuat berkas Markdown berisi susunan slide lengkap dengan metrik riil, dan memperbarui catatan briefing pengoperasian model.

**Tech Stack:** Bash shell commands, Markdown documentation.

## Global Constraints

- **NO EMOJIS EVER:** Tidak boleh menggunakan emoji apa pun dalam penulisan draf presentasi, dokumen perencanaan, log, commit message, maupun catatan briefing.
- **Surgical Changes:** Hanya memodifikasi dan membuat file yang relevan dengan tugas ini.

---

### Task 1: Manajemen File Visualisasi (Sinkronisasi Visualisasi 50 Epoch)

**Files:**
- Modify (Rename): `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train` -> `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train_1epoch_backup`
- Create (Copy): `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train` (salinan dari `fruit_detection_runs/train`)

**Interfaces:**
- Consumes: File visualisasi 50-epoch dari folder `fruit_detection_runs/train`
- Produces: Folder `fruit_detection_runs_backup/train` yang disinkronisasi berisi visualisasi 50-epoch lengkap

- [ ] **Step 1: Ganti nama folder backup lama**

Jalankan perintah untuk mengganti nama folder:
```bash
mv "/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train" "/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train_1epoch_backup"
```
Verifikasi bahwa folder lama berhasil diganti namanya.

- [ ] **Step 2: Salin visualisasi 50 epoch ke folder backup utama**

Jalankan perintah penyalinan rekursif:
```bash
cp -r "/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs/train" "/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train"
```

- [ ] **Step 3: Verifikasi hasil salinan**

Jalankan perintah untuk memverifikasi file csv hasil training:
```bash
head -n 5 "/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train/results.csv"
```
Expected: File `results.csv` terbaca dan baris datanya lebih dari 50 baris (mencerminkan hasil 50 epoch).

- [ ] **Step 4: Commit perubahan file manajemen**

Jalankan perintah commit git:
```bash
git add fruit_detection_runs_backup
git commit -m "chore: archive outdated runs and copy 50-epoch training results to backup"
```

---

### Task 2: Membuat File Draf Presentasi (fruit_detection_ppt_draft.md)

**Files:**
- Create: `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_ppt_draft.md`

**Interfaces:**
- Consumes: Metrik riil 50-epoch dan detail model YOLOv8s
- Produces: File markdown berisi draf materi PPT lengkap dalam Bahasa Indonesia

- [ ] **Step 1: Tulis file draf presentasi**

Tulis konten slide draf presentasi berikut ke dalam file `/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_ppt_draft.md`:

```markdown
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
  - **Referensi Visual:** Lihat grafik distribusi label di `labels.jpg`.

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
  - **Referensi Visual:** Gunakan grafik perkembangan belajar di `results.png`.

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
  - **Referensi Visual:** Tampilkan grafik matriks kesalahan di `confusion_matrix_normalized.png` dan grafik `BoxPR_curve.png`.

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
  - **Referensi Visual:** Tampilkan tangkapan layar prediksi deteksi pada validasi `val_batch0_pred.jpg`.

---

## Slide 9: Kesimpulan & Rencana ke Depan
* **Judul Slide:** Kesimpulan & Rencana Tindak Lanjut
* **Poin Slide:**
  - **Kesimpulan:** Model YOLOv8s berhasil dilatih secara aman tanpa kendala hardware beku (system crash). Kecepatan pemrosesan frame (59 FPS) sudah melampaui standar aplikasi real-time umum.
  - **Peluang Optimasi:**
    - Perbaikan dataset anggur (menambah variasi data anggur yang saling bertumpuk).
    - Mencoba teknik konversi model ke format ONNX atau TensorRT untuk meningkatkan efisiensi komputasi pada perangkat edge yang lebih kecil.
```
```
Lakukan pembuatan file ini di workspace.

- [ ] **Step 2: Commit file draf presentasi**

Jalankan perintah commit git:
```bash
git add fruit_detection_ppt_draft.md
git commit -m "feat: create descriptive slides draft document for PPT presentation"
```

---

### Task 3: Pembaruan Berkas Briefing

**Files:**
- Create/Modify: `/home/tgrwjya/Documents/Briefing/fruit_detection_ppt_briefing.md`

**Interfaces:**
- Consumes: Path visualisasi dan path draf PPT
- Produces: Catatan panduan penggunaan materi PPT untuk pengguna

- [ ] **Step 1: Tulis catatan briefing**

Buat file baru di `/home/tgrwjya/Documents/Briefing/fruit_detection_ppt_briefing.md` dengan konten panduan berikut:

```markdown
# Briefing Panduan Presentasi: Proyek Deteksi Buah YOLOv8s

Panduan ini berisi ringkasan aset visual dan bahan presentasi yang telah disinkronisasikan ke hasil akhir 50 epoch untuk mempermudah penyusunan PowerPoint (PPT).

## 1. Lokasi Aset Visualisasi (50 Epoch)
Seluruh visualisasi grafik evaluasi model hasil akhir 50 epoch dapat diakses langsung pada direktori lokal:
`/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_runs_backup/train/`

Daftar gambar penting untuk PPT:
- `results.png`: Grafik kurva loss training dan validasi serta kenaikan mAP. Sangat cocok diletakkan pada Slide Metrik Hasil Akhir.
- `confusion_matrix_normalized.png`: Matriks klasifikasi untuk menjelaskan buah apa yang paling sering dikenali dengan benar (Semangka) dan kelas yang sering terlewat/salah (Anggur). Cocok diletakkan pada Slide Analisis Kelas.
- `BoxPR_curve.png`: Kurva Precision-Recall per kelas buah.
- `val_batch0_pred.jpg`: Contoh gambar hasil prediksi deteksi model pada kumpulan data uji/validasi. Cocok diletakkan pada Slide Uji Coba Kamera.

## 2. Lokasi Bahan Materi Slide
Draf materi presentasi tertulis dalam Bahasa Indonesia yang sederhana dan teknis tanpa kata-kata kaku khas generator AI standar tersedia pada file:
`/home/tgrwjya/Documents/Uni/Semester 6/AI_ML/CV/Project_Work/Fruits-detection/fruit_detection_ppt_draft.md`

## 3. Catatan Penting
- **Perubahan Rasio Split Data (60:27:13):** Ingatkan dalam presentasi bahwa jumlah data train dipangkas dari ~7k menjadi 2.000 gambar secara seimbang agar GPU laptop tidak hang. Evaluasi tetap kokoh karena ukuran data validasi dipertahankan di angka 914 gambar (~27% dari total data aktif).
- **Incremental Loop:** Konsep training per 5 epoch dengan jeda 30 detik cooldown sangat krusial dipaparkan sebagai solusi rekayasa (engineering solution) terhadap keterbatasan perangkat keras.
```

- [ ] **Step 2: Simpan berkas briefing**

Pastikan file tersimpan dengan benar di direktori `/home/tgrwjya/Documents/Briefing/fruit_detection_ppt_briefing.md`.
