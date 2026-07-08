"""
Script untuk mengurangi dataset training menjadi ~2000 gambar.
Sampling dilakukan secara seimbang per kelas agar setiap buah tetap terwakili.

Dataset asli akan di-backup ke folder train_full_backup/.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ─── KONFIGURASI ───
PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_IMAGES = PROJECT_DIR / "train" / "images"
TRAIN_LABELS = PROJECT_DIR / "train" / "labels"
BACKUP_IMAGES = PROJECT_DIR / "train_full_backup" / "images"
BACKUP_LABELS = PROJECT_DIR / "train_full_backup" / "labels"

TARGET_TOTAL = 2000  # Target jumlah gambar
SEED = 42

# Nama kelas sesuai data.yaml
CLASS_NAMES = {0: "Apple", 1: "Banana", 2: "Grape", 3: "Orange", 4: "Pineapple", 5: "Watermelon"}


def get_classes_in_label(label_path):
    """Baca file label YOLO dan return set of class IDs."""
    classes = set()
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    return classes


def main():
    random.seed(SEED)

    # Kumpulkan semua file gambar
    image_files = sorted([f for f in TRAIN_IMAGES.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    total_images = len(image_files)
    print(f"📊 Total gambar training saat ini: {total_images}")

    if total_images <= TARGET_TOTAL:
        print(f"✅ Dataset sudah <= {TARGET_TOTAL} gambar. Tidak perlu dikurangi.")
        return

    # Kelompokkan gambar berdasarkan kelas dominan (kelas pertama di label)
    class_to_images = defaultdict(list)
    no_label = []

    for img_path in image_files:
        label_path = TRAIN_LABELS / (img_path.stem + ".txt")
        classes = get_classes_in_label(label_path)
        if classes:
            # Assign ke kelas pertama yang ditemukan (untuk distribusi)
            primary_class = min(classes)
            class_to_images[primary_class].append(img_path)
        else:
            no_label.append(img_path)

    # Tampilkan distribusi saat ini
    print(f"\n📋 Distribusi per kelas (sebelum sampling):")
    for cls_id in sorted(class_to_images.keys()):
        name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        print(f"   {name:12s}: {len(class_to_images[cls_id]):5d} gambar")
    if no_label:
        print(f"   {'(no label)':12s}: {len(no_label):5d} gambar")

    # Hitung sampling per kelas (proporsional, minimal 50 per kelas)
    num_classes = len(class_to_images)
    per_class_target = TARGET_TOTAL // num_classes  # ~333 per kelas untuk 6 kelas

    selected = []
    print(f"\n🎯 Target: {TARGET_TOTAL} gambar ({per_class_target} per kelas)")

    for cls_id in sorted(class_to_images.keys()):
        imgs = class_to_images[cls_id]
        n_select = min(len(imgs), per_class_target)
        sampled = random.sample(imgs, n_select)
        selected.extend(sampled)
        name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        print(f"   {name:12s}: {n_select:5d} gambar dipilih (dari {len(imgs)})")

    # Jika masih kurang dari target, tambahkan dari sisa
    remaining_needed = TARGET_TOTAL - len(selected)
    if remaining_needed > 0:
        selected_set = set(s.name for s in selected)
        all_remaining = [f for f in image_files if f.name not in selected_set]
        extra = random.sample(all_remaining, min(remaining_needed, len(all_remaining)))
        selected.extend(extra)
        print(f"   + {len(extra)} gambar tambahan untuk mencapai target")

    selected_set = set(s.name for s in selected)
    removed = [f for f in image_files if f.name not in selected_set]

    print(f"\n📊 Ringkasan:")
    print(f"   Dipertahankan : {len(selected)} gambar")
    print(f"   Dipindahkan   : {len(removed)} gambar ke backup")

    # Buat folder backup
    BACKUP_IMAGES.mkdir(parents=True, exist_ok=True)
    BACKUP_LABELS.mkdir(parents=True, exist_ok=True)

    # Pindahkan gambar yang tidak terpilih ke backup
    print(f"\n⏳ Memindahkan {len(removed)} gambar ke backup...")
    for i, img_path in enumerate(removed, 1):
        # Pindahkan gambar
        shutil.move(str(img_path), str(BACKUP_IMAGES / img_path.name))

        # Pindahkan label jika ada
        label_path = TRAIN_LABELS / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.move(str(label_path), str(BACKUP_LABELS / label_path.name))

        if i % 500 == 0:
            print(f"   ... {i}/{len(removed)} dipindahkan")

    # Verifikasi
    final_count = len(list(TRAIN_IMAGES.iterdir()))
    backup_count = len(list(BACKUP_IMAGES.iterdir()))
    print(f"\n✅ SELESAI!")
    print(f"   Train images : {final_count} gambar")
    print(f"   Backup images: {backup_count} gambar")
    print(f"   Backup folder: {BACKUP_IMAGES.parent}")
    print(f"\n💡 Untuk mengembalikan dataset penuh, jalankan:")
    print(f"   python reduce_dataset.py --restore")


def restore():
    """Kembalikan dataset dari backup ke train."""
    if not BACKUP_IMAGES.exists():
        print("❌ Folder backup tidak ditemukan!")
        return

    backup_imgs = list(BACKUP_IMAGES.iterdir())
    print(f"⏳ Mengembalikan {len(backup_imgs)} gambar dari backup...")

    for img_path in backup_imgs:
        shutil.move(str(img_path), str(TRAIN_IMAGES / img_path.name))

    if BACKUP_LABELS.exists():
        for label_path in BACKUP_LABELS.iterdir():
            shutil.move(str(label_path), str(TRAIN_LABELS / label_path.name))

    # Hapus folder backup yang sudah kosong
    shutil.rmtree(str(BACKUP_IMAGES.parent))

    final_count = len(list(TRAIN_IMAGES.iterdir()))
    print(f"✅ Dataset dikembalikan! Total: {final_count} gambar")


if __name__ == "__main__":
    import sys
    if "--restore" in sys.argv:
        restore()
    else:
        main()
