# scripts/make_binary_dataset.py
import os, shutil, random
from pathlib import Path

SRC = Path("dataset/FruitSpoilage/Train")
OUT = Path("dataset_binary")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png"}

random.seed(RANDOM_SEED)

def ensure_dirs():
    for part in ["train/fresh","train/not_fresh","val/fresh","val/not_fresh"]:
        (OUT/part).mkdir(parents=True, exist_ok=True)

def is_fresh(dirname):
    return "fresh" in dirname.lower()

def copy_split():
    ensure_dirs()
    for class_dir in sorted(os.listdir(SRC)):
        src_dir = SRC / class_dir
        if not src_dir.is_dir():
            continue
        images = [p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        if not images:
            continue
        random.shuffle(images)
        n_train = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:]
        target_train = OUT / ("train/fresh" if is_fresh(class_dir) else "train/not_fresh")
        target_val = OUT / ("val/fresh" if is_fresh(class_dir) else "val/not_fresh")
        for p in train_imgs:
            shutil.copy(p, target_train / p.name)
        for p in val_imgs:
            shutil.copy(p, target_val / p.name)
        print(f"Copied {len(train_imgs)} train and {len(val_imgs)} val from {class_dir}")

if __name__ == "__main__":
    print("Source:", SRC)
    copy_split()
    print("Done. Inspect dataset_binary/ to confirm.")
