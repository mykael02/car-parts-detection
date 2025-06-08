import os
import shutil
import random
from pathlib import Path

def split_dataset(base_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)

    part_types = [p for p in base_dir.glob("*") if p.is_dir()]

    for part_path in part_types:
        all_images = list(part_path.glob("*.jpg"))
        random.shuffle(all_images)

        val_size = int(len(all_images) * val_ratio)
        val_images = all_images[:val_size]
        train_images = all_images[val_size:]

        class_name = part_path.name
        train_class_dir = output_dir / "train" / class_name
        val_class_dir = output_dir / "val" / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy(img, train_class_dir / img.name)
        for img in val_images:
            shutil.copy(img, val_class_dir / img.name)

    print("Dataset split complete.")

if __name__ == "__main__":
    split_dataset("dataset/Internal", "dataset_split")
    split_dataset("dataset/External", "dataset_split")
