import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dirs, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    output_dir = Path(output_dir)

    all_classes = set()

    for source in source_dirs:
        source_path = Path(source)
        for class_dir in source_path.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            all_classes.add(class_name)
            images = list(class_dir.glob("*.jpg"))

            random.shuffle(images)
            val_count = int(len(images) * val_ratio)
            val_images = images[:val_count]
            train_images = images[val_count:]

            train_target = output_dir / "train" / class_name
            val_target = output_dir / "val" / class_name
            train_target.mkdir(parents=True, exist_ok=True)
            val_target.mkdir(parents=True, exist_ok=True)

            for img in train_images:
                shutil.copy(img, train_target / img.name)
            for img in val_images:
                shutil.copy(img, val_target / img.name)

    print(f"✅ Split complete. Classes: {sorted(all_classes)}")

if __name__ == "__main__":
    source_dirs = [
        "dataset/Internal",
        "dataset/External"
    ]
    output_dir = "dataset_split"
    split_dataset(source_dirs, output_dir)
