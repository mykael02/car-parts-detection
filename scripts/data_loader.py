from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
import os

def get_data_set_structure(data_dir: Path) -> dict:
    structure = {}
    for category in ["Internal", "External"]:
        category_path = data_dir / category
        structure[category] = {}
        for part_folder in category_path.iterdir():
            if part_folder.is_dir():
                image_count = len(list(part_folder.glob("*.jpg")))
                structure[category][part_folder.name] = image_count
    return structure

def load_images_from_part(part_path: Path, img_size: Tuple[int, int]=(224, 224)) -> List[np.ndarray]:
    images = []
    for img_path in part_path.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
    return images

def list_all_parts(data_dir: Path) -> List[Path]:
    part_folders = []
    for category in ["Internal", "External"]:
        category_path = data_dir / category
        part_folders.extend([p for p in category_path.iterdir() if p.is_dir()])
    return part_folders