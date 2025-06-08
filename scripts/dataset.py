import torch
from torch.utils.data import Dataset
import os
import cv2
import json

class CarPartDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        with open(annotation_file) as f:
            self.annotations = json.load(f)

    def __getitem__(self, idx):
        record = self.annotations[idx]
        img_path = os.path.join(self.image_dir, record['filename'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = torch.tensor(record['boxes'], dtype=torch.float32)
        labels = torch.tensor(record['labels'], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.annotations)
