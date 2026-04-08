import json
import cv2
import torch
from torch.utils.data import Dataset


class WeaponSSDDataset(Dataset):
    def __init__(self, ann_path, img_dir):
        self.img_dir = img_dir

        with open(ann_path, "r") as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        img_path = f"{self.img_dir}/{file_name}"
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError("Image not loaded")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        boxes = []
        labels = []

        for ann in self.annotations:
            if ann["image_id"] == image_id:
                cat_id = ann["category_id"]

                # Weapons (0) ni skip
                if cat_id == 0:
                    continue

                x, y, bw, bh = ann["bbox"]

                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                bw = bw / w
                bh = bh / h

                boxes.append([cx, cy, bw, bh])
                labels.append(cat_id)

        # image → tensor
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return image, boxes, labels


dataset = WeaponSSDDataset(
    ann_path="/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-guns-detection-coco/CGI-Weapon-Dataset-2/valid/_annotations.coco.json",
    img_dir="/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-guns-detection-coco/CGI-Weapon-Dataset-2/valid"
)

img, boxes, labels = dataset[0]

print(img.shape)     # [3, 640, 640]
print(boxes)         # [[cx, cy, w, h]]
print(labels)        # [1 / 2 / 3]