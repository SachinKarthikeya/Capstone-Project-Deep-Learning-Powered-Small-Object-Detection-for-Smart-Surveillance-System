import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset


class WeaponSSDDataset(Dataset):
    def __init__(self, csv_path, img_dir, class_map=None):

        self.img_dir = img_dir
        self.class_map = class_map

        self.df = pd.read_csv(csv_path)
        self.images = self.df["filename"].unique()


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        file_name = self.images[idx]
        img_path = f"{self.img_dir}/{file_name}"

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Image not loaded")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        boxes = []
        labels = []

        image_rows = self.df[self.df["filename"] == file_name]

        for _, row in image_rows.iterrows():
            class_name = row["class"]

            # skip unwanted class if needed
            if class_name not in self.class_map:
                continue

            label = self.class_map[class_name]

            xmin = row["xmin"]
            ymin = row["ymin"]
            xmax = row["xmax"]
            ymax = row["ymax"]

            bw = xmax - xmin
            bh = ymax - ymin
            cx = (xmin + bw / 2) / w
            cy = (ymin + bh / 2) / h
            bw = bw / w
            bh = bh / h

            boxes.append([cx, cy, bw, bh])
            labels.append(label)

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return image, boxes, labels
    

dataset = WeaponSSDDataset(
    csv_path="/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-knives-detection-csv/SOD-SSS-5-CSV/valid/_annotations.csv",
    img_dir="/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-knives-detection-csv/SOD-SSS-5-CSV/valid",
    class_map={"Weapons": 1}
)

img, boxes, labels = dataset[0]

print(img.shape)   # [3, H, W]
print(boxes)       # [[cx, cy, w, h]]
print(labels)      # [1 / 2 / 3]