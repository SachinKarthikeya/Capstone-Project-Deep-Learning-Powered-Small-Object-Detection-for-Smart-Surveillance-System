import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WeaponSSDDataset
from model_3 import SSD
from anchors import generate_anchors, compute_iou
from ssd_loss import ssd_loss


# =========================
# COLLATE FUNCTION
# =========================
def ssd_collate_fn(batch):
    images, boxes, labels = [], [], []
    for img, bxs, lbls in batch:
        images.append(img)
        boxes.append(bxs)
        labels.append(lbls)
    images = torch.stack(images, 0)
    return images, boxes, labels


# =========================
# DATASET & DATALOADER
# =========================
dataset = WeaponSSDDataset(
    ann_path="/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-knives-detection-coco/SOD-SSS-5-COCO/valid/_annotations.coco.json",
    img_dir="/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-knives-detection-coco/SOD-SSS-5-COCO/valid",
)

EPOCHS = 30
BATCH_SIZE = 4
LR = 1e-4

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=ssd_collate_fn
)


# =========================
# DEVICE, MODEL, OPTIMIZER
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SSD(num_classes=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

anchors = generate_anchors().to(device)


# =========================
# TRAINING LOOP
# =========================
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for images, gt_boxes_list, gt_labels_list in loader:
        images = images.to(device)

        cls_preds, loc_preds = model(images)

        batch_loss = torch.tensor(0.0, device=device)

        for i in range(images.size(0)):
            gt_boxes = gt_boxes_list[i].to(device)
            gt_labels = gt_labels_list[i].to(device)

            # safety: no objects in image
            if gt_boxes.numel() == 0:
                continue

            # -------------------------
            # ANCHOR MATCHING
            # -------------------------
            iou_matrix = compute_iou(anchors, gt_boxes)
            best_iou, best_gt_idx = iou_matrix.max(dim=1)

            positive_mask = best_iou >= 0.5

            # -------------------------
            # CLASS TARGETS
            # -------------------------
            cls_targets = torch.zeros(
                anchors.size(0),
                dtype=torch.long,
                device=device
            )

            cls_targets[positive_mask] = gt_labels[
                best_gt_idx[positive_mask]
            ]

            # -------------------------
            # LOCALIZATION TARGETS
            # -------------------------
            matched_gt_boxes = gt_boxes[best_gt_idx]

            tx = (matched_gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
            ty = (matched_gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
            tw = torch.log(matched_gt_boxes[:, 2] / anchors[:, 2])
            th = torch.log(matched_gt_boxes[:, 3] / anchors[:, 3])

            loc_targets = torch.stack([tx, ty, tw, th], dim=1)

            # -------------------------
            # LOSS
            # -------------------------
            loss, _, _ = ssd_loss(
                cls_preds[i],
                loc_preds[i],
                cls_targets,
                loc_targets,
                positive_mask
            )

            batch_loss = batch_loss + loss

        # normalize by batch size
        batch_loss = batch_loss / images.size(0)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}")


# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "mobilenet_ssd_model.pth")
print("Training complete..........")
print("Model saved as mobilenet_ssd_model.pth")