import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from model_2 import SSD
from anchors import generate_anchors

# =========================
# CONFIG
# =========================
ANN_PATH = "/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-knives-detection-coco/SOD-SSS-5-COCO/valid/_annotations.coco.json"
IMG_DIR = "/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-knives-detection-coco/SOD-SSS-5-COCO/valid"

IOU_THRESH = 0.5

CLASS_NAMES = {1: "Weapons"}


# =========================
# DEVICE & MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SSD(num_classes=4).to(device)
model.load_state_dict(torch.load("efficientnet_ssd_model.pth", map_location=device))
model.eval()

anchors = generate_anchors().to(device)


# =========================
# UTILS
# =========================
def decode_boxes(anchors, loc_preds):
    cx = anchors[:, 0] + loc_preds[:, 0] * anchors[:, 2]
    cy = anchors[:, 1] + loc_preds[:, 1] * anchors[:, 3]
    w = anchors[:, 2] * torch.exp(loc_preds[:, 2])
    h = anchors[:, 3] * torch.exp(loc_preds[:, 3])

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_ap(tp, fp, scores, num_gt):
    idxs = np.argsort(-np.array(scores))
    tp = np.array(tp)[idxs]
    fp = np.array(fp)[idxs]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / (num_gt + 1e-6)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])

    return ap


# =========================
# LOAD COCO
# =========================
with open(ANN_PATH, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

img_to_anns = {}
for ann in annotations:
    img_to_anns.setdefault(ann["image_id"], []).append(ann)


# =========================
# METRIC COUNTERS
# =========================
TP = 0
FP = 0
FN = 0

iou_thresholds = np.arange(0.5, 0.96, 0.05)
ap_data = {
    t: {"scores": [], "tp": [], "fp": [], "num_gt": 0}
    for t in iou_thresholds
}


# =========================
# EVALUATION LOOP
# =========================
for img_info in tqdm(images, desc="Evaluating"):
    image_id = img_info["id"]
    file_name = img_info["file_name"]

    img_path = f"{IMG_DIR}/{file_name}"
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))

    tensor_img = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
    tensor_img = tensor_img.unsqueeze(0).to(device)

    with torch.no_grad():
        cls_preds, loc_preds = model(tensor_img)

    cls_preds = cls_preds[0]
    loc_preds = loc_preds[0]

    probs = torch.softmax(cls_preds, dim=1)
    obj_probs = probs[:, 1:]
    scores_all, labels_all = obj_probs.max(dim=1)
    labels_all = labels_all + 1

    scores, idxs = scores_all.topk(1)

    pred_boxes = decode_boxes(anchors, loc_preds)
    pred_boxes = pred_boxes.clamp(0.0, 1.0)
    pred_boxes = pred_boxes[idxs].cpu().numpy()
    pred_labels = labels_all[idxs].cpu().numpy()
    pred_scores = scores.cpu().numpy()

    gt_anns = img_to_anns.get(image_id, [])
    gt_boxes = []
    gt_labels = []

    for ann in gt_anns:
        if ann["category_id"] == 0:
            continue

        x, y, bw, bh = ann["bbox"]
        gt_boxes.append([x / w, y / h, (x + bw) / w, (y + bh) / h])
        gt_labels.append(ann["category_id"])

    matched = set()

    for t in iou_thresholds:
        ap_data[t]["num_gt"] += len(gt_boxes)

    for pb, pl, sc in zip(pred_boxes, pred_labels, pred_scores):
        best_iou = 0
        best_j = -1

        for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= IOU_THRESH and best_j not in matched and pl == gt_labels[best_j]:
            TP += 1
            matched.add(best_j)
        else:
            FP += 1

        for t in iou_thresholds:
            if best_iou >= t and best_j != -1 and pl == gt_labels[best_j]:
                ap_data[t]["tp"].append(1)
                ap_data[t]["fp"].append(0)
            else:
                ap_data[t]["tp"].append(0)
                ap_data[t]["fp"].append(1)

            ap_data[t]["scores"].append(sc)

    FN += len(gt_boxes) - len(matched)


# =========================
# METRICS
# =========================
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

aps = []
for t in iou_thresholds:
    data = ap_data[t]
    aps.append(
        compute_ap(
            data["tp"],
            data["fp"],
            data["scores"],
            data["num_gt"],
        )
    )

map50 = aps[0]
map5095 = np.mean(aps)

print("\n===== EVALUATION RESULTS =====")
print(f"True Positives : {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-score       : {f1:.4f}")
print(f"mAP@50       : {map50:.4f}")
print(f"mAP@50-95  : {map5095:.4f}")