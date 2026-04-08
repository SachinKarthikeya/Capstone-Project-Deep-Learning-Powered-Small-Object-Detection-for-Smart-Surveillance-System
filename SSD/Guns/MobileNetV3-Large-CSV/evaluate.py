import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import copy

from model import SSD
from anchors import generate_anchors


# =========================
# CONFIG
# =========================
ANN_PATH = "/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-guns-detection-csv/CGI-Weapon-Dataset-1/valid/_annotations.csv"
IMG_DIR = "/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-guns-detection-csv/CGI-Weapon-Dataset-1/valid"

IOU_THRESH = 0.5

CLASS_MAP = {"pistol": 1, "rifle": 2, "shotgun": 3}
CLASS_NAMES = {1: "pistol", 2: "rifle", 3: "shotgun"}


# =========================
# DEVICE & MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SSD(num_classes=4).to(device)  # background + Weapons
model.load_state_dict(torch.load("mobilenet_ssd_model_1.pth", map_location=device))
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


# =========================
# AP COMPUTATION
# =========================
def compute_ap(predictions, gts, iou_thresh):
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    TP = np.zeros(len(predictions))
    FP = np.zeros(len(predictions))

    total_gts = sum(len(v) for v in gts.values())

    for i, pred in enumerate(predictions):
        img_id = pred["image_id"]
        pred_box = pred["bbox"]
        pred_label = pred["label"]

        best_iou = 0
        best_gt = None

        for gt in gts[img_id]:
            if gt["label"] != pred_label or gt["used"]:
                continue

            iou = compute_iou(pred_box, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_iou >= iou_thresh and best_gt is not None:
            TP[i] = 1
            best_gt["used"] = True
        else:
            FP[i] = 1

    TP_cum = np.cumsum(TP)
    FP_cum = np.cumsum(FP)

    recalls = TP_cum / (total_gts + 1e-6)
    precisions = TP_cum / (TP_cum + FP_cum + 1e-6)

    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0
        ap += p / 11

    return ap


# =========================
# LOAD CSV ANNOTATIONS
# =========================
df = pd.read_csv(ANN_PATH)

image_files = df["filename"].unique()

img_to_anns = defaultdict(list)

for _, row in df.iterrows():
    label = CLASS_MAP.get(row["class"], None)
    if label is None:
        continue

    img_to_anns[row["filename"]].append({
        "bbox": [
            row["xmin"],
            row["ymin"],
            row["xmax"],
            row["ymax"]
        ],
        "label": label
    })


# =========================
# METRIC COUNTERS
# =========================
TP = 0
FP = 0
FN = 0

all_predictions = []
all_gts = defaultdict(list)


# =========================
# EVALUATION LOOP
# =========================
for file_name in tqdm(image_files, desc="Evaluating"):
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

    scores_all, labels_all = probs[:, 1:].max(dim=1)
    labels_all = labels_all + 1

    scores, idxs = scores_all.topk(1)

    pred_boxes = decode_boxes(anchors, loc_preds)
    pred_boxes = pred_boxes.clamp(0.0, 1.0)
    pred_boxes = pred_boxes[idxs].cpu().numpy()
    pred_labels = labels_all[idxs].cpu().numpy()

    # =========================
    # GT BOXES
    # =========================
    gt_boxes = []
    gt_labels = []

    for ann in img_to_anns[file_name]:
        x1 = ann["bbox"][0] / w
        y1 = ann["bbox"][1] / h
        x2 = ann["bbox"][2] / w
        y2 = ann["bbox"][3] / h

        gt_boxes.append([x1, y1, x2, y2])
        gt_labels.append(ann["label"])

        all_gts[file_name].append({
            "bbox": [x1, y1, x2, y2],
            "label": ann["label"],
            "used": False
        })

    # store predictions
    for pb, pl, sc in zip(pred_boxes, pred_labels, scores.cpu().numpy()):
        all_predictions.append({
            "image_id": file_name,
            "bbox": pb.tolist(),
            "label": int(pl),
            "score": float(sc)
        })

    matched = set()

    for pb, pl in zip(pred_boxes, pred_labels):
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

    FN += len(gt_boxes) - len(matched)


# =========================
# METRICS
# =========================
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

ap50 = compute_ap(all_predictions, copy.deepcopy(all_gts), 0.5)

aps = []
for t in np.arange(0.5, 0.96, 0.05):
    aps.append(compute_ap(all_predictions, copy.deepcopy(all_gts), t))

map5095 = np.mean(aps)

print("\n===== EVALUATION RESULTS =====")
print(f"True Positives : {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-score       : {f1:.4f}")
print(f"mAP@50         : {ap50:.4f}")
print(f"mAP@50-95      : {map5095:.4f}")