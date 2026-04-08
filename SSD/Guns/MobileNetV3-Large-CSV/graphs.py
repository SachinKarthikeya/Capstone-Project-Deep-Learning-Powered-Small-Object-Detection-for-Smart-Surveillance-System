import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import copy

from model import SSD
from anchors import generate_anchors

# =========================
# CONFIG
# =========================
ANN_PATH = "/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-guns-detection-csv/CGI-Weapon-Dataset-2/valid/_annotations.csv"
IMG_DIR = "/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-guns-detection-csv/CGI-Weapon-Dataset-2/valid"

IOU_THRESH = 0.5
CONF_THRESH = 0.05
NMS_THRESH = 0.5

CLASS_MAP = {"pistol": 1, "rifle": 2, "shotgun": 3}
CLASS_NAMES = {1: "pistol", 2: "rifle", 3: "shotgun"}
NUM_CLASSES = 4

# =========================
# DEVICE & MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SSD(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("mobilenet_ssd_model_2.pth", map_location=device))
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

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def nms(boxes, scores, threshold=0.5):
    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]

        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious < threshold]

    return keep


# =========================
# LOAD CSV
# =========================
df = pd.read_csv(ANN_PATH)
image_files = df["filename"].unique()

img_to_anns = defaultdict(list)

for _, row in df.iterrows():
    label = CLASS_MAP.get(row["class"], None)
    if label is None:
        continue

    img_to_anns[row["filename"]].append({
        "bbox": [row["xmin"], row["ymin"], row["xmax"], row["ymax"]],
        "label": label
    })


# =========================
# STORAGE
# =========================
TP, FP, FN = 0, 0, 0
all_predictions = []
all_gts = defaultdict(list)


# =========================
# EVALUATION LOOP
# =========================
for file_name in tqdm(image_files, desc="Evaluating"):
    img = cv2.imread(f"{IMG_DIR}/{file_name}")
    if img is None:
        continue

    h, w, _ = img.shape

    img_resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640))
    tensor_img = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
    tensor_img = tensor_img.unsqueeze(0).to(device)

    with torch.no_grad():
        cls_preds, loc_preds = model(tensor_img)

    cls_preds = cls_preds[0]
    loc_preds = loc_preds[0]

    probs = torch.softmax(cls_preds, dim=1)

    scores_all, labels_all = probs[:, 1:].max(dim=1)
    labels_all = labels_all + 1

    # MULTI DETECTION
    keep = scores_all > CONF_THRESH

    scores = scores_all[keep].cpu().numpy()
    labels = labels_all[keep].cpu().numpy()

    pred_boxes = decode_boxes(anchors, loc_preds)
    pred_boxes = pred_boxes.clamp(0.0, 1.0)[keep].cpu().numpy()

    # NMS
    if len(pred_boxes) > 0:
        keep_idx = nms(pred_boxes, scores, NMS_THRESH)
        pred_boxes = pred_boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

    # GT
    gt_boxes, gt_labels = [], []

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

    # STORE PREDICTIONS
    for pb, pl, sc in zip(pred_boxes, labels, scores):
        all_predictions.append({
            "image_id": file_name,
            "bbox": pb.tolist(),
            "label": int(pl),
            "score": float(sc)
        })

    matched = set()

    for pb, pl in zip(pred_boxes, labels):
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


# =========================
# AP FUNCTION
# =========================
def compute_ap(predictions, gts, iou_thresh):
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    TP, FP = [], []
    gts_copy = copy.deepcopy(gts)
    total_gts = sum(len(v) for v in gts.values())

    for pred in predictions:
        img_id = pred["image_id"]

        best_iou = 0
        best_gt = None

        for gt in gts_copy[img_id]:
            if gt["label"] != pred["label"] or gt["used"]:
                continue

            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_iou >= iou_thresh and best_gt:
            TP.append(1)
            FP.append(0)
            best_gt["used"] = True
        else:
            TP.append(0)
            FP.append(1)

    TP = np.cumsum(TP)
    FP = np.cumsum(FP)

    recalls = TP / (total_gts + 1e-6)
    precisions = TP / (TP + FP + 1e-6)

    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0
        ap += p / 11

    return ap


ap50 = compute_ap(all_predictions, copy.deepcopy(all_gts), 0.5)
aps = [compute_ap(all_predictions, copy.deepcopy(all_gts), t)
       for t in np.arange(0.5, 0.96, 0.05)]
map5095 = np.mean(aps)


# =========================
# CONFUSION MATRIX (MULTI-CLASS)
# =========================
cm = np.zeros((NUM_CLASSES, NUM_CLASSES))
gts_copy = copy.deepcopy(all_gts)

for pred in all_predictions:
    img_id = pred["image_id"]

    best_iou = 0
    best_gt = None

    for gt in gts_copy[img_id]:
        if gt["used"]:
            continue

        iou = compute_iou(pred["bbox"], gt["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_gt = gt

    if best_iou >= IOU_THRESH and best_gt:
        cm[best_gt["label"]][pred["label"]] += 1
        best_gt["used"] = True
    else:
        cm[0][pred["label"]] += 1  # FP

# =========================
# PR CURVE
# =========================
preds_sorted = sorted(all_predictions, key=lambda x: x["score"], reverse=True)

TP_list, FP_list = [], []
gts_copy = copy.deepcopy(all_gts)
total_gts = sum(len(v) for v in all_gts.values())

for pred in preds_sorted:
    img_id = pred["image_id"]

    best_iou = 0
    best_gt = None

    for gt in gts_copy[img_id]:
        if gt["label"] != pred["label"] or gt["used"]:
            continue

        iou = compute_iou(pred["bbox"], gt["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_gt = gt

    if best_iou >= IOU_THRESH and best_gt:
        TP_list.append(1)
        FP_list.append(0)
        best_gt["used"] = True
    else:
        TP_list.append(0)
        FP_list.append(1)

TP_cum = np.cumsum(TP_list)
FP_cum = np.cumsum(FP_list)

recalls = TP_cum / (total_gts + 1e-6)
precisions = TP_cum / (TP_cum + FP_cum + 1e-6)

# =========================
# CONFIDENCE CURVES
# =========================
thresholds = np.linspace(0, 1, 100)
p_list, r_list, f1_list = [], [], []

for thr in thresholds:
    filtered = [p for p in all_predictions if p["score"] >= thr]

    TP_, FP_ = 0, 0
    gts_copy = copy.deepcopy(all_gts)
    total_gts = sum(len(v) for v in all_gts.values())

    for pred in filtered:
        img_id = pred["image_id"]

        best_iou = 0
        best_gt = None

        for gt in gts_copy[img_id]:
            if gt["label"] != pred["label"] or gt["used"]:
                continue

            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_iou >= IOU_THRESH and best_gt:
            TP_ += 1
            best_gt["used"] = True
        else:
            FP_ += 1

    FN_ = total_gts - TP_

    p = TP_ / (TP_ + FP_ + 1e-6)
    r = TP_ / (TP_ + FN_ + 1e-6)
    f = 2 * p * r / (p + r + 1e-6)

    p_list.append(p)
    r_list.append(r)
    f1_list.append(f)

# =========================
# PLOTS
# =========================
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(recalls, precisions)
axs[0, 0].set_title("PR Curve")

axs[0, 1].plot(thresholds, p_list, label="P")
axs[0, 1].plot(thresholds, r_list, label="R")
axs[0, 1].plot(thresholds, f1_list, label="F1")
axs[0, 1].legend()
axs[0, 1].set_title("Metrics vs Confidence")

sns.heatmap(cm, annot=True, fmt=".0f",
            xticklabels=CLASS_NAMES.values(),
            yticklabels=CLASS_NAMES.values(),
            ax=axs[1, 0])
axs[1, 0].set_title("Confusion Matrix")

axs[1, 1].axis("off")
axs[1, 1].text(0.1, 0.5,
               f"P: {precision:.3f}\nR: {recall:.3f}\nF1: {f1:.3f}\nmAP50: {ap50:.3f}",
               fontsize=12)

plt.tight_layout()
plt.savefig("results2.png")
plt.show()

# =========================
# PRINT
# =========================
print("\n===== RESULTS =====")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"mAP@50    : {ap50:.4f}")
print(f"mAP@50-95 : {map5095:.4f}")