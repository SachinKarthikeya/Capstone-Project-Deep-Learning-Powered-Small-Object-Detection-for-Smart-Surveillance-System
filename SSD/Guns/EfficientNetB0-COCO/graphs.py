import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model import SSD
from anchors import generate_anchors

# =========================
# CONFIG
# =========================
ANN_PATH = "/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-guns-detection-coco/CGI-Weapon-Dataset-2/valid/_annotations.coco.json"
IMG_DIR = "/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-guns-detection-coco/CGI-Weapon-Dataset-2/valid"

IOU_THRESH = 0.5
CONF_THRESH = 0.05
NMS_THRESH = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MODEL
# =========================
model = SSD(num_classes=4).to(device)
model.load_state_dict(torch.load("efficientnet_ssd_model_2.pth", map_location=device))
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
# METRICS STORAGE
# =========================
TP, FP, FN = 0, 0, 0

iou_thresholds = np.arange(0.5, 0.96, 0.05)
ap_data = {t: {"scores": [], "tp": [], "fp": [], "num_gt": 0} for t in iou_thresholds}

conf_matrix = np.zeros((2, 2))  # binary

# =========================
# EVALUATION LOOP
# =========================
for img_info in tqdm(images, desc="Evaluating"):
    image_id = img_info["id"]
    file_name = img_info["file_name"]

    img = cv2.imread(f"{IMG_DIR}/{file_name}")
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

    # 🔥 KEEP MULTIPLE DETECTIONS
    keep = scores_all > CONF_THRESH
    scores = scores_all[keep].cpu().numpy()
    labels = labels_all[keep].cpu().numpy()

    pred_boxes = decode_boxes(anchors, loc_preds)
    pred_boxes = pred_boxes.clamp(0.0, 1.0)[keep].cpu().numpy()

    # 🔥 APPLY NMS
    if len(pred_boxes) > 0:
        keep_idx = nms(pred_boxes, scores, NMS_THRESH)
        pred_boxes = pred_boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

    # GT
    gt_anns = img_to_anns.get(image_id, [])
    gt_boxes, gt_labels = [], []

    for ann in gt_anns:
        if ann["category_id"] == 0:
            continue

        x, y, bw, bh = ann["bbox"]
        gt_boxes.append([x/w, y/h, (x+bw)/w, (y+bh)/h])
        gt_labels.append(ann["category_id"])

    matched = set()

    for t in iou_thresholds:
        ap_data[t]["num_gt"] += len(gt_boxes)

    for pb, pl, sc in zip(pred_boxes, labels, scores):
        best_iou = 0
        best_j = -1

        for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= IOU_THRESH and best_j not in matched and pl == gt_labels[best_j]:
            TP += 1
            conf_matrix[1][1] += 1
            matched.add(best_j)
        else:
            FP += 1
            conf_matrix[0][1] += 1

        for t in iou_thresholds:
            if best_iou >= t and best_j != -1 and pl == gt_labels[best_j]:
                ap_data[t]["tp"].append(1)
                ap_data[t]["fp"].append(0)
            else:
                ap_data[t]["tp"].append(0)
                ap_data[t]["fp"].append(1)

            ap_data[t]["scores"].append(sc)

    fn_img = len(gt_boxes) - len(matched)
    FN += fn_img
    conf_matrix[1][0] += fn_img

# =========================
# METRICS
# =========================
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

# =========================
# PLOTS (YOLO STYLE)
# =========================
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# PR CURVE
t = 0.5
data = ap_data[t]

scores = np.array(data["scores"])
tp = np.array(data["tp"])
fp = np.array(data["fp"])

idxs = np.argsort(-scores)
tp = tp[idxs]
fp = fp[idxs]

tp_cum = np.cumsum(tp)
fp_cum = np.cumsum(fp)

recalls = tp_cum / (data["num_gt"] + 1e-6)
precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

axs[0, 0].plot(recalls, precisions)
axs[0, 0].set_title("PR Curve")
axs[0, 0].set_xlabel("Recall")
axs[0, 0].set_ylabel("Precision")

# METRICS vs CONF
thresholds = np.linspace(0, 1, 100)
p_list, r_list, f1_list = [], [], []

for thr in thresholds:
    mask = scores >= thr
    tp_thr = tp[mask].sum()
    fp_thr = fp[mask].sum()
    fn_thr = data["num_gt"] - tp_thr

    p = tp_thr / (tp_thr + fp_thr + 1e-6)
    r = tp_thr / (tp_thr + fn_thr + 1e-6)
    f = 2 * p * r / (p + r + 1e-6)

    p_list.append(p)
    r_list.append(r)
    f1_list.append(f)

axs[0, 1].plot(thresholds, p_list, label="P")
axs[0, 1].plot(thresholds, r_list, label="R")
axs[0, 1].plot(thresholds, f1_list, label="F1")
axs[0, 1].legend()
axs[0, 1].set_title("Metrics vs Confidence")

# CONFUSION MATRIX
sns.heatmap(conf_matrix, annot=True, fmt=".0f", ax=axs[1, 0])
axs[1, 0].set_title("Confusion Matrix")

# TEXT METRICS
axs[1, 1].axis("off")
axs[1, 1].text(0.1, 0.5,
               f"Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}",
               fontsize=12)

plt.tight_layout()
plt.savefig("results2.png")  # 🔥 YOLO-style output
plt.show()

# =========================
# PRINT
# =========================
print("\n===== RESULTS =====")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")