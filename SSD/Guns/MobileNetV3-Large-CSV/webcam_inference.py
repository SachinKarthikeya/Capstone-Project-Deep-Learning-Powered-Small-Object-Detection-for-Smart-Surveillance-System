import torch
import cv2

from model import SSD
from anchors import generate_anchors


# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = {"pistol": 1, "rifle": 2, "shotgun": 3}


# =========================
# DEVICE & MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SSD(num_classes=4).to(device)
model.load_state_dict(torch.load("mobilenet_ssd_model_1.pth", map_location=device))
model.eval()

anchors = generate_anchors().to(device)
print("Anchors loaded:", anchors.shape)


# =========================
# BOX DECODER
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


# =========================
# NMS
# =========================
def nms(boxes, scores, threshold=0.5):
    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())

        if idxs.numel() == 1:
            break

        cur_box = boxes[i]
        other_boxes = boxes[idxs[1:]]

        x1 = torch.max(cur_box[0], other_boxes[:, 0])
        y1 = torch.max(cur_box[1], other_boxes[:, 1])
        x2 = torch.min(cur_box[2], other_boxes[:, 2])
        y2 = torch.min(cur_box[3], other_boxes[:, 3])

        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        area1 = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
        area2 = (other_boxes[:, 2] - other_boxes[:, 0]) * (
            other_boxes[:, 3] - other_boxes[:, 1]
        )

        iou = inter / (area1 + area2 - inter)
        idxs = idxs[1:][iou < threshold]

    return keep


# =========================
# WEBCAM START
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise ValueError("Webcam not accessible")

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))

    # =========================
    # MODEL INPUT
    # =========================
    tensor_img = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
    tensor_img = tensor_img.unsqueeze(0).to(device)

    # =========================
    # INFERENCE
    # =========================
    with torch.no_grad():
        cls_preds, loc_preds = model(tensor_img)

    cls_preds = cls_preds[0]
    loc_preds = loc_preds[0]

    probs = torch.softmax(cls_preds, dim=1)

    # =========================
    # POST PROCESS
    # =========================
    boxes = decode_boxes(anchors, loc_preds)
    boxes = boxes.clamp(0.0, 1.0)

    obj_probs = probs[:, 1:]
    scores_all, labels_all = obj_probs.max(dim=1)
    labels_all = labels_all + 1

    scores, idxs = scores_all.topk(1)

    boxes = boxes[idxs]
    labels = labels_all[idxs]

    keep = nms(boxes, scores)

    # =========================
    # DRAW RESULTS
    # =========================
    h, w, _ = frame.shape

    for i in keep:
        x1, y1, x2, y2 = boxes[i]

        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        cls_id = int(labels[i].item())
        conf = float(scores[i].item())

        if cls_id not in CLASS_NAMES:
            continue

        label_text = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # =========================
    # SHOW FRAME
    # =========================
    cv2.imshow("Weapon Detection (Webcam)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()