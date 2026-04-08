# detector.py

import torch
import cv2
import numpy as np
from torchvision.ops import nms

import config
from anchors import build_all_anchors
from model_2 import SSD   # ✅ USE TRAINING MODEL


class SSDWeaponDetector:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ SAME MODEL AS TRAINING
        self.model = SSD(num_classes=config.NUM_CLASSES).to(self.device)

        # ✅ Load trained weights
        state_dict = torch.load(
            config.MODEL_PATH,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # ✅ Same anchors
        self.anchors = build_all_anchors().to(self.device)
        print("Anchors shape:", self.anchors.shape)

        self.names = ["background"] + config.WEAPON_CLASSES


    def decode_boxes(self, bbox_preds, anchors):

        cx = anchors[:, 0]
        cy = anchors[:, 1]
        w = anchors[:, 2]
        h = anchors[:, 3]

        pred_cx = bbox_preds[:, 0] * w + cx
        pred_cy = bbox_preds[:, 1] * h + cy

        pred_w = torch.exp(bbox_preds[:, 2]) * w
        pred_h = torch.exp(bbox_preds[:, 3]) * h

        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2

        return torch.stack([x1, y1, x2, y2], dim=1)


    def preprocess(self, frame):

        # Same as training
        img = cv2.resize(frame, (640, 640))

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img.unsqueeze(0)


    def detect(self, frame):

        h, w, _ = frame.shape

        img = self.preprocess(frame).to(self.device)

        with torch.no_grad():
            cls_logits, bbox_preds = self.model(img)


        probs = torch.softmax(cls_logits[0], dim=-1)

        scores, labels = torch.max(probs[:, 1:], dim=-1)
        labels += 1


        boxes = self.decode_boxes(
            bbox_preds[0],
            self.anchors
        )


        mask = scores > config.CONFIDENCE_THRESHOLD

        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]


        keep = nms(boxes, scores, 0.45)

        detections = []
        annotated = frame.copy()


        for i in keep:

            i = i.item()

            box = boxes[i].cpu().numpy()

            x1, y1, x2, y2 = (box * [w, h, w, h]).astype(int)

            conf = scores[i].item()

            cls_name = self.names[labels[i].item()]


            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"{cls_name} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


            detections.append({
                "weapon_type": cls_name,
                "confidence": conf
            })


        return annotated, detections

