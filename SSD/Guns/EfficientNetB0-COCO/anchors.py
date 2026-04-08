import torch


FEATURE_MAP_SIZE = 20
IMAGE_SIZE = 640

SCALES = [0.05, 0.1, 0.2, 0.35]
ASPECT_RATIOS = [0.5, 1.0, 2.0, 3.0]


def generate_anchors():
    anchors = []

    stride = IMAGE_SIZE / FEATURE_MAP_SIZE

    for i in range(FEATURE_MAP_SIZE):
        for j in range(FEATURE_MAP_SIZE):

            # center of grid cell (normalized)
            cx = (j + 0.5) / FEATURE_MAP_SIZE
            cy = (i + 0.5) / FEATURE_MAP_SIZE

            for scale in SCALES:
                for ratio in ASPECT_RATIOS:
                    w = scale * (ratio**0.5)
                    h = scale / (ratio**0.5)

                    anchors.append([cx, cy, w, h])

    return torch.tensor(anchors)


anchors = generate_anchors()

print(anchors.shape)
print(anchors[:5])


# IOU calculation
def cxcy_to_xy(boxes):
    """
    boxes: Tensor [N, 4] in (cx, cy, w, h)
    returns: Tensor [N, 4] in (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def compute_iou(boxes1, boxes2):
    """
    boxes1: [N, 4] anchors
    boxes2: [M, 4] ground truth
    returns: IoU matrix [N, M]
    """

    boxes1 = cxcy_to_xy(boxes1)
    boxes2 = cxcy_to_xy(boxes2)

    N = boxes1.size(0)
    M = boxes2.size(0)

    iou = torch.zeros(N, M)

    for i in range(N):
        for j in range(M):
            x1 = max(boxes1[i, 0], boxes2[j, 0])
            y1 = max(boxes1[i, 1], boxes2[j, 1])
            x2 = min(boxes1[i, 2], boxes2[j, 2])
            y2 = min(boxes1[i, 3], boxes2[j, 3])

            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            inter_area = inter_w * inter_h

            area1 = (boxes1[i, 2] - boxes1[i, 0]) * (boxes1[i, 3] - boxes1[i, 1])
            area2 = (boxes2[j, 2] - boxes2[j, 0]) * (boxes2[j, 3] - boxes2[j, 1])

            union = area1 + area2 - inter_area

            iou[i, j] = inter_area / union if union > 0 else 0

    return iou


anchors = generate_anchors()  # [1600, 4]

from dataset import WeaponSSDDataset

dataset = WeaponSSDDataset(
    ann_path="/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-guns-detection-coco/CGI-Weapon-Dataset-2/valid/_annotations.coco.json",
    img_dir="/Users/sachinkarthikeya/Downloads/Capstone Project/ssd-guns-detection-coco/CGI-Weapon-Dataset-2/valid"
)

_, gt_boxes, _ = dataset[0]  

iou_matrix = compute_iou(anchors, gt_boxes)

print(iou_matrix.shape)
print(iou_matrix.max())