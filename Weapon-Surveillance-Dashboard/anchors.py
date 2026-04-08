import torch


FEATURE_MAP_SIZE = 20
IMAGE_SIZE = 640

SCALES = [0.05, 0.1, 0.2, 0.35]
ASPECT_RATIOS = [0.5, 1.0, 2.0, 3.0]


def generate_anchors():

    anchors = []

    for i in range(FEATURE_MAP_SIZE):
        for j in range(FEATURE_MAP_SIZE):

            cx = (j + 0.5) / FEATURE_MAP_SIZE
            cy = (i + 0.5) / FEATURE_MAP_SIZE

            for scale in SCALES:
                for ratio in ASPECT_RATIOS:

                    w = scale * (ratio ** 0.5)
                    h = scale / (ratio ** 0.5)

                    anchors.append([cx, cy, w, h])

    return torch.tensor(anchors, dtype=torch.float32)


def build_all_anchors():

    return generate_anchors()
