# config.py
MODEL_PATH = "mobilenet_ssd_model.pth"
CONFIDENCE_THRESHOLD = 0.4

# For YOLO model
WEAPON_CLASSES = ["knife", "gun"]

# For SSD model
NUM_CLASSES = 4
ANCHORS_PER_CELL = 16
CLASS_MAPS = {
        1: "Weapons",
    }