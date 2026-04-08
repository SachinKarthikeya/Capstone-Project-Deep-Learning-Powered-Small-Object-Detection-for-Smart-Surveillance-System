import json
import os
from datetime import datetime
import cv2

ALERTS_JSON = "alerts/alerts.json"
IMAGES_DIR = "alerts/images"

def trigger_alert(camera_id, frame, weapon_type, confidence):
    # Ensure folders exist (SAFE)
    if not os.path.exists("alerts"):
        os.mkdir("alerts")
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)

    timestamp = datetime.now()
    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    file_time = timestamp.strftime("%Y%m%d_%H%M%S")

    image_name = f"cam{camera_id}_{file_time}.jpg"
    image_path = os.path.join(IMAGES_DIR, image_name)

    # Save image
    cv2.imwrite(image_path, frame)

    alert_data = {
        "camera_id": camera_id,
        "weapon_type": weapon_type,
        "confidence": confidence,
        "time": time_str,
        "image_path": image_path
    }

    # Load existing alerts
    if os.path.exists(ALERTS_JSON):
        with open(ALERTS_JSON, "r") as f:
            alerts = json.load(f)
    else:
        alerts = []

    alerts.append(alert_data)

    with open(ALERTS_JSON, "w") as f:
        json.dump(alerts, f, indent=4)

    print(f"🚨 ALERT SAVED | Camera {camera_id} | {weapon_type}")
