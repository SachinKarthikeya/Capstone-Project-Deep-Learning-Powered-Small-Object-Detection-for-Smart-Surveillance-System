from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import json
import os
from fastapi.responses import JSONResponse
from alerts import trigger_alert
import random
from detector import SSDWeaponDetector
import cv2

from camera import Camera
detector = SSDWeaponDetector()

from fastapi.staticfiles import StaticFiles
app = FastAPI()
app.mount("/alerts", StaticFiles(directory="alerts"), name="alerts")

templates = Jinja2Templates(directory="templates")

# 🔹 Multiple cameras
cameras = {
    0: Camera(0),                
}


def generate_frames(cam_id):
    cam = cameras[cam_id]
    frame_count = 0

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        frame_count += 1

        # run detection every 5 frames
        if frame_count % 5 == 0:
            frame, detections = detector.detect(frame)

            if detections:
                for d in detections:
                    trigger_alert(
                        camera_id=cam_id,
                        frame=frame,
                        weapon_type=d["weapon_type"],
                        confidence=float(d["confidence"])
                    )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.get("/video/{cam_id}")
def video_feed(cam_id: int):
    return StreamingResponse(
        generate_frames(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
@app.get("/alerts")
def get_alerts():
    alerts_file = "alerts/alerts.json"

    if not os.path.exists(alerts_file):
        return JSONResponse(content=[])

    with open(alerts_file, "r") as f:
        data = json.load(f)

    return JSONResponse(content=data)