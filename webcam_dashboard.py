import streamlit as st
import cv2
from ultralytics import YOLO
import time

weapon_model = YOLO("best.pt")  

st.set_page_config(page_title="Surveillance Dashboard", layout="wide")
st.title("Military Surveillance System - Weapon Detection")

st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.50, 0.05)

if "notifications" not in st.session_state:
    st.session_state.notifications = []

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

start_camera = st.button("Start Camera")
stop_camera = st.button("Stop Camera")

frame_placeholder = st.empty()

if start_camera:
    st.session_state.camera_running = True
    st.text("Starting live camera feed...")

if stop_camera:
    st.session_state.camera_running = False
    st.text("Camera stopped.")

cap = None

if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)

    while st.session_state.camera_running and cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam.")
            break

        detection_made = False  

        # Weapon Detection Only
        results_weapon = weapon_model.predict(
            frame,
            conf=confidence_threshold,
            verbose=False
        )

        annotated_frame = frame.copy()

        if results_weapon and results_weapon[0].boxes:

            annotated_frame = results_weapon[0].plot()

            for box in results_weapon[0].boxes:

                cls_id = int(box.cls[0])
                label = weapon_model.names[cls_id]

                st.session_state.notifications.append(
                    f"⚠️ Weapon detected: {label} at {time.strftime('%H:%M:%S')}"
                )

                detection_made = True

        frame_placeholder.image(
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

        if detection_made:
            with st.sidebar:

                st.subheader("Live Notifications")

                for note in reversed(st.session_state.notifications[-10:]):
                    st.warning(note)

    if cap:
        cap.release()
