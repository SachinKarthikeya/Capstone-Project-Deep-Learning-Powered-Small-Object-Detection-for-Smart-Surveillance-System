import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

knife_model11 = YOLO('Yolov11/knives_best1.pt')
knife_model12 = YOLO('Yolov12/knives_best2.pt')
knife_model26 = YOLO('Yolov26/knives_best3.pt')
gun_model11 = YOLO('guns_best1.pt')
gun_model12 = YOLO('guns_best2.pt')
gun_model26 = YOLO('guns_best3.pt')

def detect_knife_yolov11(image):
    results = knife_model11(image)  
    return results

def detect_knife_yolov12(image):
    results = knife_model12(image)
    return results

def detect_knife_yolov26(image):
    results = knife_model26(image)
    return results

def detect_gun_yolov11(image):
    results = gun_model11(image)  
    return results

def detect_gun_yolov12(image):
    results = gun_model12(image)
    return results

def detect_gun_yolov26(image):
    results = gun_model26(image)
    return results

def main():
    st.set_page_config(page_title="Surveillance Dashboard")
    st.title("Weapon Detection for Smart Surveillance Systems")
    st.write("This dashboard shows the comparative analysis of YOLO models for weapon detection.")

    option = st.selectbox("Select a YOLO Model", ("YOLOv11-Knife", "YOLOv12-Knife", "YOLOv26-Knife", "YOLOv11-Gun", "YOLOv12-Gun", "YOLOv26-Gun"))

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        results = None
        if option == "YOLOv11-Knife":
            st.subheader("YOLOv11 Knife Detection")
            results = detect_knife_yolov11(image)

        elif option == "YOLOv12-Knife":
            st.subheader("YOLOv12 Knife Detection")
            results = detect_knife_yolov12(image)

        elif option == "YOLOv26-Knife":
            st.subheader("YOLOv26 Knife Detection")
            results = detect_knife_yolov26(image)

        elif option == "YOLOv11-Gun":
            st.subheader("YOLOv11 Gun Detection")
            results = detect_gun_yolov11(image)

        elif option == "YOLOv12-Gun":
            st.subheader("YOLOv12 Gun Detection")
            results = detect_gun_yolov12(image)

        elif option == "YOLOv26-Gun":
            st.subheader("YOLOv26 Gun Detection")
            results = detect_gun_yolov26(image)

        result_img = results[0].plot()

        st.image(result_img, caption="Detection Result", use_column_width=True)

if __name__ == "__main__":
    main()