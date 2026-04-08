# Capstone Project - Deep Learning-Powered Small Object Detection for Smart Surveillance - A focus on Gun and Knife Identification

This repository is the result of my one-year hardwork and experimentation on my university's final year project, as a part of my curriculum. I along with my team mates have read many research papers, understood how object detection works in real-world scenarios under different conditions, searched for many datasets in the internet, annotated several images of knives and guns with different pre-processing and augmentation techniques, exported the images in different formats, trained several object detection-specific models and finally analyzed the results of each model, both numerically and graphically to understand how accurate a specific model detects a weapon in different conditions and which model is efficient in all terms for real-world scenarios. 

## Existing Systems 

There are already many existing systems related to detecting weapons using object detection models and techniques. Some of them which we researched are given below:

- Using old-version YOLO (You Only Look Once) models like YOLOv5, YOLOv7, YOLOv8
- Introducing Iterative Model Generation Framework (IMGF) with ROI(Region of Interest) extraction
- Detection using Background Segmentation Approaches like PBAS (Pixel-based Adaptive Segmenter), VIBE (Visual Background Extractor), GMM (Gaussian Mixture Models), etc.
- Detection using Template matching algorithms with similarity detections
- Detection using ROI and Holistic-based classification

## Proposed System

Our proposed system addresses the limitations of traditional models and approaches by implementing the latest versions of YOLO and SSD (Single Shot Detector) models, mainly focusing on comparative analysis. As these models operate end-to-end using CNNs (Convolutional Neural Networks) for accurate detection of small and partially occluded weapons, comparing the performance of these models can help determine which model is best suitable in handling complex CCTV environments.

## Datasets Description

There are mainly three datasets, the first two containing different types of guns such as Pistols, Shotguns and Rifles and the third dataset specifically contains the knives. We have annotated all the images in Roboflow website, myself personally annotating dataset 3 for easier exportation of datasets in different formats.

**Dataset-1**

Total Images: 1026
Train Images: 716
Valid Images: 207
Test Images: 103

Additional Info:
- Export Formats: YOLO, COCO, CSV
- Pre-processing: Auto-orientation, Resize to 640x640
- No image augmentation techniques were applied specifically, as we collected the already augmented raw images 
- Split Ratio: 70% Train, 20% Valid, 10% Test

**Dataset-2**

Total Images: 239
Train Images: 168
Valid Images: 47
Test Images: 24

Additional Info:
- Export Formats: YOLO, COCO, CSV
- Pre-processing: Auto-orientation, Resize to 640x640
- No image augmentation techniques were applied specifically, as we collected the already augmented raw images 
- Split Ratio: 70% Train, 20% Valid, 10% Test

**Dataset-3**

Total Images: 566
Train Images: 395
Valid Images: 112
Test Images: 59

No. of Negative Samples out of total images: 118

Additional Info:
- Export Formats: YOLO, COCO, CSV
- Pre-processing: Auto-orientation, Resize to 640x640
- No image augmentation techniques were applied specifically, as we collected the already augmented raw images 
- Split Ratio: 70% Train, 20% Valid, 10% Test

## Models Description

We have trained mainly two base models i.e., YOLO (You Only Look Once) and SSD (Single Shot Detector), with different version and backbones in action. 

**YOLOv11**

YOLOv11 serves as an improved baseline model with enhanced feature extraction and real-time capability. It includes features such as:

- Improved backbone for better feature extraction
- Faster inference suitable for real-time applications
- Balanced performance between accuracy and speed
- Lightweight architecture for deployment on mid-range hardware

We have trained YOLOv11 on all the datasets, myself personally training on Dataset-3 (Knives)

**YOLOv12**

YOLOv12 is used as a high-performance comparison model, particularly strong in accuracy and robustness. It includes features such as:

- Advanced attention mechanisms for cluttered environments
- Improved bounding-box regression for precise localization
- Lightweight architecture for deployment on mid-range hardware
- Robust to occlusion, low resolution, and lighting variations

We have trained YOLOv12 on all the datasets, myself personally training on Dataset-3 (Knives)

**YOLOv26**

Although it was a newly-released version during our YOLO-training phase, YOLOv26 is selected due to its advanced architecture and superior efficiency in detecting small and occluded objects in real-time. It includes features such as:

- End-to-end NMS-free detection, reducing latency and improving inference speed
- Multi-scale feature learning for detecting small and distant objects
- Improved bounding box localization and stability
- Optimized for real-time deployment with better accuracy-speed tradeoff

I myself have personally trained YOLOv26 on all the datasets

**SSD - Single Shot Detector**

SSD is used as a baseline model to compare traditional one-stage detection with advanced YOLO architectures. It is implemented with different backbone networks such as MobileNetV3-Large, EfficientNetB0 and ResNet50. It includes features such as:

- Multi-scale feature maps for detecting objects of different sizes
- Fast inference suitable for real-time applications
- Backbone flexibility for performance vs efficiency trade-offs
- Stable detection for medium-sized object

We have trained all SSD-backboned models on all the datasets in formats like COCO and CSV, myself personally training SSD with EfficientNetB0 and MobileNetV3-Large as backbone on all the datasets.

## Results Summary

For our evaluation, we have used the following metrics:
- Precision
- Recall
- F1-Score
- mAP@50 (Mean Average Precision at 50% IoU (Intersection over Union) threshold)
- mAP@50-95 (Mean Average Precision between 50%-95% IoU (Intersection over Union) thresholds)

After training and evaluating YOLO versions (v11, v12, v26) and SSD variants with EfficientNetB0, MobileNetV3-Large and ResNet50 backbones on multiple annotated datasets, we observed that YOLO-based models consistently achieved strong performance, with YOLOv26 on the Guns Dataset-2 recording the highest F1-score of 0.85 and mAP of 0.56. SSD models showed moderate performance on knife detection but failed critically on gun detection, with mAP@50-95 values as low as 0.0079, making them unsuitable for safety-critical applications. Among YOLO variants, all three performed comparably on knife data, while YOLOv26 demonstrated a clear advantage on gun data — benefiting from better precision-recall balance. Based on overall F1-score, mAP, and consistent performance across both weapon categories, YOLOv26 trained on the Guns Dataset-2 was identified as the best-performing model. The system's use of multi-dataset training, data augmentation, and confidence-based detection ensures reliable performance and makes it suitable for real-time deployment in sensitive environments such as schools, airports, and public spaces.

## Conclusion and Future Work

Our one-year hardwork has led to the conclusion that the latest YOLO models perform better in overall terms when compared to the SSD models. This is to note that the datasets and models have been trained with limited number of images with limited resources available to us. 

Future work can include modifications and advancements like:

- Increasing number of images with adequate resources to train
- Adding video frames to the dataset for more robust detection
- Object detection using advanced models like Transformers and improved YOLO versions. 
- Deployment on edge devices for real-time use with multi-object tracking. 
- Extending detection to advance threats like drones and missiles, military vehicles, suspicious activities, etc and adding automated alert systems for faster response.
