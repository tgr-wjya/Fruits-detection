# YOLOv8s vs YOLO26s Fruit Detection Model Training Report

This report documents the final evaluation metrics, architectural trade-offs, and performance analysis comparing the original YOLOv8s model and the upgraded YOLO26s model on the target fruits detection dataset (evaluated on the validation set containing 914 images with 3227 labeled fruit instances).

## 1. Executive Summary & Overall Metrics

The table below compares the overall performance metrics after completing 50 training epochs:

| Model | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8s** | 59.8% | **40.1%** | **42.8%** | **24.6%** |
| **YOLO26s** | **60.5%** | **40.1%** | 40.6% | 23.2% |

*Key Takeaway:* YOLO26s delivers a slight improvement in Precision (+0.7%) while maintaining the exact same overall Recall (40.1%). However, its mAP50 is slightly lower (-2.2%) and mAP50-95 is slightly lower (-1.4%) compared to YOLOv8s.

---

## 2. Model Complexity & Hardware Footprint

The hardware benchmarks were evaluated on the local host device (NVIDIA GeForce GTX 1650, 4GB VRAM):

| Metric | YOLOv8s | YOLO26s | Change |
| :--- | :---: | :---: | :---: |
| **Parameters** | 11.2M | **9.5M** | -15.2% |
| **GFLOPs** | 28.6 | **20.5** | -28.3% |
| **Inference Latency** | **15.7 ms** | 21.6 ms | +37.6% |
| **Total Frame Processing Time** | **16.9 ms** | 22.3 ms | +32.0% |
| **Actual Frame Rate** | **59.1 FPS** | 44.8 FPS | -24.2% |

*Key Takeaway:* YOLO26s is significantly more compact, requiring 15.2% fewer parameters and 28.3% fewer GFLOPs. However, on the target GPU architecture (GTX 1650), YOLOv8s shows better optimized inference speeds (59.1 FPS vs 44.8 FPS).

---

## 3. Class-by-Class Performance Comparison

Below is the class-by-class comparison of Precision (P), Recall (R), and mAP50 across the six target fruit categories:

| Class | YOLOv8s P | YOLO26s P | YOLOv8s R | YOLO26s R | YOLOv8s mAP50 | YOLO26s mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple** | 56.7% | **60.2%** | 36.2% | **43.6%** | 41.2% | **45.5%** |
| **Banana** | 61.4% | **66.1%** | **37.5%** | 36.4% | **40.8%** | 39.2% |
| **Grape** | 58.4% | **62.5%** | **34.6%** | 31.5% | **36.3%** | 32.5% |
| **Orange** | **64.6%** | 51.8% | **36.9%** | 31.5% | **40.7%** | 30.5% |
| **Pineapple** | 53.6% | **60.4%** | **45.5%** | 41.6% | **43.5%** | 40.4% |
| **Watermelon** | 64.3% | 62.3% | 50.2% | **55.8%** | 54.1% | **55.3%** |

---

## 4. Key Observations & Analysis

*   **Apple Performance Boost:** YOLO26s shows a substantial performance increase for Apples, with Precision rising to 60.2% (+3.5%), Recall rising to 43.6% (+7.4%), and mAP50 rising to 45.5% (+4.3%).
*   **Watermelon Recall Gain:** Watermelon remains highly recognizable, with YOLO26s increasing its Recall to 55.8% (+5.6%) and mAP50 to 55.3% (+1.2%).
*   **Orange Performance Drop:** YOLO26s shows a noticeable decrease in performance on Oranges, with mAP50 dropping to 30.5% (-10.2%) and Precision dropping to 51.8% (-12.8%).
*   **Precision Bias:** YOLO26s shows an improvement in Precision for 4 out of 6 classes (Apple, Banana, Grape, Pineapple), reflecting a general design preference for reducing false positive detections.

---

## 5. Training Convergence Visualization

Below is the training history overlay visualization comparing both models:

![YOLOv8s vs YOLO26s Comparison](/home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/yolov8s_vs_yolo26s_comparison.png)
