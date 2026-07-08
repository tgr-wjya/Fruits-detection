# YOLOv8s Fruit Detection Model Training Report

This report documents the final evaluation metrics and performance analysis of the fine-tuned YOLOv8s fruit detection model after completing all 50 training epochs.

## 1. Evaluation Summary
The model was evaluated on the validation split containing 914 images with 3227 labeled fruit instances.

- **Precision (P):** 59.8% (out of all fruit predictions made, 59.8% were correct)
- **Recall (R):** 40.1% (the model successfully detected 40.1% of the actual fruits present)
- **mAP50:** 42.8% (mean Average Precision at Intersection over Union threshold of 0.5)
- **mAP50-95:** 24.6% (mean Average Precision across IoU thresholds from 0.5 to 0.95)

---

## 2. Class-by-Class Performance Metrics

Below is the detailed performance breakdown across all six fruit categories:

| Class | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **Watermelon** | 64.3% | 50.2% | 54.1% | 33.8% |
| **Orange** | 64.6% | 36.9% | 40.7% | 23.3% |
| **Banana** | 61.4% | 37.5% | 40.8% | 21.0% |
| **Apple** | 56.7% | 36.2% | 41.2% | 26.1% |
| **Grape** | 58.4% | 34.6% | 36.3% | 19.9% |
| **Pineapple** | 53.6% | 45.5% | 43.5% | 23.7% |
| **Overall Average** | **59.8%** | **40.1%** | **42.8%** | **24.6%** |

---

## 3. Key Observations & Performance Analysis

### Strengths
- **Watermelon** achieves the highest overall accuracy (mAP50 of 54.1% and recall of 50.2%). This is likely due to watermelons having distinct, large shapes and highly recognizable patterns (stripes) that make them easy for the feature extractor to capture.
- **Orange** has the highest precision at 64.6%, meaning false positive detections for oranges are very low.

### Areas for Improvement
- **Grape** has the lowest overall accuracy (mAP50 of 36.3%). Grapes are small, often clustered together, and can easily blend into the background or overlap with other fruits, leading to lower recall (34.6%).
- **Pineapple** has a high recall of 45.5% but a relatively low precision of 53.6%, indicating that the model occasionally mistakes background noise or other texture patterns for pineapples.

---

## 4. Hardware Speed & Latency (NVIDIA GeForce GTX 1650)
- **Pre-processing:** 0.5 ms per image
- **Inference:** 15.7 ms per image
- **Post-processing:** 0.7 ms per image
- **Total Frame Processing Time:** 16.9 ms
- **Theoretical Frame Rate:** ~59.1 Frames Per Second (FPS)

This makes the model highly suitable for real-time video stream processing on local edge devices.
