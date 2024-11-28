
# Land Cover Classification Model 

This repository contains a deep learning model for land cover classification using **DeepLabV3** with a **ResNet-50** backbone. The model predicts land cover types (e.g., urban, agriculture, water) from satellite imagery, leveraging **semi-supervised learning** to reduce the time and cost of labeling large datasets.

## Table of Contents

- [Overview](#overview)
- [Semi-Supervised Learning](#semi-supervised-learning)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training Process](#training-process)
- [Model Testing](#model-testing)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)

---

## Overview

The model is trained using satellite images and their corresponding masks (segmentation labels). **Semi-supervised learning** is applied by labeling only a subset of pixels in each image, which reduces annotation effort while maintaining high model performance.

---

## Semi-Supervised Learning

**Semi-supervised learning** allows training the model with fewer labeled pixels, improving the efficiency of training. This is achieved by randomly selecting a small number of labeled pixels for each image and calculating the loss for only those labeled points.

---

## Model Architecture

The model is based on **DeepLabV3** with a **ResNet-50** backbone. The model has been adapted to classify 7 land cover types using pixel-wise classification, with the output being a segmentation mask.

---

## Dataset

The dataset consists of satellite images and corresponding segmentation masks:
- **Images**: Satellite imagery (e.g., `.jpg` files).
- **Masks**: Ground truth segmentation masks (e.g., `.png` files), with classes defined in RGB format.

---

## Training Process

The model is trained using a subset of labeled pixels from each image. The loss is computed based on these labeled pixels, and the model is updated accordingly. The custom loss function, `PartialCrossEntropyLoss`, ensures that only the labeled pixels contribute to the loss.

---

## Model Testing

After training, the model is evaluated on unseen test data. Performance metrics such as **IoU**, **Precision**, **Recall**, and **F1-Score** are computed for each land cover class. Additionally, segmentation results can be visualized alongside original satellite images.

---

## Evaluation Metrics

The model's performance is evaluated using the following metrics:
- **Jaccard Index (IoU)**: Measures the intersection over union between predicted and true segments.
- **Precision**: The proportion of true positive predictions out of all predicted positives.
- **Recall**: The proportion of true positive predictions out of all actual positives.
- **F1-Score**: The harmonic mean of precision and recall.

---

## Installation

To run the training and testing scripts, you need the following Python libraries:
```bash
pip install torch torchvision scikit-learn scikit-image Pillow matplotlib
```
## Disclaimer

This code is provided **as-is** for educational and research purposes. The author makes no warranty regarding the model’s accuracy, reliability, or suitability for specific applications. By using this code, you agree that the author is not responsible for any consequences, including errors, misuse, or adverse effects in real-world applications. Users are encouraged to evaluate the model's performance and adapt it to their specific needs.

