# Brain-Tumor-MRI-Segmentation-Using-U-Net-with-Hausdorff-Distance-Optimization

This project focuses on enhancing brain tumor segmentation in MRI scans using a U-Net-based deep learning model, optimized with a Hausdorff Distance-based loss function. Accurate segmentation is critical in medical imaging for effective diagnosis, treatment planning, and monitoring. The proposed approach aims to minimize boundary errors and improve clinical reliability.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Execution Environment](#execution-environment)
- [Results](#results)
- [Future Scope](#future-scope)

---

## Overview

Brain tumors are life-threatening conditions that require accurate detection and segmentation from MRI scans. Manual segmentation is time-consuming and prone to variability. This project presents an automated solution using a U-Net architecture, enhanced by a Hausdorff Distance-based loss function to deliver high-precision segmentation.

---

## Problem Statement

Develop a deep learning-based method to improve the accuracy and boundary precision of brain tumor segmentation in MRI scans, reducing reliance on manual interpretation and enhancing clinical reliability.

---

## Objectives

- Develop a U-Net-based brain tumor segmentation model with optimized boundary delineation.
- Minimize the **Hausdorff Distance** to enhance the alignment between predicted and actual tumor boundaries.
- Validate improvements over traditional loss functions (e.g., Dice, IoU).
- Support application across multiple imaging modalities for broader clinical use.

---

## Dataset

Two public MRI datasets were used for training and validation:

- [Brain Tumor Segmentation Dataset](https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset): Contains labeled MRI scans for pixel-wise tumor segmentation.
- [Brain Tumor Classification MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri): Includes images of glioma, meningioma, pituitary tumors, and non-tumor cases.

---

## Methodology

The model architecture and training pipeline include the following:

- **U-Net Model**: An encoder-decoder CNN designed for biomedical image segmentation.
- **Custom Loss Function**: A Hausdorff Distance-based loss to optimize boundary accuracy beyond standard metrics.
- **Preprocessing**: Normalization and resizing of MRI images to standard formats.
- **Data Augmentation**: Rotation, flipping, zoom, and contrast adjustments to enhance generalization.
- **Validation**: Comparative experiments with Dice and IoU loss functions.

---

## Evaluation Metrics

The model's performance is assessed using both region-based and boundary-based metrics:

- **Dice Coefficient**: Overlap between predicted and true masks.
- **Intersection over Union (IoU)**: Ratio of intersection to union of prediction and ground truth.
- **Hausdorff Distance (HD)**: Measures maximum distance between the boundaries of prediction and ground truth, emphasizing precision.

---

## Execution Environment

- **Platform**: Kaggle Notebooks
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - PyTorch (model implementation)
  - OpenCV, NumPy (image processing)
  - Matplotlib, Seaborn (visualization)
  - SciPy (scientific computing)

---

## Results

### Quantitative Performance

| Metric             | Training | Validation |
|--------------------|----------|------------|
| Dice Coefficient   | 0.9054   | 0.8807     |
| Intersection over Union (IoU) | 0.8275 | 0.7877 |
| Accuracy           | 99.51%   | 99.47%     |
| Hausdorff Distance | Improved compared to baseline models |

### Qualitative Performance

- The segmentation masks show high alignment with ground-truth labels.
- The model performed well across various tumor types.
- Boundary precision was significantly improved by using the HD-based loss.

---

## Future Scope

- **Expand Dataset**: Incorporate larger and multi-institutional datasets for broader generalization.
- **Multi-Class Segmentation**: Enable classification and segmentation of tumor subtypes.
- **Post-Processing**: Apply morphological techniques to further refine predicted boundaries.
- **Deployment**: Develop a web-based interface for real-time use in clinical environments.
- **Cross-Modality Adaptation**: Extend the model to CT, ultrasound, and X-ray modalities.

---

## Summary

This project introduces a Hausdorff Distance-optimized U-Net model that improves tumor boundary segmentation in brain MRI scans. By addressing the critical aspect of boundary accuracy, it enhances the clinical relevance of AI-based tools in medical diagnostics.
