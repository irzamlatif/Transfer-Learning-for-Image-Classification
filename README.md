# Transfer Learning for Image Classification (Reduced CIFAR-10)

This repository contains a high-performance image classification pipeline built using **Transfer Learning** and **Fine-Tuning** on a reduced version of the CIFAR-10 dataset. The project was developed as part of a technical hackathon challenge to demonstrate deep learning architecture implementation, optimization strategies, and thorough error analysis.

## 📌 Project Overview
The objective was to construct an intelligent classifier capable of categorizing low-resolution ($32 \times 32$) RGB images across 5 balanced classes: **airplane, automobile, bird, cat, and dog** (1,000 images per class). 

To maximize accuracy under development time constraints, the project leverages a pre-trained **EfficientNetB0** model, taking advantage of deep visual features learned on the massive ImageNet dataset.

---

## 🛠️ Methodology & Model Architecture

The implementation follows a structured two-stage training workflow to achieve optimal convergence and prevent overfitting on small data sets:

1. **Data Preprocessing & Augmentation:** * Low-resolution images are upscaled to $224 \times 224$ pixels to match the expected input dimensions of EfficientNetB0.
   * On-the-fly data augmentation (random flips, rotations, and zooms) is applied to artificially expand training variety and enhance generalization.

2. **Stage 1: Feature Extraction (Frozen Base)**
   * The core weights of the pre-trained EfficientNetB0 model are frozen.
   * A custom classification head is appended, consisting of a **Global Average Pooling** layer, a **Dense layer** with **Dropout regularization** to counter overfitting, and a final **Softmax output layer** spanning the 5 target classes.
   * **Stage 1 Validation Accuracy:** 93.46%

3. **Stage 2: Fine-Tuning (Unfrozen Base)**
   * Top layers of the EfficientNetB0 backbone are selectively unfrozen.
   * The entire network is trained with a highly reduced learning rate alongside an **Early Stopping** callback. This updates the foundational weights to specialize in subtle class boundaries (such as distinguishing between cats and dogs) without destroying pre-learned shapes.
   * **Stage 2 Validation Accuracy:** 95.62% *(An improvement of +2.16%)*

---

## 📊 Evaluation & Performance Metrics

The finalized network was evaluated against a completely unseen test dataset to ensure real-world robustness.

* **Test Accuracy:** 95.4%
* **Macro F1-Score:** 0.95

### Confusion & Error Analysis
* **Near-Perfect Classes:** Airplanes and automobiles achieved near-flawless classification due to distinct geometric edges and contrasting backgrounds.
* **Primary Mistakes:** The majority of misclassifications occurred exclusively between the **Cat** and **Dog** classes. This error footprint stems from natural visual similarities and pose overlaps rather than model instability, validating the classifier's structural success.

---

## 🤖 Generative AI Tools Comparison
A unique aspect of this project was evaluating the performance and assistance provided by various generative AI tools during the hackathon workspace development lifecycle:
* **ChatGPT:** Utilized for conceptual project structuring, defining reporting layouts, and creating clear textual documentation.
* **GitHub Copilot:** Assisted in writing standard syntax quickly, speeding up data pipeline building.
* **Gemini & DeepSeek:** Leveraged for advanced deep learning conceptual troubleshooting and tuning hyperparameters like learning rates and dropout ratios.

---

## 🚀 How to Run the Project

### Prerequisites
Make sure you have a Python environment setup with a GPU (Google Colab T4/L4 runtime recommended) and the following frameworks installed:
```bash
pip install tensorflow keras matplotlib numpy
