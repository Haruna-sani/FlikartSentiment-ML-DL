# Flikart Bottle Review Analysis

## Overview
This repository contains a comprehensive analysis of Flikart bottle company reviews using Machine Learning (ML) and Deep Learning (DL) models. The study applies various preprocessing techniques, including **TF-IDF** and **Bag of Words**, and implements **SMOTE** to address data imbalance. Multiple classifiers and deep learning architectures are evaluated for review rating predictions.

## Data Preprocessing
- **Text Vectorization**: Applied both **TF-IDF** and **Bag of Words** techniques.
- **Handling Data Imbalance**: Implemented **SMOTE (Synthetic Minority Over-sampling Technique)**.
- **Class Weight Allocation**: Used to mitigate class imbalance issues in classification models.

## Machine Learning Models
Three machine learning classifiers were trained and evaluated:
- **Random Forest (RF)**
- **XGBoost**
- **AdaBoost**

### ML Model Performance Comparison
| Model  | Bag of Words Accuracy | TF-IDF Accuracy |
|--------|----------------------|---------------|
| Random Forest | 0.9233 | 0.9389 |
| XGBoost | 0.9287 | 0.9389 |
| AdaBoost | 0.8223 | 0.9389 |

- **Observation:** Under TF-IDF representation, all models except AdaBoost performed similarly, with RF and XGBoost achieving **0.9389 accuracy**.

## Deep Learning Models
Two deep learning architectures were explored:
- **LSTM (Long Short-Term Memory)**
- **Bi-LSTM (Bidirectional LSTM)**

### DL Model Performance Comparison
| Model  | Accuracy |
|--------|----------|
| LSTM   | 0.1585   |
| Bi-LSTM | 0.9428   |

- **Observation:** The **LSTM model underperformed**, achieving **0.1585 accuracy**, while the **Bi-LSTM model performed exceptionally well with 0.9428 accuracy**.

## Accuracy Plot
![image](https://github.com/user-attachments/assets/541fc8dc-1270-4e01-a7c1-ea961df921a0)

## Conclusion
- **Bi-LSTM emerged as the best-performing model** with an accuracy of **0.9428**, outperforming both ML and LSTM models.
- **Among ML models, TF-IDF representation yielded the highest accuracy across all models.**
- **XGBoost and Random Forest were the best ML classifiers** with **0.9389 accuracy**.
- **Class weight allocation and SMOTE contributed to improved model performance.**


