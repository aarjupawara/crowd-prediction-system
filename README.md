


# AI Crowd Comfort Prediction System 

## Overview

AI Crowd Comfort Prediction System is a Machine Learning project that predicts crowd comfort levels by analyzing environmental conditions, crowd behavior, location factors, and user preferences.

The system classifies crowd situations into five categories:
- Calm
- Comfortable
- Moderate
- Noisy
- Overloaded


## Problem Statement

Crowd management is a complex challenge because crowd conditions are influenced by multiple dynamic factors such as footfall, weather, traffic, time, events, and environmental conditions.

Traditional crowd monitoring methods often depend on manual observation, which can be difficult to scale and may not provide predictive insights.

This project explores how Machine Learning can be used to analyze different crowd-related factors and predict the expected comfort level of a crowd environment.

The goal is to build a data-driven system that can help understand crowd behavior and support better planning for events, public spaces, and smart environments.

---



# Why I Built This Project

I created this project as part of my journey to learn and understand Machine Learning by solving a real-world inspired problem.

While exploring different applications of AI, I wanted to understand how machine learning could be used to analyze situations where multiple factors affect human experience and decision-making.

Crowd comfort is influenced by several changing conditions such as:

- Number of people present
- Weather and environmental conditions
- Traffic and accessibility
- Time-based patterns
- Events and special occasions
- Individual user preferences


Instead of building only a prediction model, I wanted to understand the complete machine learning lifecycle:

- Defining a real-world problem
- Creating and designing a custom dataset
- Performing data preprocessing
- Engineering meaningful features
- Training and evaluating machine learning models
- Understanding model decisions through feature importance

This project represents my practical learning experience in Machine Learning and my attempt to build something that can be extended into a real-world intelligent crowd monitoring system.

-----



##  Key Features

| Capability | Description |
|------------|-------------|
| Custom Dataset Creation | Designed and generated a complete crowd comfort dataset from scratch |
| Data Preprocessing | Processed numerical and categorical data for machine learning |
| Feature Engineering | Created meaningful features to improve model understanding |
| Crowd Comfort Classification | Predicts five different crowd comfort categories |
| Machine Learning Pipeline | Built a complete training and evaluation workflow |
| Class Imbalance Handling | Applied SMOTE technique to improve minority class learning |
| Model Training | Trained a Random Forest classification model |
| Model Evaluation | Evaluated using accuracy, F1-score, classification report, and confusion matrix |
| Feature Importance Analysis | Identified the factors that influence predictions the most |
| Model Saving | Saved the trained model for future predictions and deployment |

---


## Project Highlights

- Built an end-to-end Machine Learning pipeline from data generation to model evaluation.
- Created a custom dataset instead of using an existing public dataset.
- Simulated realistic crowd scenarios using environmental, behavioral, and contextual information.
- Combined multiple factors to predict overall crowd comfort conditions.
- Used advanced preprocessing techniques for handling different types of data.
- Applied SMOTE to improve model performance on imbalanced classes.
- Analyzed model decisions using feature importance visualization.
- Developed a foundation that can be extended into a real-time crowd monitoring system.

---


## Dataset Creation

One of the most distinctive aspects of this project is that the dataset was **designed and generated entirely from scratch**.

A major challenge while developing this project was that the concept of **"crowd comfort" is highly subjective**. What feels comfortable to one person may feel overcrowded or noisy to another. Factors such as individual preferences, noise tolerance, weather conditions, accessibility requirements, and personal comfort levels make it difficult to define a single universal measure of crowd comfort.

Since no publicly available dataset captured these combined factors in a way that suited this problem, I designed and generated a custom synthetic dataset specifically for this project. The dataset was created in Python by simulating realistic crowd scenarios while maintaining logical relationships between environmental conditions, crowd characteristics, and user preferences.

The generated dataset contains **1,000 records**, with each record representing a unique crowd environment. Instead of generating random values independently, the features were designed to influence one another to better represent realistic situations.

### Dataset Characteristics

| Category | Features |
|----------|----------|
| Temporal | Day, Hour, Weekend |
| Environmental | Weather, Temperature, Humidity, Cloudiness, Season, Air Quality Index |
| Crowd Metrics | Footfall, Crowd Density |
| Location Context | Traffic, Parking Availability, Parking Distance, Nearby Attractions |
| Event Information | Events, Holidays, Festivals, Promotional Offers |
| User Preferences | Maximum Crowd Preference, Noise Tolerance, Preferred Temperature, Preferred Humidity, Accessibility, Seating Preference |

The target variable, **Comfort**, was generated using a rule-based decision process that considers crowd density, environmental conditions, and user preferences to classify each scenario into one of five comfort levels:

- Calm
- Comfortable
- Moderate
- Noisy
- Overloaded

Designing the dataset from scratch gave me complete control over the problem definition while also helping me understand the importance of feature engineering, data quality, and realistic data generation in machine learning.

---


## Machine Learning Pipeline

The project follows a complete end-to-end machine learning workflow:

```
Dataset Generation
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Encoding & Scaling
      ↓
SMOTE (Class Balancing)
      ↓
Random Forest Training
      ↓
Model Evaluation
      ↓
Prediction
```

### Key Implementation

- Performed feature engineering by creating **Footfall per Hour**, **Crowd Pressure**, **Special Event Score**, and cyclic time features.
- Applied **StandardScaler** for numerical features and **One-Hot Encoding** for categorical features.
- Used **SMOTE** to address class imbalance during training.
- Trained a **Random Forest Classifier** within a Scikit-learn Pipeline for consistent preprocessing and prediction.
- Evaluated the model using cross-validation, classification metrics, confusion matrix, and feature importance analysis.

---

The model was evaluated using cross-validation and an independent test dataset to measure its reliability and generalization performance.

| Metric | Result |
|---------|--------|
| Model | Random Forest Classifier |
| Cross Validation | 5-Fold Stratified Cross Validation |
| Cross Validation F1 Score | **0.8661** |
| Test Accuracy | **98%** |
| Weighted F1 Score | **0.98** |

### Evaluation & Analysis

The trained model includes multiple evaluation techniques to better understand its performance:

- **Classification Report** – Precision, Recall, and F1-Score for each comfort level.
- **Confusion Matrix** – Visualizes correct and incorrect predictions across all classes.
- **Feature Importance Analysis** – Identifies the most influential features affecting crowd comfort prediction.
- **Prediction Distribution** – Shows how predictions are distributed among different comfort categories.

These analyses not only validate the model's predictive performance but also improve its interpretability by highlighting the factors that most influence crowd comfort predictions.

---



## Tech Stack

### Programming & Machine Learning

- Python
- Scikit-learn
- Pandas
- NumPy
- Imbalanced-learn (SMOTE)
- Joblib

### Data Visualization

- Matplotlib
- Seaborn

### Development

- Visual Studio Code
- Git & GitHub

---



## Project Structure

```text
crowd-prediction-system/
│
├── app/
│   └── app.py                    # Application interface
│
├── data/
│   └── dataset.py                # Custom dataset generation
│
├── model/
│   ├── train.py                  # Model training pipeline
│   └── test.py                   # Model evaluation
│
├── saved_models/
│   └── comfort_model_improved.pkl
│
├── context_crowd_comfort.csv     # Custom generated dataset
│
└── README.md
```

---

## Author

**Aarju Mahendra Pawara**

B.Tech - Artificial Intelligence Student

- Machine Learning
- Data Science

GitHub:
https://github.com/aarjupawara
