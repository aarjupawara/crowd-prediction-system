# crowd-prediction-system
A machine learning project for predicting crowd conditions using real-world environmental and crowd behavior features. The system analyzes factors such as footfall, density, humidity, traffic, and time patterns to classify crowd levels and support intelligent crowd monitoring.




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



## System Capabilities

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
