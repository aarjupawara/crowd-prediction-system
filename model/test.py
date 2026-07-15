# project/model/test_model.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Load Dataset ------------------
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'context_crowd_comfort.csv')
df = pd.read_csv(dataset_path)

# ------------------ Feature Engineering (same as training) ------------------
df['Footfall_per_hour'] = df['Footfall'] / (df['Hour'] - 7 + 1e-5)
df['CrowdPressure'] = df['Footfall'] / df['ParkingDistance_m']
df['Hot_Rainy'] = df['Weather'].apply(lambda x: 1 if x in ['Hot', 'Rainy'] else 0)
df['SpecialEvent'] = df[['Event','Holiday','Festival','Offer']].sum(axis=1)

# Cyclical encoding
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
day_map = {day:i for i, day in enumerate(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])}
df['Day_num'] = df['Day'].map(day_map)
df['Day_sin'] = np.sin(2 * np.pi * df['Day_num']/7)
df['Day_cos'] = np.cos(2 * np.pi * df['Day_num']/7)
df = df.drop(columns=['Day','Hour','Day_num'])

# ------------------ Features and Target ------------------
X_test = df.drop(columns=['Comfort'])
y_test = df['Comfort']

# ------------------ Load Trained Model ------------------
model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'comfort_model_improved.pkl')
model = joblib.load(model_path)

# ------------------ Predict ------------------
y_pred = model.predict(X_test)

# ------------------ Evaluation ------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Test Data')
plt.show()

# ------------------ Prediction Distribution ------------------
plt.figure(figsize=(6,4))
sns.countplot(x=y_pred)
plt.title('Predicted Comfort Level Distribution')
plt.xlabel('Comfort Level')
plt.ylabel('Count')
plt.show()

# ------------------ Feature Importance ------------------
feature_names_num = X_test.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X_test.select_dtypes(include='object').columns.tolist()

if categorical_cols:
    cat_ohe_names = model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([feature_names_num, cat_ohe_names])
else:
    feature_names = feature_names_num

importances = model.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Feature Importances:\n", importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Top 10 Feature Importances - Test Data')
plt.show()
