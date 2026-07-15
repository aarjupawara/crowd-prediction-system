import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ Load Dataset ------------------
df = pd.read_csv(r"C:\Users\AARJU\OneDrive\Desktop\Projects\Crawd Predication\context_crowd_comfort.csv")

# ------------------ Feature Engineering ------------------
df['Footfall_per_hour'] = df['Footfall'] / (df['Hour'] - 7 + 1e-5)  # avoid divide by zero
df['CrowdPressure'] = df['Footfall'] / df['ParkingDistance_m']
df['Hot_Rainy'] = df['Weather'].apply(lambda x: 1 if x in ['Hot', 'Rainy'] else 0)
df['SpecialEvent'] = df[['Event','Holiday','Festival','Offer']].sum(axis=1)

# ------------------ Cyclical Encoding ------------------
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)

day_map = {day:i for i, day in enumerate(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])}
df['Day_num'] = df['Day'].map(day_map)
df['Day_sin'] = np.sin(2 * np.pi * df['Day_num']/7)
df['Day_cos'] = np.cos(2 * np.pi * df['Day_num']/7)

# Drop original columns
df = df.drop(columns=['Day', 'Hour', 'Day_num'])

# ------------------ Features and Target ------------------
X = df.drop(columns=['Comfort'])
y = df['Comfort']

# ------------------ Identify column types ------------------
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ------------------ Preprocessing ------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# ------------------ Train-Test Split ------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------ Pipeline with SMOTE for imbalance ------------------
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# ------------------ Cross-Validation ------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
print(f"5-Fold Cross-Validation F1 Score: {np.mean(cv_scores):.4f}")

# ------------------ Train Model ------------------
pipeline.fit(X_train, y_train)

# ------------------ Evaluate ------------------
y_pred = pipeline.predict(X_val)
print("\nValidation Metrics:\n")
print(classification_report(y_val, y_pred))

# ------------------ Confusion Matrix ------------------
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# ✅ Shows correct vs incorrect predictions for each comfort level.

# ------------------ Prediction Distribution ------------------
plt.figure(figsize=(6,4))
sns.countplot(x=y_pred)
plt.title('Predicted Comfort Level Distribution')
plt.xlabel('Comfort Level')
plt.ylabel('Count')
plt.show()
# ✅ Shows how many times each comfort level was predicted, helps identify bias.

# ------------------ Feature Importance ------------------
feature_names_num = numerical_cols
if categorical_cols:
    cat_ohe_names = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([feature_names_num, cat_ohe_names])
else:
    feature_names = feature_names_num

importances = pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\nTop Features:\n", importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Top 10 Feature Importances')
plt.show()
# ✅ Shows which features most influence crowd comfort predictions.

# ------------------ Save Model ------------------
os.makedirs('saved_models', exist_ok=True)
joblib.dump(pipeline, 'saved_models/comfort_model_improved.pkl')
print("\nModel trained, cross-validated, visualized, and saved successfully!")
