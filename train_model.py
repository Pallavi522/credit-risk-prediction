import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv(r"C:/Users/Dell/New folder/credit/train.csv")  # Replace with your actual file path

# Handle missing values
df = df.dropna()

# Encode target variable
le = LabelEncoder()
df['Credit_Score'] = le.fit_transform(df['Credit_Score'])

# Features and target variable
X = df.drop(columns=['Credit_Score'])
y = df['Credit_Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert categorical features to dummy/indicator variables
X_train_dummies = pd.get_dummies(X_train)
X_test_dummies = pd.get_dummies(X_test)

# Ensure consistent columns between training and test sets
X_train_dummies, X_test_dummies = X_train_dummies.align(X_test_dummies, join='left', axis=1, fill_value=0)

# Save the transformer (if needed)
# joblib.dump(dummy_transformer, 'dummy_transformer.pkl')

# Check for non-numeric data
if pd.DataFrame(X_train_dummies).select_dtypes(include=['object']).shape[1] > 0:
    print("Non-numeric data found in X_train_dummies:")
    print(pd.DataFrame(X_train_dummies).select_dtypes(include=['object']).columns)
else:
    print("All data in X_train_dummies is numeric.")

# Assert no non-numeric data
assert pd.DataFrame(X_train_dummies).select_dtypes(include=['object']).empty, "Non-numeric data found in X_train_dummies"

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_dummies)
X_test_scaled = scaler.transform(X_test_dummies)

# Train the XGBoost model
xgb = XGBClassifier()
xgb.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred_xgb = xgb.predict(X_test_scaled)
y_pred_proba_xgb = xgb.predict_proba(X_test_scaled)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Calculate ROC AUC Score for multi-class
roc_auc = roc_auc_score(y_test, y_pred_proba_xgb, multi_class='ovr')
print("ROC AUC Score:", roc_auc)

# Save the model and scaler
joblib.dump(xgb, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')





