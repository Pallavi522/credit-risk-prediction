import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:/Users/Dell/New folder/credit/train.csv")
print(df.head())
print(df.info())
print(df.describe())
df = df.dropna()
# Assuming 'Credit_Score' is the target variable
X = df.drop(columns=['Credit_Score'],axis=1)  # Features
y = df['Credit_Score']  # Target variable
from sklearn.preprocessing import LabelEncoder

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Encoded classes:", label_encoder.classes_)
print("Encoded labels:", y_encoded[:10])
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Apply the transformations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Check split data
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print("Transformed training set shape:", X_train_transformed.shape)
print("Transformed test set shape:", X_test_transformed.shape)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)  # Increase the max_iter parameter
logreg.fit(X_train_transformed, y_train)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_transformed, y_train)

from sklearn.preprocessing import LabelEncoder

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Assuming 'y' contains your original target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)  # Encode training labels
y_test_encoded = label_encoder.transform(y_test)     # Encode test labels using the same encoder

xgb = XGBClassifier()
xgb.fit(X_train_transformed, y_train_encoded)  # Use encoded labels for training

# Model Evaluation
# Decode target labels for evaluation if needed
y_test_decoded = label_encoder.inverse_transform(y_test_encoded)

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

y_pred_logreg_encoded = label_encoder.transform(logreg.predict(X_test_transformed))
# Predict probabilities instead of class labels
y_pred_proba = logreg.predict_proba(X_test_transformed) 

print("Logistic Regression Accuracy:", accuracy_score(y_test_encoded, y_pred_logreg_encoded))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred_logreg_encoded))

# Use predicted probabilities for ROC AUC calculation
print("ROC AUC Score:", roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')) # Using 'ovr' strategy


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Predict class labels using the trained Random Forest model
y_pred_rf_encoded = rf.predict(X_test_transformed) 

# Calculate probabilities for ROC AUC
y_pred_proba_rf = rf.predict_proba(X_test_transformed) 

# Ensure both arrays have the same data type (numerical labels)
y_pred_rf_numeric = label_encoder.transform(y_pred_rf_encoded)  # Transform predicted labels to numerical

print("Random Forest Accuracy:", accuracy_score(y_test_encoded, y_pred_rf_numeric))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred_rf_numeric)) # Use encoded predictions for confusion matrix
print("ROC AUC Score:", roc_auc_score(y_test_encoded, y_pred_proba_rf, multi_class='ovr'))


y_pred_proba_xgb = xgb.predict_proba(X_test_transformed) 
# Predict class labels for XGBoost
y_pred_xgb = xgb.predict(X_test_transformed)  

print("XGBoost Accuracy:", accuracy_score(y_test_encoded, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred_xgb))
print("ROC AUC Score:", roc_auc_score(y_test_encoded, y_pred_proba_xgb, multi_class='ovr'))


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Predict probabilities for each model
y_pred_proba_logreg = logreg.predict_proba(X_test_transformed)
y_pred_proba_rf = rf.predict_proba(X_test_transformed)
y_pred_proba_xgb = xgb.predict_proba(X_test_transformed)

# Define the plot_roc_curve function
def plot_roc_curve(y_true, y_pred_proba, model_name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    for i in range(3):
        plt.plot(fpr[i], tpr[i], label=f'{model_name} Class {i} (area = {roc_auc[i]:.2f})')

# Binarize the y_test_encoded variable (assuming it's available)
y_test_binarized = label_binarize(y_test_encoded, classes=np.unique(y_test_encoded)) # Assuming you want to binarize y_test_encoded

plt.figure()
# Plot ROC curves
plot_roc_curve(y_test_binarized, y_pred_proba_logreg, 'Logistic Regression')
plot_roc_curve(y_test_binarized, y_pred_proba_rf, 'Random Forest')
plot_roc_curve(y_test_binarized, y_pred_proba_xgb, 'XGBoost')

# Add labels and legend
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

import joblib

# Save models
joblib.dump(logreg, 'logreg_model.pkl')
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(xgb, 'xgb_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Load models
loaded_logreg = joblib.load('logreg_model.pkl')
loaded_rf = joblib.load('rf_model.pkl')
loaded_xgb = joblib.load('xgb_model.pkl')
loaded_label_encoder = joblib.load('label_encoder.pkl')

# Predictions with loaded models
y_pred_logreg_loaded = loaded_logreg.predict(X_test_transformed)
y_pred_rf_loaded = loaded_rf.predict(X_test_transformed)
y_pred_xgb_loaded = loaded_xgb.predict(X_test_transformed)

# Evaluate loaded models
print("Loaded Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg_loaded))
print("Loaded Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf_loaded))
print("Loaded XGBoost Accuracy:", accuracy_score(y_test_encoded, y_pred_xgb_loaded))



