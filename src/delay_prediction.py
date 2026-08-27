import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("../data/logistics_data.csv")

print("Dataset Shape:", df.shape)

# ==========================================
# 2. Select Features and Target
# ==========================================

features = [
    "Warehouse",
    "Customer_Location",
    "Delivery_Distance_km",
    "Transportation_Mode",
    "Package_Weight_kg",
    "Delivery_Priority",
    "Weather_Condition",
    "Traffic_Level",
    "Warehouse_Processing_Time_min",
    "Planned_Delivery_Time_hr"
]

X = df[features]

y = df["Delivery_Delay"]

# ==========================================
# 3. Categorical Features
# ==========================================

categorical_features = [
    "Warehouse",
    "Customer_Location",
    "Transportation_Mode",
    "Delivery_Priority",
    "Weather_Condition",
    "Traffic_Level"
]

# ==========================================
# 4. Numerical Features
# ==========================================

numerical_features = [
    "Delivery_Distance_km",
    "Package_Weight_kg",
    "Warehouse_Processing_Time_min",
    "Planned_Delivery_Time_hr"
]

# ==========================================
# 5. Data Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# ==========================================
# 6. Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# ==========================================
# 7. Create Pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ==========================================
# 8. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# 9. Train Model
# ==========================================

pipeline.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================================
# 10. Make Predictions
# ==========================================

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]

# ==========================================
# 11. Calculate Performance
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

# ==========================================
# 12. Display Model Performance
# ==========================================

print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# ==========================================
# 13. Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["On Time", "Delayed"]
    )
)

# ==========================================
# 14. Confusion Matrix
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("\n======================================")
print("MODEL TRAINING AND EVALUATION COMPLETE!")
print("======================================")
# ==========================================
# 15. Feature Importance
# ==========================================

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Get feature names after preprocessing
feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get importance values from Random Forest
importances = pipeline.named_steps[
    "model"
].feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Select top 10 features
top_features = feature_importance.head(10)

print("\n======================================")
print("TOP 10 IMPORTANT FEATURES")
print("======================================")

print(top_features.to_string(index=False))


# ==========================================
# 16. Feature Importance Graph
# ==========================================

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"].iloc[::-1],
    top_features["Importance"].iloc[::-1]
)

plt.title("Top 10 Features for Delivery Delay Prediction")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "../visualizations/feature_importance.png"
)

plt.show()


# ==========================================
# 17. Confusion Matrix Graph
# ==========================================

plt.figure(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["On Time", "Delayed"]
)

plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "../visualizations/confusion_matrix.png"
)

plt.show()


print("\n======================================")

print("Feature importance and confusion matrix saved.")
print("======================================")