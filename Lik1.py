import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# Create folders to save outputs
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/confusion_matrices", exist_ok=True)

# -----------------------------------
# Load Dataset (Iris)
# -----------------------------------
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# -----------------------------------
# Train/Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------------
# Initialize Models
# -----------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "LightGBM": lgb.LGBMClassifier()
}

# -----------------------------------
# Evaluation and Result Collection
# -----------------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    # Save confusion matrix as image
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrices/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    # Save model performance
    results.append({
        'Model': name,
        'Accuracy': acc,
        'F1 Score': f1
    })

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------
# Save Results to Excel
# -----------------------------------
results_df = pd.DataFrame(results)
results_df.to_excel("outputs/model_evaluation.xlsx", index=False)

# -----------------------------------
# Plot Accuracy and F1 Score Comparison
# -----------------------------------
plt.figure(figsize=(10, 6))
x = np.arange(len(results_df['Model']))
width = 0.35

plt.bar(x - width/2, results_df['Accuracy'], width, label='Accuracy', color='skyblue')
plt.bar(x + width/2, results_df['F1 Score'], width, label='F1 Score', color='orange')
plt.xticks(x, results_df['Model'], rotation=45)
plt.ylabel('Score')
plt.title('Model Accuracy vs F1 Score')
plt.legend()
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()
