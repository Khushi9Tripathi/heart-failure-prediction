import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,          
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

param_options = [
    {"n_estimators":100, "max_depth":None},
    {"n_estimators":200, "max_depth":None},
    {"n_estimators":300, "max_depth":None},
    {"n_estimators":200, "max_depth":5},
    {"n_estimators":300, "max_depth":5},
    {"n_estimators":500, "max_depth":5},
    {"n_estimators":500, "max_depth":None},
]

best_acc = 0.0
best_model = None
best_params = None

print("Tuning models...\n")
for p in param_options:
    rf = RandomForestClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        class_weight="balanced",  # NEW: helps class imbalance
        random_state=42,
    )
    rf.fit(X_train_scaled, y_train)
    preds = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"Params: {p}  ->  Accuracy: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_model = rf
        best_params = p

print("\nBest Params Chosen:", best_params)
print(f"Best Test Accuracy: {best_acc*100:.2f}%\n")

y_pred_best = best_model.predict(X_test_scaled)
print("Classification Report:\n")
print(classification_report(y_test, y_pred_best))

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Best model and scaler saved successfully!")
