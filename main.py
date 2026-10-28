from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Folders for experiment outputs
BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"
MLRUNS_DIR = BASE_DIR / "mlruns"
REPORTS_DIR.mkdir(exist_ok=True)

# Load UCI Wine dataset through scikit-learn
wine = load_wine(as_frame=True)
data = wine.frame.copy()

# Use only input features for data-drift analysis
features = wine.feature_names
reference_data = data.sample(frac=0.70, random_state=42).reset_index(drop=True)
current_data = data.drop(reference_data.index).reset_index(drop=True)

# Simulate data drift in current data
current_data["alcohol"] = current_data["alcohol"] + 2.0
current_data["color_intensity"] = current_data["color_intensity"] * 1.45
current_data["proline"] = current_data["proline"] + 180

# Train model using reference data
X = reference_data[features]
y = reference_data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]
)

model.fit(X_train, y_train)

reference_accuracy = accuracy_score(y_test, model.predict(X_test))
current_accuracy = accuracy_score(
    current_data["target"], model.predict(current_data[features])
)

# Configure local MLflow tracking
mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
mlflow.set_experiment("Wine Data Drift Analysis")

with mlflow.start_run(run_name="logistic_regression_with_simulated_drift"):
    mlflow.log_params(
        {
            "dataset": "UCI Wine dataset via scikit-learn",
            "model": "Logistic Regression",
            "drift_type": "Simulated data drift",
            "drifted_features": "alcohol, color_intensity, proline",
            "reference_rows": len(reference_data),
            "current_rows": len(current_data),
        }
    )

    mlflow.log_metrics(
        {
            "reference_accuracy": reference_accuracy,
            "current_accuracy": current_accuracy,
        }
    )

    mlflow.sklearn.log_model(model, "model")

    # Evidently data-drift report
    report = Report([DataDriftPreset()])
    snapshot = report.run(
        current_data=current_data[features],
        reference_data=reference_data[features],
    )

    html_path = REPORTS_DIR / "data_drift_report.html"
    json_path = REPORTS_DIR / "data_drift_report.json"

    snapshot.save_html(str(html_path))
    json_path.write_text(snapshot.json(), encoding="utf-8")

    mlflow.log_artifact(str(html_path))
    mlflow.log_artifact(str(json_path))

print("Experiment completed")
print(f"Reference accuracy: {reference_accuracy:.4f}")
print(f"Current accuracy after simulated drift: {current_accuracy:.4f}")
print(f"Evidently report: {REPORTS_DIR / 'data_drift_report.html'}")
print("Open MLflow UI with: mlflow ui --backend-store-uri ./mlruns --port 5001")