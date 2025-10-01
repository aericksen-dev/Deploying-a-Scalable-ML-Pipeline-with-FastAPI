# ml/model.py
from typing import Tuple, Iterable, Optional
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train and return a model.
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def inference(model, X: np.ndarray) -> np.ndarray:
    """
    Run model inferences and return predictions.
    """
    return model.predict(X)

def save_model(model, encoder, label_binarizer, model_path: str, encoder_path: str, lb_path: str) -> None:
    """
    Save model and encoders.
    """
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(label_binarizer, lb_path)

def load_model(model_path: str, encoder_path: str, lb_path: str):
    """
    Load model and encoders.
    """
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
    return model, encoder, lb

def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Precision, recall, f1.
    """
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    return precision, recall, f1

def performance_on_categorical_slice(
    data,
    feature: str,
    values: Iterable,
    process_fn,
    model,
    encoder,
    lb,
    categorical_features: Iterable[str]
) -> str:
    """
    Compute metrics for each value of a categorical feature and return a text block.
    """
    lines = []
    for v in values:
        df_slice = data[data[feature] == v]
        if df_slice.empty:
            continue
        X_slice, y_slice, _, _ = process_fn(
            df_slice,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
        y_pred = inference(model, X_slice)
        p, r, f = compute_model_metrics(y_slice, y_pred)
        lines.append(
            f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}\n{feature}: {v}, Count: {len(df_slice)}"
        )
    return "\n\n".join(lines) + ("\n" if lines else "")
