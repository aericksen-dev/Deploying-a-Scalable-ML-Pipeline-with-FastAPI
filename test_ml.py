# test_ml.py
from pathlib import Path

import pandas as pd

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

# same list used in training (hyphenated names)
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def _load_sample(n=400, seed=0):
    data_path = Path(__file__).resolve().parent / "data" / "census.csv"
    df = pd.read_csv(data_path)
    if len(df) > n:
        df = df.sample(n, random_state=seed)
    return df


def test_process_data_shapes():
    df = _load_sample(200, seed=1)
    X, y, enc, lb = process_data(
        df, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    assert X.shape[0] == y.shape[0] > 0
    assert enc is not None and lb is not None


def test_train_and_infer_binary():
    df = _load_sample(500, seed=2)
    train = df.iloc[:350]
    test = df.iloc[350:]
    Xtr, ytr, enc, lb = process_data(train, CAT_FEATURES, label="salary", training=True)
    Xte, yte, *_ = process_data(
        test, CAT_FEATURES, label="salary", training=False, encoder=enc, lb=lb
    )
    model = train_model(Xtr, ytr)
    preds = inference(model, Xte)
    assert set(preds.tolist()) <= {0, 1}
    assert len(preds) == len(yte)


def test_metrics_range():
    p, r, f = compute_model_metrics([0, 1, 1, 0, 1], [0, 1, 0, 0, 1])
    for v in (p, r, f):
        assert 0.0 <= v <= 1.0
