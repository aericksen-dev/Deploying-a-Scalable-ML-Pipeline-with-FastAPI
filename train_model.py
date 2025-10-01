from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

if __name__ == "__main__":
    # ---- paths
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "census.csv"
    model_dir = project_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    encoder_path = model_dir / "encoder.pkl"
    lb_path = model_dir / "lb.pkl"
    slice_out = project_dir / "slice_output.txt"

    # ---- load data
    df = pd.read_csv(data_path)

    # ---- split
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["salary"]
    )

    # ---- process
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # ---- train
    model = train_model(X_train, y_train)

    # ---- evaluate
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fbeta:.4f}")

    # ---- save
    save_model(model, encoder, lb, str(model_path), str(encoder_path), str(lb_path))
    print(f"Saved: {model_path}, {encoder_path}, {lb_path}")

    # ---- (optional) reload to verify artifacts (do NOT do this before saving)
    if model_path.exists() and encoder_path.exists() and lb_path.exists():
        model, encoder, lb = load_model(
            str(model_path), str(encoder_path), str(lb_path)
        )
    else:
        print("Artifacts missing after save; skipping reload.")

    # ---- slice metrics
    blocks = []
    for col in cat_features:
        vals = sorted(test[col].dropna().unique().tolist())
        if not vals:
            continue
        block = performance_on_categorical_slice(
            data=test,
            feature=col,
            values=vals,  # <-- key fix: pass all values as an iterable
            process_fn=process_data,
            model=model,
            encoder=encoder,
            lb=lb,
            categorical_features=cat_features,
        )
        if block:
            blocks.append(block)

    slice_out.write_text("".join(blocks))
    print(f"Slice metrics written to {slice_out}")
