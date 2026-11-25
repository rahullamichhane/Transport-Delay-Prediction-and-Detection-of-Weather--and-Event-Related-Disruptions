"""Entry point: train models on Toronto historical data, run live streaming,
and save all results (tables, images, detections) to the output/ folder."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    roc_curve,
)

from src.data_generator import TransportDataGenerator
from src.models import (
    MODEL_CONFIGS,
    build_model_pipeline,
    evaluate_model,
    save_pipeline,
    train_and_evaluate,
)
from src.preprocessing import build_preprocessor, prepare_features, split_data
from src.streaming import StreamingPredictor, simulate_stream

OUTPUT_DIR = Path("output")
MODELS_DIR = OUTPUT_DIR / "models"
TABLES_DIR = OUTPUT_DIR / "tables"
IMAGES_DIR = OUTPUT_DIR / "images"
DETECTIONS_DIR = OUTPUT_DIR / "detections"


def _ensure_dirs() -> None:
    for d in [OUTPUT_DIR, MODELS_DIR, TABLES_DIR, IMAGES_DIR, DETECTIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _save_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4),
                "ROC-AUC": round(m["roc_auc"], 4) if pd.notna(m["roc_auc"]) else None,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    # Image: bar chart of key metrics
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    for ax, metric in zip(axes.flat, metrics):
        df_sorted = df.sort_values(metric, ascending=True)
        sns.barplot(
            x=metric,
            y="Model",
            data=df_sorted,
            ax=ax,
            palette="viridis",
            hue="Model",
            legend=False,
        )
        ax.set_title(f"Model Comparison - {metric}")
        ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df


def _save_model_details(
    name: str,
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred,
    y_prob,
) -> None:
    safe = name.lower().replace(" ", "_")

    # Confusion matrix image
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title(f"Confusion Matrix - {name}")
    fig.savefig(IMAGES_DIR / f"confusion_matrix_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ROC curve image
    if y_prob is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
        ax.set_title(f"ROC Curve - {name}")
        fig.savefig(IMAGES_DIR / f"roc_curve_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _save_delay_detections(history: list[dict]) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    # Keep relevant columns
    keep = [
        c
        for c in [
            "record_id",
            "timestamp",
            "vehicle_id",
            "vehicle_class",
            "route_id",
            "stop_id",
            "route_type",
            "weather_condition",
            "event_type",
            "is_peak_hour",
            "is_weekend",
            "delay_minutes",
            "delay_label",
        ]
        if c in df.columns
    ]
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    df[keep + pred_cols].to_csv(DETECTIONS_DIR / "delay_detections.csv", index=False)

    # Delay distribution by vehicle class image
    if "vehicle_class" in df.columns and "delay_minutes" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="vehicle_class", y="delay_minutes", data=df, ax=ax, palette="Set2")
        ax.set_title("Delay Minutes Distribution by Vehicle Class")
        fig.savefig(IMAGES_DIR / "delay_by_vehicle_class.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Delay label distribution image
    if "delay_label" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        df["delay_label"].value_counts().sort_index().plot(
            kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"]
        )
        ax.set_title("Delay Label Distribution (0=On Time, 1=Delayed)")
        ax.set_xlabel("Delay Label")
        ax.set_ylabel("Count")
        ax.set_xticklabels(["On Time", "Delayed"], rotation=0)
        fig.savefig(IMAGES_DIR / "delay_label_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("  Public Transport Delays Detection - Toronto (TTC)")
    print("  Essential for Data Science - ML Project")
    print("=" * 70)

    _ensure_dirs()

    # 1. Generate historical training data
    gen = TransportDataGenerator(seed=42)
    df = gen.generate_historical(n_records=6000)
    print(f"[1/5] Generated {len(df)} historical Toronto records.")

    # 2. Prepare features / split
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    print("[2/5] Features prepared and train/test split complete.")

    # 3. Train all models
    print("[3/5] Training models...")
    pipelines: dict = {}
    results: dict = {}
    all_y_pred: dict = {}
    all_y_prob: dict = {}

    for name in MODEL_CONFIGS:
        pipeline, metrics = train_and_evaluate(
            name, preprocessor, X_train, y_train, X_test, y_test
        )
        pipelines[name] = pipeline
        results[name] = metrics

        # Re-run predictions for detail images
        y_pred = pipeline.predict(X_test)
        y_prob = None
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        all_y_pred[name] = y_pred
        all_y_prob[name] = y_prob

        print(f"      {name:28s} Accuracy={metrics['accuracy']:.4f} F1={metrics['f1']:.4f}")

    # 4. Save comparison table + images
    comp_df = _save_comparison_table(results)
    for name, pipeline in pipelines.items():
        save_pipeline(pipeline, MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl")
    save_pipeline(preprocessor, MODELS_DIR / "preprocessor.pkl")
    print("[4/5] Saved model pipelines to output/models/")

    # Model detail images (confusion matrix, ROC)
    for name, pipeline in pipelines.items():
        _save_model_details(
            name,
            pipeline,
            X_test,
            y_test,
            all_y_pred[name],
            all_y_prob[name],
        )
    print("      Saved model detail images to output/images/")

    # 5. Live streaming demo with vehicle classification
    print("[5/5] Running live streaming demo (500 records)...")
    stream_gen = TransportDataGenerator(seed=7)
    stream_records = stream_gen.stream_records(n_records=500)

    predictor = StreamingPredictor(window_size=200)
    for name in pipelines:
        predictor.register_model(name)

    simulate_stream(stream_records, pipelines, preprocessor, predictor, delay_seconds=0.0)

    history = predictor.to_dataframe()
    _save_delay_detections(history.to_dict("records"))

    print(f"      Processed {len(history)} streaming records.")
    print(f"      True delay rate: {predictor.recent_delay_rate():.3f}")
    for name in pipelines:
        print(f"      {name:28s} predicted delay rate: {predictor.recent_delay_rate(name):.3f}")

    # Vehicle class summary
    if "vehicle_class" in history.columns:
        vc = history["vehicle_class"].value_counts()
        print("\nVehicle class distribution in stream:")
        for cls, cnt in vc.items():
            print(f"      {cls:12s}: {cnt} records")

    print("\n" + "=" * 70)
    print("All outputs saved to the output/ folder:")
    print(f"  {TABLES_DIR}/          - model_comparison.csv")
    print(f"  {IMAGES_DIR}/          - comparison, confusion matrices, ROC curves")
    print(f"  {MODELS_DIR}/          - trained pipeline .pkl files")
    print(f"  {DETECTIONS_DIR}/      - delay_detections.csv")
    print("=" * 70)
    print("Launch dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
