"""Batch inference functions."""

import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def batch_inference(
    features_test_path: Path,
    output_dir: Path,
    model_path: Path,
) -> Path:
    """Run batch inference on test FEATURES and save predictions."""
    logger.info("=" * 60)
    logger.info("Starting batch inference")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Model loaded: {type(model).__name__}")

    # Load test features
    logger.info(f"Loading test features from {features_test_path}")
    test_df = pd.read_parquet(features_test_path)
    logger.info(f"Loaded {len(test_df)} rows with {len(test_df.columns)} columns")

    # Drop non-feature columns if present
    drop_cols = [c for c in ["trip_duration"] if c in test_df.columns]
    if drop_cols:
        logger.warning(f"Dropping columns in test features: {drop_cols}")
        test_df = test_df.drop(columns=drop_cols)

    # Predict (keep DataFrame to preserve feature names)
    logger.info("Making predictions...")
    preds = model.predict(test_df)
    logger.info(f"Generated {len(preds)} predictions")

    # Output format (required file name format)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = output_dir / f"{date_str}_predictions.csv"

    # If test has id, include it, otherwise just output predictions
    if "id" in test_df.columns:
        out_df = pd.DataFrame({"id": test_df["id"].values, "trip_duration": preds})
    else:
        out_df = pd.DataFrame({"trip_duration": preds})

    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to: {out_path}")

    logger.info("=" * 60)
    logger.info("Batch inference complete!")
    logger.info("=" * 60)

    return out_path