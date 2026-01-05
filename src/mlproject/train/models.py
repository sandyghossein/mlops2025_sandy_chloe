"""Model training functions."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_models(
    features_train_path: Path,
    clean_train_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    """Train multiple models and select the best one."""
    logger.info("=" * 60)
    logger.info("Starting model training")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features (X)
    logger.info(f"\nLoading features from {features_train_path}")
    X_df = pd.read_parquet(features_train_path)
    logger.info(f"Loaded X: {len(X_df)} rows with {len(X_df.columns)} columns")

    # Load cleaned train to get target (y)
    logger.info(f"Loading cleaned train from {clean_train_path}")
    clean_df = pd.read_csv(clean_train_path)
    if "trip_duration" not in clean_df.columns:
        raise ValueError("trip_duration not found in cleaned training CSV")

    y = clean_df["trip_duration"].to_numpy()

    # Drop non-feature columns if present
    drop_cols = [c for c in ["id", "trip_duration"] if c in X_df.columns]
    feature_cols = [c for c in X_df.columns if c not in drop_cols]

    X = X_df[feature_cols].to_numpy()

    # Safety check: same number of rows
    if len(X) != len(y):
        raise ValueError(f"Row mismatch: features rows={len(X)} vs target rows={len(y)}")

    logger.info(f"Features used: {len(feature_cols)}")
    logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Train/validation split
    logger.info(f"\nSplitting data (test_size={test_size}, random_state={random_state})")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")

    # Define models
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        ),
    }

    # Train and evaluate models
    logger.info("\nTraining and evaluating models...")
    results = {}

    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        results[name] = {
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "train_mae": float(train_mae),
            "val_mae": float(val_mae),
        }

        logger.info(f"{name} - Train RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}")
        logger.info(f"{name} - Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")

    # Select best model by validation RMSE
    best_model_name = min(results.keys(), key=lambda k: results[k]["val_rmse"])
    best_model = models[best_model_name]

    logger.info(
        f"\nBest model: {best_model_name} (Val RMSE: {results[best_model_name]['val_rmse']:.2f})"
    )

    # Retrain best model on full training data
    logger.info(f"\nRetraining {best_model_name} on full training data...")
    best_model.fit(X, y)

    # Save model
    model_path = output_dir / "model.joblib"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved model: {model_path}")

    # Save metrics
    metrics = {"best_model": best_model_name, "models": results}
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    # Save report
    report_path = output_dir / "model_report.txt"
    with open(report_path, "w") as f:
        f.write("Model Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Validation RMSE: {results[best_model_name]['val_rmse']:.2f}\n")
        f.write(f"Validation MAE: {results[best_model_name]['val_mae']:.2f}\n\n")
        f.write("All Models:\n")
        f.write("-" * 60 + "\n")
        for name, m in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Train RMSE: {m['train_rmse']:.2f}\n")
            f.write(f"  Val RMSE: {m['val_rmse']:.2f}\n")
            f.write(f"  Train MAE: {m['train_mae']:.2f}\n")
            f.write(f"  Val MAE: {m['val_mae']:.2f}\n")

    logger.info(f"Saved model report: {report_path}")

    logger.info("=" * 60)
    logger.info("Model training complete!")
    logger.info("=" * 60)

    return model_path
