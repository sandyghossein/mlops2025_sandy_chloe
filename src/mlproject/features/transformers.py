"""Feature engineering functions."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def haversine_distance(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """
    Calculate haversine distance between two points in kilometers.
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    return distance


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from pickup_datetime only (no leakage)."""
    logger.info("Creating time-based features...")
    
    if "pickup_datetime" not in df.columns:
        logger.warning("pickup_datetime not found, skipping time features")
        return df
    
    df = df.copy()
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)
    
    logger.info("Created time features: pickup_hour, pickup_dayofweek, pickup_month, pickup_is_weekend")
    return df


def create_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create distance feature using haversine distance."""
    logger.info("Creating distance feature...")
    
    required_cols = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    if not all(col in df.columns for col in required_cols):
        logger.warning("Missing required columns for distance calculation")
        return df
    
    df = df.copy()
    df["haversine_distance_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    
    logger.info("Created haversine_distance_km feature")
    return df


def build_feature_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline for features."""
    logger.info("Building feature preprocessing pipeline...")
    
    # Identify columns
    numeric_cols = []
    categorical_cols = []
    
    feature_cols = [
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "pickup_is_weekend",
        "haversine_distance_km",
        "passenger_count",
    ]
    
    for col in feature_cols:
        if col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                numeric_cols.append(col)
    
    categorical_feature_cols = ["store_and_fwd_flag", "vendor_id"]
    for col in categorical_feature_cols:
        if col in df.columns:
            categorical_cols.append(col)
    
    logger.info(f"Numeric features: {numeric_cols}")
    logger.info(f"Categorical features: {categorical_cols}")
    
    transformers = []
    
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_cols)
        )
    
    if not transformers:
        raise ValueError("No features found to transform")
    
    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    
    return preprocessor


def engineer_features(
    clean_train_path: Path,
    clean_test_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Engineer features from cleaned data."""
    logger.info("=" * 60)
    logger.info("Starting feature engineering")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cleaned data
    logger.info("\nLoading cleaned data...")
    train_df = pd.read_csv(clean_train_path)
    test_df = pd.read_csv(clean_test_path)
    logger.info(f"Training data: {len(train_df)} rows")
    logger.info(f"Test data: {len(test_df)} rows")
    
    # Parse datetime columns (they may be strings when read from CSV)
    if "pickup_datetime" in train_df.columns:
        train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"], errors="coerce")
    if "pickup_datetime" in test_df.columns:
        test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")
    
    # Create features
    logger.info("\nCreating features...")
    train_df = create_time_features(train_df)
    train_df = create_distance_feature(train_df)
    
    test_df = create_time_features(test_df)
    test_df = create_distance_feature(test_df)
    
    # Build and fit preprocessing pipeline on training data
    logger.info("\nBuilding preprocessing pipeline...")
    preprocessor = build_feature_pipeline(train_df)
    
    # Prepare feature columns
    feature_cols = [
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "pickup_is_weekend",
        "haversine_distance_km",
        "passenger_count",
        "store_and_fwd_flag",
        "vendor_id",
    ]
    available_features = [col for col in feature_cols if col in train_df.columns]
    
    X_train = train_df[available_features].copy()
    X_test = test_df[available_features].copy()
    
    # Fit and transform
    logger.info("Fitting preprocessing pipeline on training data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    logger.info("Transforming test data...")
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    
    # Create DataFrames
    train_features_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=train_df.index)
    test_features_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=test_df.index)
    
  
    
    # Save features
    train_features_path = output_dir / "features_train.parquet"
    test_features_path = output_dir / "features_test.parquet"
    
    train_features_df.to_parquet(train_features_path, index=False)
    test_features_df.to_parquet(test_features_path, index=False)
    
    logger.info(f"Saved training features: {train_features_path} ({len(train_features_df)} rows, {len(feature_names)} features)")
    logger.info(f"Saved test features: {test_features_path} ({len(test_features_df)} rows, {len(feature_names)} features)")
    
    # Save preprocessor
    import joblib
    preprocessor_path = output_dir / "feature_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved feature preprocessor: {preprocessor_path}")
    
    logger.info("=" * 60)
    logger.info("Feature engineering complete!")
    logger.info("=" * 60)
    
    return train_features_path, test_features_path, preprocessor_path

