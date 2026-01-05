"""Standalone training script."""

import argparse
import logging
import sys
from pathlib import Path

from mlproject.train.models import train_models

# Simple logging (no utils needed)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Train regression models")

    parser.add_argument("--features_train", type=str, required=True, help="Path to training features parquet (X only)")
    parser.add_argument("--clean_train", type=str, required=True, help="Path to cleaned training CSV (has trip_duration)")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory (default: outputs/)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation set size (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random state (default: 42)")

    args = parser.parse_args()

    features_train_path = Path(args.features_train).resolve()
    clean_train_path = Path(args.clean_train).resolve()
    output_dir = Path(args.out_dir).resolve()

    if not features_train_path.exists():
        logger.error(f"Training features file not found: {features_train_path}")
        sys.exit(1)

    if not clean_train_path.exists():
        logger.error(f"Cleaned training CSV not found: {clean_train_path}")
        sys.exit(1)

    try:
        model_path = train_models(
            features_train_path=features_train_path,
            clean_train_path=clean_train_path,
            output_dir=output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        logger.info("Training complete!")
        logger.info(f"Model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
