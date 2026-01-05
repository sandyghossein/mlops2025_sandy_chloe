"""Standalone batch inference script."""

import argparse
import logging
import sys
from pathlib import Path

from mlproject.inference.predict import batch_inference

# Simple logging (no utils)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("batch_inference")


def main():
    parser = argparse.ArgumentParser(description="Run batch inference on test features")
    parser.add_argument(
        "--features_test",
        type=str,
        required=True,
        help="Path to test features parquet (e.g., outputs/features_test.parquet)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/model.joblib",
        help="Path to trained model (default: outputs/model.joblib)",
    )

    args = parser.parse_args()

    features_test_path = Path(args.features_test).resolve()
    output_dir = Path(args.out_dir).resolve()
    model_path = Path(args.model_path).resolve()

    if not features_test_path.exists():
        logger.error(f"Test features file not found: {features_test_path}")
        sys.exit(1)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Train a model first (train stage).")
        sys.exit(1)

    try:
        preds_path = batch_inference(
            features_test_path=features_test_path,
            output_dir=output_dir,
            model_path=model_path,
        )
        logger.info(f"Predictions saved to: {preds_path}")
    except Exception as e:
        logger.error(f"Batch inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
