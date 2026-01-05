import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("feature_engineering")

import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlproject.features.transformers import engineer_features


def main():
    """Main entry point for feature engineering script."""
    parser = argparse.ArgumentParser(description="Engineer features from cleaned data")
    parser.add_argument(
        "--clean_train",
        type=str,
        required=True,
        help="Path to cleaned training CSV",
    )
    parser.add_argument(
        "--clean_test",
        type=str,
        required=True,
        help="Path to cleaned test CSV",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    output_dir = Path(args.out_dir).resolve()
    clean_train_path = Path(args.clean_train).resolve()
    clean_test_path = Path(args.clean_test).resolve()
    
    # Check if files exist
    if not clean_train_path.exists():
        logger.error(f"Cleaned training file not found: {clean_train_path}")
        sys.exit(1)
    
    if not clean_test_path.exists():
        logger.error(f"Cleaned test file not found: {clean_test_path}")
        sys.exit(1)
    
    try:
        features_train_path, features_test_path, preprocessor_path = engineer_features(
            clean_train_path, clean_test_path, output_dir
        )
        logger.info(f"\nFeature engineering complete!")
        logger.info(f"Training features: {features_train_path}")
        logger.info(f"Test features: {features_test_path}")
        logger.info(f"Preprocessor: {preprocessor_path}")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

