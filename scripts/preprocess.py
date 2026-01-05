#!/usr/bin/env python3
"""
Preprocessing script for NYC taxi trip data.
Cleans training and test datasets and saves to data/processed/
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path so we can import our module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlproject.preprocess.clean import clean_data


def main():
    print("=" * 70)
    print("NYC TAXI TRIP PREPROCESSING")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print(f"✅ Loaded {len(train)} training rows and {len(test)} test rows")
    
    # Clean training data
    print("\n🧹 Cleaning training data...")
    print("-" * 70)
    train_clean = clean_data(train, is_train=True)
    
    # Clean test data
    print("\n🧹 Cleaning test data...")
    print("-" * 70)
    test_clean = clean_data(test, is_train=False)
    
    # Save cleaned data
    print("\n💾 Saving cleaned data...")
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    
    train_clean.to_csv(output_dir / 'train_clean.csv', index=False)
    test_clean.to_csv(output_dir / 'test_clean.csv', index=False)
    
    print(f"✅ Saved to {output_dir}/")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Training: {len(train):,} → {len(train_clean):,} rows ({100 * len(train_clean) / len(train):.1f}% retained)")
    print(f"Test:     {len(test):,} → {len(test_clean):,} rows ({100 * len(test_clean) / len(test):.1f}% retained)")
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()