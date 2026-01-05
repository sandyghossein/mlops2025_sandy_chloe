import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Clean NYC taxi trip data by removing outliers and invalid entries.
    
    Args:
        df: Raw dataframe
        is_train: Whether this is training data (has trip_duration column)
    
    Returns:
        Cleaned dataframe
    """
    df = df.copy()
    initial_count = len(df)
    
    print(f"Initial rows: {initial_count}")
    
    # 1. Remove trips with invalid duration (only for training data)
    if is_train and 'trip_duration' in df.columns:
        # Remove very short trips (< 60 seconds)
        df = df[df['trip_duration'] >= 60]
        print(f"After removing trips < 60s: {len(df)} ({initial_count - len(df)} removed)")
        
        # Remove very long trips (> 3 hours = 10800 seconds)
        df = df[df['trip_duration'] <= 10800]
        print(f"After removing trips > 3h: {len(df)} ({initial_count - len(df)} removed total)")
    
    # 2. Remove trips with 0 passengers
    before_passenger = len(df)
    df = df[df['passenger_count'] > 0]
    print(f"After removing 0 passengers: {len(df)} ({before_passenger - len(df)} removed)")
    
    # 3. Remove trips with coordinates outside NYC bounds
    # NYC approximate bounds:
    # Latitude: 40.5 to 41.0
    # Longitude: -74.3 to -73.7
    before_coords = len(df)
    df = df[
        (df['pickup_latitude'].between(40.5, 41.0)) &
        (df['pickup_longitude'].between(-74.3, -73.7)) &
        (df['dropoff_latitude'].between(40.5, 41.0)) &
        (df['dropoff_longitude'].between(-74.3, -73.7))
    ]
    print(f"After coordinate filtering: {len(df)} ({before_coords - len(df)} removed)")
    
    print(f"Final rows: {len(df)} (removed {initial_count - len(df)} total, {100 * (initial_count - len(df)) / initial_count:.1f}%)")
    
    return df