"""
Quick script to inspect the created dataset
"""
import pandas as pd
import json
from pathlib import Path

def inspect_dataset():
    data_dir = Path("data/processed")
    
    # Load metadata
    if (data_dir / "metadata.json").exists():
        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)
        print("=== Dataset Metadata ===")
        print(f"Feature count: {metadata['feature_count']}")
        print(f"Dataset sizes: {metadata['dataset_stats']}")
        print()
    
    # Load and inspect train set
    if (data_dir / "train.parquet").exists():
        train_df = pd.read_parquet(data_dir / "train.parquet")
        print("=== Training Set Inspection ===")
        print(f"Shape: {train_df.shape}")
        print(f"Columns: {list(train_df.columns[:10])}...")  # First 10 columns
        print()
        
        # Label distribution
        print("Label distribution:")
        label_counts = train_df['label'].round(1).value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = "Loss" if label == 0.0 else "Draw" if label == 0.5 else "Win"
            print(f"  {label_name} ({label}): {count}")
        print()
        
        # Feature columns (excluding metadata)
        feature_cols = [col for col in train_df.columns 
                       if col.startswith(('my_', 'opp_', 'material_', 'mobility_', 'center_', 'development', 'is_endgame', 'ply_number'))]
        print(f"Feature columns ({len(feature_cols)}):")
        print(f"  {feature_cols[:15]}...")  # First 15 features
        print()
        
        # Sample some feature values
        print("Sample feature values (first position):")
        sample_features = {col: train_df[col].iloc[0] for col in feature_cols[:10]}
        for feat, val in sample_features.items():
            print(f"  {feat}: {val:.3f}")
        
        # ELO distribution
        if 'avg_elo' in train_df.columns:
            print(f"\nELO range: {train_df['avg_elo'].min():.0f} - {train_df['avg_elo'].max():.0f}")
            print(f"Average ELO: {train_df['avg_elo'].mean():.0f}")
        
    else:
        print("No dataset found. Please run make_dataset.py first.")

if __name__ == "__main__":
    inspect_dataset()