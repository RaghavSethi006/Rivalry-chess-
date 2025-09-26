#!/usr/bin/env python3
"""
Quick test to see if we can load the training data
"""

import pandas as pd
from pathlib import Path

def test_data_loading():
    data_dir = Path("data/processed")
    
    print("🔍 Testing data loading...")
    print(f"Data directory: {data_dir}")
    print(f"Directory exists: {data_dir.exists()}")
    
    # List all files
    if data_dir.exists():
        all_files = list(data_dir.iterdir())
        print(f"\nAll files in directory:")
        for f in all_files:
            print(f"  {f.name} ({'file' if f.is_file() else 'dir'})")
        
        # Filter parquet files
        parquet_files = [f for f in all_files if f.suffix == '.parquet']
        print(f"\nParquet files found:")
        for f in parquet_files:
            print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Try loading each parquet file
    files_to_test = ['train.parquet', 'val.parquet', 'test.parquet', 'full_dataset.parquet']
    
    for filename in files_to_test:
        file_path = data_dir / filename
        if file_path.exists():
            try:
                print(f"\n✅ Testing {filename}...")
                df = pd.read_parquet(file_path)
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)[:10]}... (showing first 10)")
                
                # Check for target column
                if 'label' in df.columns:
                    print(f"   Label distribution: {df['label'].value_counts().to_dict()}")
                    print(f"   Label range: {df['label'].min():.3f} - {df['label'].max():.3f}")
                elif 'target' in df.columns:
                    print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
                elif 'result' in df.columns:
                    print(f"   Result distribution: {df['result'].value_counts().to_dict()}")
                
                # Show feature columns (non-metadata)
                meta_cols = ['fen', 'side_to_move', 'label', 'game_result', 'ply', 'game_id', 'target', 'result']
                feature_cols = [col for col in df.columns if col not in meta_cols]
                print(f"   Feature columns: {len(feature_cols)}")
                print(f"   Sample features: {feature_cols[:10]}")
                
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        else:
            print(f"⚪ {filename}: Not found")

def test_training_setup():
    """Test that we can import and set up training"""
    print(f"\n🚀 Testing training setup...")
    
    try:
        # Add current directory to path for imports
        import sys
        sys.path.append('.')
        
        from ml.train import ChessModelTrainer
        
        trainer = ChessModelTrainer("data/processed")
        print("✅ Trainer initialized successfully")
        
        # Test data loading
        data_result = trainer.load_data()
        print("✅ Data loaded successfully")
        
        if isinstance(data_result[0], tuple):
            (X_train, X_val, X_test), (y_train, y_val, y_test), (df_train, df_val, df_test) = data_result
            print(f"✅ Pre-split data detected:")
            print(f"   Train: {len(X_train)} samples, {X_train.shape[1]} features")
            print(f"   Val: {len(X_val)} samples")  
            print(f"   Test: {len(X_test)} samples")
            print(f"   Target range: {y_train.min():.3f} - {y_train.max():.3f}")
        else:
            X, y, df = data_result
            print(f"✅ Single dataset detected:")
            print(f"   Total: {len(X)} samples, {len(X.columns)} features")
            print(f"   Target range: {y.min():.3f} - {y.max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_loading()
    test_training_setup()