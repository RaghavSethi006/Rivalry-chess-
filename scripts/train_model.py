#!/usr/bin/env python3
"""
Simple script to run the complete Phase 3 training pipeline
Usage: python scripts/train_model.py [--data-path path/to/positions.parquet]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.train import ChessModelTrainer
from ml.metrics import ChessModelAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Train chess evaluation model")
    parser.add_argument(
        "--data-dir", 
        default="data/processed",
        help="Directory containing train.parquet, val.parquet, test.parquet"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Skip calibration and detailed analysis for faster training"
    )
    parser.add_argument(
        "--analyze-only", 
        action="store_true",
        help="Skip training, just analyze existing model"
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        print("🔍 Analyzing existing model...")
        analyzer = ChessModelAnalyzer()
        try:
            analyzer.load_model_and_data()
            analyzer.model_summary_report()
            
            # Load some test data for speed benchmark
            import pandas as pd
            data_dir = Path(args.data_dir)
            test_file = data_dir / "test.parquet"
            if test_file.exists():
                df = pd.read_parquet(test_file)
            else:
                df = pd.read_parquet(data_dir / "fulldataset.parquet")
                
            meta_cols = ['fen', 'side_to_move', 'result', 'ply', 'game_id', 'target']
            feature_cols = [col for col in df.columns if col not in meta_cols]
            X_sample = df[feature_cols].iloc[:1000]  # Sample for speed test
            
            analyzer.benchmark_prediction_speed(X_sample)
            
        except Exception as e:
            print(f"Error analyzing model: {e}")
            return 1
        
        return 0
    
    # Check if data exists
    data_dir = Path(args.data_dir)
    train_file = data_dir / "train.parquet"
    full_file = data_dir / "full_dataset.parquet"  # Updated to match your filename
    alt_full_file = data_dir / "fulldataset.parquet"
    
    if not (train_file.exists() or full_file.exists() or alt_full_file.exists()):
        print(f"❌ Error: No training data found in {data_dir}")
        print("Expected files: train.parquet, val.parquet, test.parquet")
        print("           OR: full_dataset.parquet")
        print("           OR: fulldataset.parquet")
        print("Make sure you've completed Phase 2 (data preparation)")
        return 1
    
    print("🚀 Starting Chess ML Training Pipeline")
    print("=" * 50)
    print(f"📊 Data directory: {data_dir}")
    print(f"⚡ Quick mode: {args.quick}")
    print()
    
    try:
        # Initialize trainer with data directory
        trainer = ChessModelTrainer(str(data_dir))
        
        # Load data (handles both pre-split and single file)
        data_result = trainer.load_data()
        
        if len(data_result) == 3 and isinstance(data_result[0], tuple):
            # Pre-split data
            (X_train, X_val, X_test), (y_train, y_val, y_test), (df_train, df_val, df_test) = data_result
            print("Using pre-split train/val/test data")
        else:
            # Single file data - need to split
            X, y, df = data_result
            (X_train, X_val, X_test), (y_train, y_val, y_test), (df_train, df_val, df_test) = trainer.split_data(X, y, df)
        
        # Train baseline models
        print("\n🤖 Training baseline models...")
        trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
        trainer.train_gradient_boosting(X_train, y_train, X_val, y_val)
        
        if not args.quick:
            # Calibrate models
            print("\n🎯 Calibrating models...")
            trainer.calibrate_models(X_val, y_val)
            
            # Detailed evaluation
            print("\n📈 Running detailed evaluation...")
            trainer.evaluate_by_phase(X_test, y_test, df_test)
            trainer.plot_calibration_curves(X_val, y_val)
        
        # Select and save best model
        print("\n💾 Saving best model...")
        best_model = trainer.select_best_model()
        trainer.save_model(best_model)
        
        # Final sanity check
        trainer.quick_sanity_check()
        
        # Run quick analysis
        print("\n📊 Quick model analysis...")
        analyzer = ChessModelAnalyzer()
        analyzer.load_model_and_data()
        
        # Test on a sample
        X_sample = X_test.iloc[:1000] if len(X_test) > 1000 else X_test  # type: ignore
        y_sample = y_test.iloc[:1000] if len(y_test) > 1000 else y_test
        
        metrics, predictions = analyzer.comprehensive_metrics(X_sample, y_sample, "test_sample")
        analyzer.benchmark_prediction_speed(X_sample)
        
        if not args.quick:
            # Create analysis plots
            analyzer.plot_comprehensive_analysis(
                X_sample, y_sample, 
                save_path="models/model_analysis.png"
            )
        
        print("\n✅ Training completed successfully!")
        print("🎯 Your model is ready for Phase 4 (integration with search engine)")
        print(f"📁 Model files saved in: models/")
        print(f"📈 Analysis plots: models/model_analysis.png")
        
        # Next steps
        print("\n🔧 Next steps:")
        print("1. Integrate model with your search engine (Phase 4)")
        print("2. Test the engine: python -c \"from engine.search import ChessEngine; engine = ChessEngine(); print(engine.get_best_move('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))\"")
        print("3. Start the API server: cd api && python main.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())