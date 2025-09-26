#!/usr/bin/env python3
"""
Fixed Phase 3: Train ML predictor for chess position evaluation
Handles multi-class target (0.0, 0.5, 1.0) properly
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, 
    classification_report, confusion_matrix, accuracy_score
)
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CalibratedWrapper:
    def __init__(self, base_model, calibrator, scaler=None):
        self.base_model = base_model
        self.calibrator = calibrator
        self.scaler = scaler

    def predict_proba(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        base_probs = self.base_model.predict_proba(X)
        win_probs = base_probs[:, 2]  # Get win probabilities
        calibrated_win_probs = self.calibrator.predict(win_probs)
        # Return probabilities in the same format as base model
        result = base_probs.copy()
        result[:, 2] = calibrated_win_probs
        result[:, :2] *= (1 - calibrated_win_probs)[:, None]  # Adjust other probs
        return result

class FreshChessModelTrainer:
    def __init__(self, data_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Will store our models and metadata
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.results = {}
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load your pre-split train/val/test data"""
        print(f"Loading data from {self.data_dir}...")
        
        # Load your specific files
        train_path = self.data_dir / "train.parquet"
        val_path = self.data_dir / "val.parquet"
        test_path = self.data_dir / "test.parquet"
        
        print("Loading pre-split train/val/test files...")
        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)
        df_test = pd.read_parquet(test_path)
        
        print(f"Train: {len(df_train)} positions")
        print(f"Val: {len(df_val)} positions")
        print(f"Test: {len(df_test)} positions")
        
        # Your metadata columns based on the diagnostic
        meta_cols = ['fen', 'label', 'side_to_move', 'game_result', 'ply', 'game_id']
        self.feature_columns = [col for col in df_train.columns if col not in meta_cols]
        
        # Extract features and target
        X_train = df_train[self.feature_columns].copy()
        X_val = df_val[self.feature_columns].copy()
        X_test = df_test[self.feature_columns].copy()
        
        # Handle the target properly - convert to integer classes
        # 0.0 -> 0 (loss), 0.5 -> 1 (draw), 1.0 -> 2 (win)
        y_train_raw = df_train['label'].copy()
        y_val_raw = df_val['label'].copy()
        y_test_raw = df_test['label'].copy()
        
        # Fit label encoder on all data to ensure consistency
        all_labels = pd.concat([y_train_raw, y_val_raw, y_test_raw])
        self.label_encoder.fit(sorted(all_labels.unique()))
        
        y_train = self.label_encoder.transform(y_train_raw)
        y_val = self.label_encoder.transform(y_val_raw)
        y_test = self.label_encoder.transform(y_test_raw)
        
        # Add game phase info for analysis
        for dataset in [df_train, df_val, df_test]:
            dataset['game_phase'] = pd.cut(dataset['ply'], 
                                           bins=[0, 20, 40, 200], 
                                           labels=['opening', 'middlegame', 'endgame'])
        
        print(f"Features: {len(self.feature_columns)}")
        print(f"Original target distribution:")
        for val, count in zip(*np.unique(y_train_raw, return_counts=True)):
            print(f"  {val} -> {count}")
        print(f"Encoded target distribution:")
        for val, count in zip(*np.unique(y_train, return_counts=True)):
            orig_val = self.label_encoder.inverse_transform([val])[0]
            print(f"  {val} ({orig_val}) -> {count}")
        print(f"Sample features: {self.feature_columns[:10]}")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test), (df_train, df_val, df_test)
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train logistic regression model for multi-class classification"""
        print("\n=== Training Logistic Regression (Multi-class) ===")
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train multi-class logistic regression
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            solver='liblinear',  # Good for multi-class
            multi_class='ovr'    # One-vs-Rest for 3 classes
        )
        
        lr.fit(X_train_scaled, y_train)
        
        # Predictions - get probabilities for all classes
        train_pred_proba = lr.predict_proba(X_train_scaled)
        val_pred_proba = lr.predict_proba(X_val_scaled)
        
        # For win probability, we want P(win) = P(class=2)
        train_pred_win = train_pred_proba[:, 2]  # P(win)
        val_pred_win = val_pred_proba[:, 2]
        
        # For AUC, convert to binary: win vs not-win
        y_train_binary = (y_train == 2).astype(int)  # 1 if win, 0 if draw/loss
        y_val_binary = (y_val == 2).astype(int)
        
        # Metrics
        train_auc = roc_auc_score(y_train_binary, train_pred_win)
        val_auc = roc_auc_score(y_val_binary, val_pred_win)
        train_brier = brier_score_loss(y_train_binary, train_pred_win)
        val_brier = brier_score_loss(y_val_binary, val_pred_win)
        
        # Also compute multi-class accuracy
        train_pred_class = lr.predict(X_train_scaled)
        val_pred_class = lr.predict(X_val_scaled)
        train_acc = accuracy_score(y_train, train_pred_class)
        val_acc = accuracy_score(y_val, val_pred_class)
        
        results = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'train_brier': train_brier,
            'val_brier': val_brier,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'model_type': 'multi_class'
        }
        
        print(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        print(f"Train Brier: {train_brier:.4f}, Val Brier: {val_brier:.4f}")
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        self.models['logistic'] = lr
        self.scalers['logistic'] = scaler
        self.results['logistic'] = results
        
        return lr, scaler, results
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train gradient boosting model for multi-class classification"""
        print("\n=== Training Gradient Boosting (Multi-class) ===")
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        )
        
        gb.fit(X_train, y_train)
        
        # Predictions
        train_pred_proba = gb.predict_proba(X_train)
        val_pred_proba = gb.predict_proba(X_val)
        
        # For win probability, we want P(win) = P(class=2)
        train_pred_win = train_pred_proba[:, 2]
        val_pred_win = val_pred_proba[:, 2]
        
        # For AUC, convert to binary: win vs not-win
        y_train_binary = (y_train == 2).astype(int)
        y_val_binary = (y_val == 2).astype(int)
        
        # Metrics
        train_auc = roc_auc_score(y_train_binary, train_pred_win)
        val_auc = roc_auc_score(y_val_binary, val_pred_win)
        train_brier = brier_score_loss(y_train_binary, train_pred_win)
        val_brier = brier_score_loss(y_val_binary, val_pred_win)
        
        # Multi-class accuracy
        train_pred_class = gb.predict(X_train)
        val_pred_class = gb.predict(X_val)
        train_acc = accuracy_score(y_train, train_pred_class)
        val_acc = accuracy_score(y_val, val_pred_class)
        
        results = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'train_brier': train_brier,
            'val_brier': val_brier,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'feature_importance': dict(zip(self.feature_columns, gb.feature_importances_)),
            'model_type': 'multi_class'
        }
        
        print(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        print(f"Train Brier: {train_brier:.4f}, Val Brier: {val_brier:.4f}")
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Show top features
        importance = gb.feature_importances_
        top_features = sorted(zip(self.feature_columns, importance), 
                            key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 features:")
        for feat, imp in top_features:
            print(f"  {feat}: {imp:.4f}")
        
        self.models['gradient_boosting'] = gb
        self.results['gradient_boosting'] = results
        
        return gb, results
    
    def calibrate_models(self, X_val, y_val):
        """Apply calibration to improve probability estimates"""
        print("\n=== Calibrating Models ===")
        
        # For calibration, we'll focus on the win probability (binary: win vs not-win)
        y_val_binary = (y_val == 2).astype(int)
        
        # Get list of models to calibrate (avoid dictionary changing during iteration)
        models_to_calibrate = [(name, model) for name, model in self.models.items() if 'calibrated' not in name]
        
        for name, model in models_to_calibrate:
            if 'calibrated' in name:
                continue
                
            print(f"Calibrating {name}...")
            
            if name == 'logistic':
                X_val_processed = self.scalers[name].transform(X_val)
            else:
                X_val_processed = X_val
            
            # Get win probabilities from the original model
            pred_before = model.predict_proba(X_val_processed)[:, 2]  # Win probability
            
            # Use isotonic regression for calibration (simpler approach)
            from sklearn.isotonic import IsotonicRegression
            
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(pred_before, y_val_binary)
            
            # Create a simple wrapper for the calibrated model
            scaler_to_use = self.scalers.get(name) if name == 'logistic' else None
            calibrated_wrapper = CalibratedWrapper(model, calibrator, scaler_to_use)
            
            self.models[f'{name}_calibrated'] = calibrated_wrapper
            
            # Test calibration
            pred_after = calibrator.predict(pred_before)
            
            brier_before = brier_score_loss(y_val_binary, pred_before)
            brier_after = brier_score_loss(y_val_binary, pred_after)
            
            print(f"  Brier before: {brier_before:.4f}, after: {brier_after:.4f}")
            self.results[f'{name}_calibrated'] = {
                'val_brier': brier_after,
                'val_brier_improvement': brier_before - brier_after,
                'model_type': 'calibrated_binary'
            }
    
    def evaluate_by_phase(self, X_test, y_test, df_test):
        """Evaluate model performance by game phase"""
        print("\n=== Performance by Game Phase ===")
        
        # Find best model (excluding calibrated ones for now)
        base_models = {k: v for k, v in self.results.items() if 'calibrated' not in k}
        best_model_name = min(base_models.keys(), 
                            key=lambda k: base_models[k].get('val_brier', float('inf')))
        best_model = self.models[best_model_name]
        
        y_test_binary = (y_test == 2).astype(int)  # Win vs not-win
        
        for phase in ['opening', 'middlegame', 'endgame']:
            mask = df_test['game_phase'] == phase
            if not mask.any():
                continue
                
            X_phase = X_test[mask]
            y_phase = y_test_binary[mask]
            
            if 'logistic' in best_model_name:
                scaler = self.scalers.get('logistic')
                if scaler:
                    X_phase = scaler.transform(X_phase)
            
            pred_phase = best_model.predict_proba(X_phase)[:, 2]  # Win probability
            auc_phase = roc_auc_score(y_phase, pred_phase)
            brier_phase = brier_score_loss(y_phase, pred_phase)
            
            print(f"{phase:>12}: AUC={auc_phase:.4f}, Brier={brier_phase:.4f}, n={len(X_phase)}")
    
    def plot_calibration_curves(self, X_val, y_val):
        """Plot calibration curves for all models"""
        print("\n=== Creating Calibration Plots ===")
        
        y_val_binary = (y_val == 2).astype(int)  # Win vs not-win
        
        plt.figure(figsize=(12, 8))
        
        plot_idx = 1
        for name, model in self.models.items():
            if 'calibrated' in name:
                continue
                
            if name == 'logistic':
                X_val_processed = self.scalers[name].transform(X_val)
            else:
                X_val_processed = X_val
            
            pred_proba = model.predict_proba(X_val_processed)
            pred_win = pred_proba[:, 2]  # Win probability
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_val_binary, pred_win, n_bins=10, strategy='quantile'
            )
            
            plt.subplot(1, 2, plot_idx)
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                    label=f"{name}")
            plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(f"Calibration Curve - {name.title()}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(self.models_dir / "calibration_curves.png", dpi=150, bbox_inches='tight')
        print(f"Saved calibration curves to {self.models_dir / 'calibration_curves.png'}")
    
    def select_best_model(self):
        """Select the best model based on validation Brier score"""
        print("\n=== Model Selection ===")
        
        best_name = None
        best_brier = float('inf')
        
        print("Model comparison:")
        for name, results in self.results.items():
            val_brier = results.get('val_brier', float('inf'))
            val_auc = results.get('val_auc', 'N/A')
            val_acc = results.get('val_accuracy', 'N/A')
            print(f"  {name:25}: Brier={val_brier:.4f}, AUC={val_auc}, Acc={val_acc}")
            
            if val_brier < best_brier:
                best_brier = val_brier
                best_name = name
        
        print(f"\nBest model: {best_name} (Brier: {best_brier:.4f})")
        return best_name
    
    def save_model(self, model_name):
        """Save the selected model and metadata"""
        print(f"\n=== Saving {model_name} ===")
        
        model = self.models[model_name]
        scaler = self.scalers.get('logistic' if 'logistic' in model_name else None)
        
        # Save model
        model_path = self.models_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Save scaler if needed
        if scaler:
            scaler_path = self.models_dir / "scaler.joblib"
            joblib.dump(scaler, scaler_path)
        
        # Save label encoder
        encoder_path = self.models_dir / "label_encoder.joblib"
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save metadata
        metadata = {
            'model_type': model_name,
            'features': self.feature_columns,
            'feature_count': len(self.feature_columns),
            'training_date': datetime.now().isoformat(),
            'performance': self.results[model_name],
            'needs_scaling': 'logistic' in model_name,
            'model_version': '1.0',
            'target_column': 'label',
            'label_mapping': {
                '0': '0.0 (loss)',
                '1': '0.5 (draw)', 
                '2': '1.0 (win)'
            },
            'data_info': 'Multi-class (0.0=loss, 0.5=draw, 1.0=win) encoded as integers (0, 1, 2)',
            'usage_notes': 'Use predict_proba()[:, 2] for win probability'
        }
        
        meta_path = self.models_dir / "model_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {meta_path}")
        print(f"Label encoder saved to: {encoder_path}")
        if scaler:
            print(f"Scaler saved to: {scaler_path}")

def main():
    """Main training pipeline"""
    print("🚀 Starting Fresh Chess ML Training Pipeline")
    print("=" * 50)
    
    trainer = FreshChessModelTrainer()
    
    # Load pre-split data
    (X_train, X_val, X_test), (y_train, y_val, y_test), (df_train, df_val, df_test) = trainer.load_data()
    
    # Train models
    trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
    trainer.train_gradient_boosting(X_train, y_train, X_val, y_val)
    
    # Calibrate models
    trainer.calibrate_models(X_val, y_val)
    
    # Evaluate
    trainer.evaluate_by_phase(X_test, y_test, df_test)
    trainer.plot_calibration_curves(X_val, y_val)
    
    # Select and save best model
    best_model = trainer.select_best_model()
    trainer.save_model(best_model)
    
    print("\n✅ Training complete! Your model is ready for Phase 4.")
    print("🎯 Next step: Integrate with search engine")
    print("\n📝 Usage notes:")
    print("  - Load model with joblib.load('models/model.joblib')")
    print("  - For win probability: model.predict_proba(features)[:, 2]")
    print("  - Check model_meta.json for feature requirements")

if __name__ == "__main__":
    main()