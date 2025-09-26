#!/usr/bin/env python3
"""
Comprehensive metrics and analysis for chess position evaluation models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import joblib
from pathlib import Path

class ChessModelAnalyzer:
    """Comprehensive analysis of chess evaluation models"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_model_and_data(self, test_data_path=None):
        """Load saved model and test data"""
        # Load model
        model_path = self.models_dir / "model.joblib"
        meta_path = self.models_dir / "model_meta.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")
            
        self.model = joblib.load(model_path)
        
        with open(meta_path, 'r') as f:
            import json
            self.metadata = json.load(f)
        
        self.scaler = None
        if self.metadata.get('needs_scaling'):
            scaler_path = self.models_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
    
    def comprehensive_metrics(self, X_true, y_true, sample_name="test"):
        """Calculate comprehensive metrics for a dataset"""
        # Get predictions
        if self.scaler:
            X_processed = self.scaler.transform(X_true)
        else:
            X_processed = X_true
            
        y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        # Core metrics
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'accuracy': (y_pred_binary == y_true).mean(),
            'sample_size': len(y_true)
        }
        
        # Calibration metrics
        cal_slope, cal_intercept = self._calibration_slope_intercept(y_true, y_pred_proba)
        metrics['calibration_slope'] = cal_slope
        metrics['calibration_intercept'] = cal_intercept
        
        # Class-wise metrics for multi-class if needed
        if len(np.unique(y_true)) > 2:
            metrics['classification_report'] = classification_report(
                y_true, y_pred_binary, output_dict=True
            )
        
        print(f"\n=== {sample_name.upper()} METRICS ===")
        print(f"AUC-ROC:        {metrics['auc']:.4f}")
        print(f"Brier Score:    {metrics['brier_score']:.4f} (lower is better)")
        print(f"Log Loss:       {metrics['log_loss']:.4f}")
        print(f"Accuracy:       {metrics['accuracy']:.4f}")
        print(f"Cal. Slope:     {cal_slope:.4f} (1.0 is perfect)")
        print(f"Cal. Intercept: {cal_intercept:.4f} (0.0 is perfect)")
        print(f"Sample Size:    {metrics['sample_size']:,}")
        
        return metrics, y_pred_proba
    
    def _calibration_slope_intercept(self, y_true, y_pred_proba, n_bins=10):
        """Calculate calibration slope and intercept"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy='quantile'
        )
        
        # Fit line: true_freq = slope * predicted_freq + intercept
        valid_mask = ~np.isnan(fraction_of_positives) & ~np.isnan(mean_predicted_value)
        if np.sum(valid_mask) >= 2:
            slope, intercept = np.polyfit(
                mean_predicted_value[valid_mask], 
                fraction_of_positives[valid_mask], 
                1
            )
            return slope, intercept
        return np.nan, np.nan
    
    def analyze_by_game_phase(self, X, y, game_phases):
        """Analyze performance by opening/middlegame/endgame"""
        print("\n=== PERFORMANCE BY GAME PHASE ===")
        
        phases = ['opening', 'middlegame', 'endgame']
        phase_results = {}
        
        for phase in phases:
            mask = game_phases == phase
            if not mask.any():
                continue
                
            X_phase = X[mask]
            y_phase = y[mask]
            
            metrics, pred_proba = self.comprehensive_metrics(
                X_phase, y_phase, f"{phase}"
            )
            phase_results[phase] = {
                'metrics': metrics,
                'predictions': pred_proba,
                'true_labels': y_phase
            }
        
        return phase_results
    
    def analyze_by_prediction_confidence(self, X, y, bins=5):
        """Analyze performance by model confidence levels"""
        print(f"\n=== PERFORMANCE BY CONFIDENCE BINS ===")
        
        if self.scaler:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
            
        y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
        
        # Create confidence bins
        confidence = np.abs(y_pred_proba - 0.5)  # Distance from 0.5
        bin_edges = np.linspace(0, 0.5, bins + 1)
        bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(bins)]
        
        confidence_bins = pd.cut(confidence, bin_edges.tolist(), labels=bin_labels, include_lowest=True)
        
        for bin_label in bin_labels:
            mask = confidence_bins == bin_label
            if not mask.any():
                continue
                
            bin_pred = y_pred_proba[mask]
            bin_true = y[mask]
            
            if len(bin_true) > 10:  # Only analyze if enough samples
                auc = roc_auc_score(bin_true, bin_pred) if len(np.unique(bin_true)) > 1 else np.nan
                brier = brier_score_loss(bin_true, bin_pred)
                accuracy = ((bin_pred > 0.5) == bin_true).mean()
                
                print(f"Confidence {bin_label}: n={len(bin_true):,}, "
                      f"AUC={auc:.4f}, Brier={brier:.4f}, Acc={accuracy:.4f}")
    
    def plot_comprehensive_analysis(self, X, y, save_path=None):
        """Create comprehensive analysis plots"""
        if self.scaler:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
            
        y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), squeeze=False)
        fig.suptitle('Chess Model Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        axes[0, 1].plot(recall, precision, linewidth=2)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Calibration Curve
        fraction_pos, mean_pred = calibration_curve(y, y_pred_proba, n_bins=10)
        axes[0, 2].plot(mean_pred, fraction_pos, 's-', linewidth=2, label='Model')
        axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        axes[0, 2].set_xlabel('Mean Predicted Probability')
        axes[0, 2].set_ylabel('Fraction of Positives')
        axes[0, 2].set_title('Calibration Curve')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        axes[1, 0].hist(y_pred_proba, bins=50, alpha=0.7, density=True)
        axes[1, 0].axvline(0.5, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Reliability Diagram (binned)
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y[in_bin].mean()
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
                bin_counts.append(in_bin.sum())
        
        axes[1, 1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, 
                       color='skyblue', edgecolor='black', linewidth=0.5)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title('Reliability Diagram')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            importances = self.model.feature_importances_
            feature_names = self.metadata.get('features', [f'feature_{i}' for i in range(len(importances))])
            
            # Get top 15 features
            top_indices = np.argsort(importances)[-15:]
            top_names = [feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            axes[1, 2].barh(range(len(top_importances)), top_importances)
            axes[1, 2].set_yticks(range(len(top_importances)))
            axes[1, 2].set_yticklabels(top_names, fontsize=8)
            axes[1, 2].set_xlabel('Importance')
            axes[1, 2].set_title('Top Feature Importances')
            
        elif hasattr(self.model, 'coef_'):
            # For linear models
            coefs = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            feature_names = self.metadata.get('features', [f'feature_{i}' for i in range(len(coefs))])
            
            # Get top 15 by absolute coefficient
            top_indices = np.argsort(np.abs(coefs))[-15:]
            top_names = [feature_names[i] for i in top_indices]
            top_coefs = coefs[top_indices]
            
            colors = ['red' if c < 0 else 'blue' for c in top_coefs]
            axes[1, 2].barh(range(len(top_coefs)), top_coefs, color=colors, alpha=0.7)
            axes[1, 2].set_yticks(range(len(top_coefs)))
            axes[1, 2].set_yticklabels(top_names, fontsize=8)
            axes[1, 2].set_xlabel('Coefficient')
            axes[1, 2].set_title('Top Feature Coefficients')
            axes[1, 2].axvline(0, color='black', linestyle='-', alpha=0.3)
            
        else:
            axes[1, 2].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model type', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Analysis plots saved to: {save_path}")
        
        return fig
    
    def error_analysis(self, X, y, top_k=20):
        """Analyze worst predictions for debugging"""
        print(f"\n=== ERROR ANALYSIS (Top {top_k} worst predictions) ===")
        
        if self.scaler:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
            
        y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
        
        # Calculate absolute prediction errors
        errors = np.abs(y - y_pred_proba)
        worst_indices = np.argsort(errors)[-top_k:][::-1]  # Worst first
        
        print(f"{'Rank':>4} {'True':>5} {'Pred':>6} {'Error':>6} {'Index':>7}")
        print("-" * 35)
        
        for i, idx in enumerate(worst_indices):
            true_val = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
            pred_val = y_pred_proba[idx]
            error = errors[idx]
            
            print(f"{i+1:>4} {true_val:>5.2f} {pred_val:>6.3f} {error:>6.3f} {idx:>7}")
        
        return worst_indices
    
    def model_summary_report(self):
        """Generate a comprehensive model summary"""
        print("\n" + "="*60)
        print("               CHESS MODEL SUMMARY REPORT")
        print("="*60)
        
        print(f"Model Type:       {self.metadata.get('model_type', 'Unknown')}")
        print(f"Training Date:    {self.metadata.get('training_date', 'Unknown')}")
        print(f"Model Version:    {self.metadata.get('model_version', 'Unknown')}")
        print(f"Feature Count:    {self.metadata.get('feature_count', 'Unknown')}")
        print(f"Requires Scaling: {self.metadata.get('needs_scaling', False)}")
        
        if 'performance' in self.metadata:
            perf = self.metadata['performance']
            print(f"\nValidation Performance:")
            print(f"  AUC:          {perf.get('val_auc', 'N/A')}")
            print(f"  Brier Score:  {perf.get('val_brier', 'N/A')}")
        
        print("\nModel is ready for integration with chess engine!")
        print("Next step: Phase 4 - Integrate with search algorithm")
    
    def benchmark_prediction_speed(self, X_sample, n_trials=100):
        """Benchmark model prediction speed"""
        print(f"\n=== PREDICTION SPEED BENCHMARK ===")
        
        if len(X_sample) > 1000:
            X_sample = X_sample[:1000]  # Limit sample size
        
        if self.scaler:
            X_processed = self.scaler.transform(X_sample)
        else:
            X_processed = X_sample
        
        import time
        
        # Warm up
        for _ in range(10):
            _ = self.model.predict_proba(X_processed)
        
        # Benchmark
        times = []
        for _ in range(n_trials):
            start = time.time()
            _ = self.model.predict_proba(X_processed)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        predictions_per_sec = len(X_sample) / avg_time
        
        print(f"Sample Size:        {len(X_sample):,}")
        print(f"Average Time:       {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"Predictions/sec:    {predictions_per_sec:,.0f}")
        print(f"Time per position:  {avg_time/len(X_sample)*1000:.3f} ms")
        
        # Speed assessment for chess engine
        if predictions_per_sec > 10000:
            speed_rating = "EXCELLENT - Perfect for real-time chess"
        elif predictions_per_sec > 5000:
            speed_rating = "GOOD - Suitable for chess with batching"
        elif predictions_per_sec > 1000:
            speed_rating = "ADEQUATE - May need optimization"
        else:
            speed_rating = "SLOW - Consider model simplification"
        
        print(f"Speed Rating:       {speed_rating}")


def main():
    """Example usage of the metrics analyzer"""
    analyzer = ChessModelAnalyzer()
    
    try:
        # Load model
        analyzer.load_model_and_data()
        print("Model loaded successfully!")
        
        # Generate summary report
        analyzer.model_summary_report()
        
        # Note: In real usage, you would load your test data here
        print("\nTo run full analysis, load your test data and call:")
        print("metrics, predictions = analyzer.comprehensive_metrics(X_test, y_test)")
        print("analyzer.plot_comprehensive_analysis(X_test, y_test, 'analysis.png')")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run the training script first!")

if __name__ == "__main__":
    main()