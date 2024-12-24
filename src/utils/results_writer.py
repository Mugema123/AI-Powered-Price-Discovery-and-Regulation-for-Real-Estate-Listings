# src/utils/results_writer.py

import pandas as pd
import os
import numpy as np

def save_results_summary(results, price_ranges, best_model_name, logger):
    """Save a comprehensive results summary."""
    # Create reports directory if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    # Create summary file
    with open('reports/analysis_summary.txt', 'w') as f:
        # Write model comparison
        f.write("="*50 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("="*50 + "\n\n")
        
        for name, metrics in results.items():
            f.write(f"Model: {name}\n")
            f.write("-"*len(f"Model: {name}") + "\n")
            
            train_mae, train_rmse, train_r2 = metrics['train_metrics']
            test_mae, test_rmse, test_r2 = metrics['test_metrics']
            
            f.write(f"Training Metrics:\n")
            f.write(f"  MAE:  {train_mae:.2f} lakhs\n")
            f.write(f"  RMSE: {train_rmse:.2f} lakhs\n")
            f.write(f"  R2:   {train_r2:.4f}\n\n")
            
            f.write(f"Test Metrics:\n")
            f.write(f"  MAE:  {test_mae:.2f} lakhs\n")
            f.write(f"  RMSE: {test_rmse:.2f} lakhs\n")
            f.write(f"  R2:   {test_r2:.4f}\n\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write("="*50 + "\n\n")
        
        # Write prediction statistics
        prediction_error = abs(price_ranges['Actual'] - price_ranges['Predicted'])
        f.write("PREDICTION STATISTICS\n")
        f.write("-"*20 + "\n")
        f.write(f"Average Prediction Error: {prediction_error.mean():.2f} lakhs\n")
        f.write(f"Error Standard Deviation: {prediction_error.std():.2f} lakhs\n")
        f.write(f"90% of predictions are within: ±{np.percentile(prediction_error, 90):.2f} lakhs\n\n")
        
        # Write interval statistics
        interval_width = price_ranges['Upper_Bound'] - price_ranges['Lower_Bound']
        coverage = np.mean((price_ranges['Actual'] >= price_ranges['Lower_Bound']) & 
                         (price_ranges['Actual'] <= price_ranges['Upper_Bound'])) * 100
        
        f.write("CONFIDENCE INTERVAL STATISTICS\n")
        f.write("-"*30 + "\n")
        f.write(f"Average Interval Width: {interval_width.mean():.2f} lakhs\n")
        f.write(f"Actual Coverage: {coverage:.1f}%\n\n")
        
        # Write sample predictions
        f.write("SAMPLE PREDICTIONS\n")
        f.write("-"*17 + "\n")
        f.write(price_ranges.head(10).to_string())
        
    logger.info(f"Results summary saved to reports/analysis_summary.txt")