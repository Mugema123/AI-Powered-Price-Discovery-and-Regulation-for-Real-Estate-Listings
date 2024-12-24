


# src/main.py
from utils.results_writer import save_results_summary
# src/main.py

from data.data_preprocessing import DataPreprocessor
from features.feature_engineering import FeatureEngineer
from models.model_training import ModelTrainer
from models.model_evaluation import ModelEvaluator
from visualization.visualize import DataVisualizer
from utils.logger_config import setup_logger
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

def print_model_results(results, logger):
    """Print detailed results for all models and save to file."""
    logger.info("\n" + "="*50)
    logger.info("DETAILED MODEL PERFORMANCE METRICS")
    logger.info("="*50)
    
    # Create a DataFrame for model results
    models_df = pd.DataFrame(columns=['Model', 'Train MAE', 'Train RMSE', 'Train R2', 
                                    'Test MAE', 'Test RMSE', 'Test R2', 'Generalization Score'])
    
    # Track best model based on generalization and performance
    best_combined_score = float('inf')
    best_model = None
    
    for name, metrics in results.items():
        # Get metrics
        train_mae, train_rmse, train_r2 = metrics['train_metrics']
        test_mae, test_rmse, test_r2 = metrics['test_metrics']
        
        # Calculate generalization metrics
        rmse_diff = abs(train_rmse - test_rmse)
        rmse_ratio = max(train_rmse, test_rmse) / (min(train_rmse, test_rmse) + 1e-10)
        r2_diff = abs(train_r2 - test_r2)
        
        # Penalize severe overfitting (when training metrics are too good)
        overfitting_penalty = 0
        if train_rmse < 1.0 or train_r2 > 0.99:  # Signs of severe overfitting
            overfitting_penalty = 100  # Large penalty for unrealistic training performance
        
        # Combined score (lower is better)
        generalization_score = (
            0.3 * test_rmse +          # Test performance
            0.3 * rmse_diff +          # RMSE stability
            0.2 * (rmse_ratio - 1) +   # Ratio between train and test
            0.2 * r2_diff +            # R² stability
            overfitting_penalty        # Penalty for unrealistic performance
        )
        
        # Add to DataFrame
        models_df.loc[len(models_df)] = [
            name, train_mae, train_rmse, train_r2,
            test_mae, test_rmse, test_r2, generalization_score
        ]
        
        # Log individual model results
        logger.info(f"\nModel: {name}")
        logger.info("-" * (len(name) + 7))
        
        logger.info("Training Metrics:")
        logger.info(f"  MAE:  {train_mae:.2f} lakhs")
        logger.info(f"  RMSE: {train_rmse:.2f} lakhs")
        logger.info(f"  R2:   {train_r2:.4f}")
        
        logger.info("\nTest Metrics:")
        logger.info(f"  MAE:  {test_mae:.2f} lakhs")
        logger.info(f"  RMSE: {test_rmse:.2f} lakhs")
        logger.info(f"  R2:   {test_r2:.4f}")
        
        logger.info("\nGeneralization Metrics:")
        logger.info(f"  RMSE Difference: {rmse_diff:.2f} lakhs")
        logger.info(f"  RMSE Ratio: {rmse_ratio:.2f}")
        logger.info(f"  R² Stability: {r2_diff:.4f}")
        logger.info(f"  Generalization Score: {generalization_score:.4f}")
        
        # Track best model
        if generalization_score < best_combined_score:
            best_combined_score = generalization_score
            best_model = name
            
    # Sort models by generalization score
    models_df = models_df.sort_values('Generalization Score')
    
    # Save model results to CSV
    os.makedirs('reports', exist_ok=True)
    models_df.to_csv('reports/model_comparison.csv', index=False)
    logger.info("\nModel Comparison Results (saved to reports/model_comparison.csv):")
    logger.info("\n" + str(models_df.to_string()))
    
    logger.info("\n" + "="*50)
    logger.info(f"Best Model (Most Robust): {best_model}")
    logger.info(f"Generalization Score: {best_combined_score:.4f}")
    logger.info("="*50)
    
    return best_model

def save_results_summary(results, price_ranges, best_model_name, logger):
    """Save a comprehensive results summary."""
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
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

def save_detailed_predictions(price_ranges, logger):
    """Save and display detailed prediction analysis."""
    prediction_error = np.abs(price_ranges['Actual'] - price_ranges['Predicted'])
    
    # Calculate statistics
    stats = {
        'Average Error': prediction_error.mean(),
        'Error Std': prediction_error.std(),
        '90th Percentile Error': np.percentile(prediction_error, 90),
        'Median Error': np.median(prediction_error)
    }
    
    # Save detailed predictions
    price_ranges['Absolute_Error'] = prediction_error
    price_ranges.to_csv('reports/detailed_predictions.csv', index=False)
    
    # Log statistics
    logger.info("\nPrediction Error Statistics:")
    for stat_name, value in stats.items():
        logger.info(f"{stat_name}: {value:.2f} lakhs")
    
    return stats

def main():
    # Setup logger
    logger = setup_logger()
    
    start_time = time.time()
    logger.info("Starting Real Estate Analysis Pipeline")
    
    try:
        logger.info("Initializing components...")
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer()
        model_evaluator = ModelEvaluator()
        visualizer = DataVisualizer()
        
        # Check if data file exists
        data_path = "data/raw/Pune_Real_Estate_Data_1.xlsx"
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at: {data_path}")
            return
        
        # Load and preprocess data
        logger.info("Starting data preprocessing...")
        raw_data = preprocessor.preprocess_data(data_path)
        logger.info(f"Shape after preprocessing: {raw_data.shape}")
        
        if raw_data.shape[0] < 2:
            logger.error("Not enough samples after preprocessing")
            return
            
        # Engineer features
        logger.info("Engineering features...")
        processed_data = feature_engineer.engineer_features(raw_data)
        logger.info(f"Shape after feature engineering: {processed_data.shape}")
        
        if processed_data.shape[0] < 2:
            logger.error("Not enough samples after feature engineering")
            return
        
        # Remove any NaN values
        logger.info("Cleaning data...")
        initial_shape = processed_data.shape
        processed_data = processed_data.dropna()
        logger.info(f"Shape before NaN removal: {initial_shape}")
        logger.info(f"Shape after NaN removal: {processed_data.shape}")
        
        if processed_data.shape[0] < 2:
            logger.error("Not enough samples after NaN removal")
            return
        
        # Display final columns before modeling
        logger.info("\nFinal columns before splitting:")
        logger.info("\n".join(f"- {col}" for col in processed_data.columns))
        logger.info(f"\nTotal number of features: {len(processed_data.columns)-1}")  # -1 for target variable
        
        # Prepare data for modeling
        X = processed_data.drop('Price in lakhs', axis=1)
        y = processed_data['Price in lakhs']
        
        # Log sample of feature values
        logger.info("\nSample of first few features (first 5 rows):")
        logger.info("\n" + str(X.head().to_string()))
        
        # Convert to numeric and ensure no NaN or infinite values
        y = pd.to_numeric(y, errors='coerce')
        mask = ~np.isnan(y) & ~np.isinf(y)
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Final X shape: {X.shape}")
        logger.info(f"Final y shape: {y.shape}")
        
        if X.shape[0] < 2:
            logger.error("Not enough samples for model training")
            return
        
        # Split data
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = model_trainer.split_data(X, y)
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        # Train models
        logger.info("Training models...")
        model_trainer.train_models(X_train, y_train)
        
        # Evaluate models
        logger.info("Evaluating models...")
        results = model_evaluator.evaluate_all_models(
            model_trainer.models,
            X_train, X_test,
            y_train, y_test
        )
        
        # Print and save model results
        best_model_name = print_model_results(results, logger)
        best_model = model_trainer.models[best_model_name]
        
        # Calculate and save prediction intervals
        logger.info("\nCalculating prediction intervals...")
        price_ranges = model_evaluator.calculate_prediction_intervals(
            best_model, X_test, y_test
        )
        
        # Save comprehensive results
        save_results_summary(results, price_ranges, best_model_name, logger)
        prediction_stats = save_detailed_predictions(price_ranges, logger)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualizer.plot_price_distribution(raw_data)
        visualizer.plot_area_distribution(raw_data)
        visualizer.plot_price_by_property_type(raw_data)
        visualizer.plot_model_results(results)
        visualizer.plot_amenities_analysis(raw_data)
        
        # Log execution time and completion
        execution_time = time.time() - start_time
        logger.info(f"\nPipeline completed successfully in {execution_time:.2f} seconds")
        logger.info("\nResults have been saved in:")
        logger.info("- reports/model_comparison.csv (Model performance comparison)")
        logger.info("- reports/detailed_predictions.csv (Detailed predictions)")
        logger.info("- reports/analysis_summary.txt (Comprehensive analysis)")
        logger.info("- reports/figures/ (Visualizations)")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Full error traceback:", exc_info=True)

if __name__ == "__main__":
    main()