# src/models/model_evaluation.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import pandas as pd
import logging

class ModelEvaluator:
    def __init__(self):
        self.logger = logging.getLogger('RealEstateAnalysis.ModelEvaluator')
        self.results = {}

    def evaluate_model(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        try:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            self.logger.debug(f"Calculated metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
            return mae, rmse, r2
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def evaluate_all_models(self, models, X_train, X_test, y_train, y_test):
        """Evaluate all models and store results."""
        self.logger.info("Starting evaluation of all models")
        
        for name, model in models.items():
            self.logger.info(f"Evaluating {name}")
            try:
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_metrics = self.evaluate_model(y_train, train_pred)
                test_metrics = self.evaluate_model(y_test, test_pred)
                
                self.results[name] = {
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics
                }
                
                self.logger.info(f"{name} evaluation completed")
                self.logger.info(f"Train metrics - MAE: {train_metrics[0]:.2f}, RMSE: {train_metrics[1]:.2f}, R2: {train_metrics[2]:.4f}")
                self.logger.info(f"Test metrics - MAE: {test_metrics[0]:.2f}, RMSE: {test_metrics[1]:.2f}, R2: {test_metrics[2]:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {str(e)}")
                raise
        
        self.logger.info("Model evaluation completed for all models")
        return self.results

    def calculate_prediction_intervals(self, model, X_test, y_test, confidence=0.95):
        """Calculate prediction intervals for the model."""
        self.logger.info(f"Calculating prediction intervals with {confidence*100}% confidence")
        
        try:
            test_pred = model.predict(X_test)
            residuals = y_test - test_pred
            std_residuals = np.std(residuals)
            
            z_score = stats.norm.ppf((1 + confidence) / 2)
            margin_of_error = z_score * std_residuals
            
            price_ranges = pd.DataFrame({
                'Actual': y_test,
                'Predicted': test_pred,
                'Lower_Bound': test_pred - margin_of_error,
                'Upper_Bound': test_pred + margin_of_error
            })
            
            self.logger.info(f"Average margin of error: {margin_of_error:.2f} lakhs")
            self.logger.debug(f"Prediction intervals calculated with shape: {price_ranges.shape}")
            
            return price_ranges
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction intervals: {str(e)}")
            raise

    def get_best_model_name(self):
        """Determine the best model based on test metrics."""
        self.logger.info("Determining best model")
        try:
            model_scores = {}
            for name, metrics in self.results.items():
                train_rmse = metrics['train_metrics'][1]
                test_rmse = metrics['test_metrics'][1]
                rmse_difference = abs(train_rmse - test_rmse)
                test_performance = test_rmse
                
                # Combined score: balance between test performance and stability
                combined_score = 0.7 * test_performance + 0.3 * rmse_difference
                model_scores[name] = combined_score
                
                self.logger.debug(f"{name} - Test RMSE: {test_rmse:.2f}, RMSE Difference: {rmse_difference:.2f}, Combined Score: {combined_score:.2f}")
            
            best_model = min(model_scores.items(), key=lambda x: x[1])[0]
            self.logger.info(f"Best model determined: {best_model}")
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error determining best model: {str(e)}")
            raise