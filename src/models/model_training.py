# src/models/model_training.py

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
import optuna
import logging

class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger('RealEstateAnalysis.ModelTrainer')
        
        self.logger.info("Initializing models")
        self.models = {
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'Lasso': Lasso(random_state=42),
            'Ridge': Ridge(random_state=42),
            'CatBoost': CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)
        }
        self.best_model = None

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        self.logger.info(f"Splitting data with test_size={test_size}")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            self.logger.info(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

    def train_models(self, X_train, y_train):
        """Train all models."""
        self.logger.info("Starting model training")
        
        for name, model in self.models.items():
            try:
                self.logger.info(f"Training {name}")
                model.fit(X_train, y_train)
                self.logger.info(f"{name} training completed")
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                raise

    def optimize_lightgbm(self, X_train, y_train, n_trials=50):
        """Optimize LightGBM hyperparameters using Optuna."""
        self.logger.info(f"Starting LightGBM optimization with {n_trials} trials")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
                'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 10.0),
                'subsample': trial.suggest_uniform('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.7, 0.9),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
            }
            
            try:
                model = lgb.LGBMRegressor(**params, random_state=42)
                model.fit(X_train, y_train)
                return model.score(X_train, y_train)
            except Exception as e:
                self.logger.warning(f"Trial failed with parameters {params}: {str(e)}")
                return float('-inf')

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            self.logger.info("LightGBM optimization completed")
            self.logger.info(f"Best parameters: {study.best_params}")
            
            # Train final model with best parameters
            self.best_model = lgb.LGBMRegressor(**study.best_params, random_state=42)
            self.best_model.fit(X_train, y_train)
            
            return self.best_model
            
        except Exception as e:
            self.logger.error(f"Error in LightGBM optimization: {str(e)}")
            raise

    def get_best_model(self):
        """Return the best model (optimized LightGBM)."""
        return self.best_model