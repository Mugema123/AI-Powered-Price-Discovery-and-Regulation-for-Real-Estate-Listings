# src/visualization/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
import os

class DataVisualizer:
    def __init__(self):
        self.logger = logging.getLogger('RealEstateAnalysis.DataVisualizer')
        sns.set_theme()
        
        # Create figures directory if it doesn't exist
        self.figures_dir = 'reports/figures'
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
            self.logger.info(f"Created figures directory: {self.figures_dir}")

    def save_plot(self, filename):
        """Helper method to save plots with proper error handling."""
        try:
            filepath = os.path.join(self.figures_dir, filename)
            plt.savefig(filepath)
            plt.close()
            self.logger.info(f"Saved plot to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving plot {filename}: {str(e)}")
            plt.close()  # Close figure even if save fails
            raise

    def plot_price_distribution(self, df):
        """Plot distribution of property prices."""
        self.logger.info("Creating price distribution plot")
        try:
            plt.figure(figsize=(12, 8))
            sns.histplot(df['Price in lakhs'], kde=True, bins=30, color='blue')
            plt.title('Distribution of Property Prices (in Lakhs)')
            plt.xlabel('Price in Lakhs')
            plt.ylabel('Frequency')
            self.save_plot('price_distribution.png')
        except Exception as e:
            self.logger.error(f"Error creating price distribution plot: {str(e)}")
            plt.close()
            raise

    def plot_area_distribution(self, df):
        """Plot distribution of property areas."""
        self.logger.info("Creating area distribution plot")
        try:
            plt.figure(figsize=(8, 8))
            sns.histplot(df['Property Area in Sq. Ft.'], kde=True, bins=30, color='green')
            plt.title('Distribution of Property Area (in Sq. Ft.)')
            plt.xlabel('Area in Sq. Ft.')
            plt.ylabel('Frequency')
            self.save_plot('area_distribution.png')
        except Exception as e:
            self.logger.error(f"Error creating area distribution plot: {str(e)}")
            plt.close()
            raise

    def plot_price_by_property_type(self, df):
        """Plot boxplot of prices by property type."""
        self.logger.info("Creating price by property type plot")
        try:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Propert Type', y='Price in lakhs', data=df)
            plt.xticks(rotation=45)
            plt.title('Property Prices by Property Type')
            plt.xlabel('Property Type')
            plt.ylabel('Price in Lakhs')
            plt.tight_layout()
            self.save_plot('price_by_property_type.png')
        except Exception as e:
            self.logger.error(f"Error creating price by property type plot: {str(e)}")
            plt.close()
            raise

    def plot_model_results(self, results):
        """Plot model evaluation results."""
        self.logger.info("Creating model results plots")
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot train vs test RMSE
            plt.subplot(1, 2, 1)
            names = list(results.keys())
            train_rmse = [results[name]['train_metrics'][1] for name in names]
            test_rmse = [results[name]['test_metrics'][1] for name in names]
            
            x = np.arange(len(names))
            width = 0.35
            
            plt.bar(x - width/2, train_rmse, width, label='Train RMSE', color='blue', alpha=0.7)
            plt.bar(x + width/2, test_rmse, width, label='Test RMSE', color='red', alpha=0.7)
            plt.xticks(x, names, rotation=45)
            plt.ylabel('RMSE')
            plt.title('Train vs Test RMSE')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot test metrics
            plt.subplot(1, 2, 2)
            test_mae = [results[name]['test_metrics'][0] for name in names]
            test_r2 = [results[name]['test_metrics'][2] for name in names]
            
            width = 0.25
            plt.bar(x - width, test_mae, width, label='MAE', color='green', alpha=0.7)
            plt.bar(x, test_rmse, width, label='RMSE', color='red', alpha=0.7)
            plt.bar(x + width, test_r2, width, label='R2', color='blue', alpha=0.7)
            plt.xticks(x, names, rotation=45)
            plt.title('Test Set Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.save_plot('model_results.png')
            
        except Exception as e:
            self.logger.error(f"Error creating model results plots: {str(e)}")
            plt.close()
            raise

    def plot_amenities_analysis(self, df):
        """Plot price distribution by amenities."""
        self.logger.info("Creating amenities analysis plot")
        try:
            plt.figure(figsize=(12, 8))
            amenities = ['ClubHouse', 'School / University in Township ',
                        'Hospital in TownShip', 'Mall in TownShip',
                        'Park / Jogging track', 'Swimming Pool', 'Gym']
            
            df_melted = pd.melt(df, 
                              id_vars=['Price in lakhs'],
                              value_vars=amenities,
                              var_name='Amenity',
                              value_name='Present')
            
            sns.boxplot(x='Amenity', y='Price in lakhs', hue='Present',
                       data=df_melted, palette='Set2')
            
            plt.xticks(rotation=45, ha='right')
            plt.title('Distribution of Price by Amenities')
            plt.ylabel('Price in Lakhs')
            plt.tight_layout()
            self.save_plot('amenities_analysis.png')
            
        except Exception as e:
            self.logger.error(f"Error creating amenities analysis plot: {str(e)}")
            plt.close()
            raise