# src/data/data_preprocessing.py

import pandas as pd
import numpy as np
import re
import logging

class DataPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger('RealEstateAnalysis.DataPreprocessor')
        self.columns_to_remove = ['Sr. No.', 'Price in Millions', 'Total TownShip Area in Acres', 'Location']
        self.binary_columns = ['ClubHouse', 'School / University in Township ', 
                             'Hospital in TownShip', 'Mall in TownShip', 
                             'Park / Jogging track', 'Swimming Pool', 'Gym']

    def load_data(self, file_path):
        """Load data from Excel file."""
        self.logger.info(f"Loading data from {file_path}")
        try:
            df = pd.read_excel(file_path)
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def remove_unnecessary_features(self, df):
        """Remove unnecessary columns from DataFrame."""
        self.logger.info("Removing unnecessary features")
        initial_cols = df.columns.tolist()
        df = df.drop(columns=[col for col in self.columns_to_remove if col in df.columns], errors='ignore')
        removed_cols = set(initial_cols) - set(df.columns.tolist())
        self.logger.info(f"Removed columns: {removed_cols}")
        return df

    def clean_property_area(self, area):
        """Clean property area values."""
        try:
            if isinstance(area, (int, float)):
                return float(area)
            area = str(area).strip()
            if '+' in area:
                return float(area.replace(' +', ''))
            if ',' in area:
                numbers = [float(x.strip()) for x in area.split(',')]
                return sum(numbers) / len(numbers)
            if 'to' in area:
                lower, upper = map(float, area.split('to'))
                return (lower + upper) / 2
            return float(area)
        except Exception as e:
            self.logger.warning(f"Error cleaning property area value '{area}': {str(e)}")
            return None

    def clean_property_type(self, value):
        """Clean property type values."""
        try:
            value = str(value).strip().lower()
            value = re.sub(r'\s*bhk\s*', ' BHK', value)
            value = re.sub(r'\s+', ' ', value)
            if '+' in value:
                numbers = map(int, re.findall(r'\d+', value))
                value = f"{sum(numbers)} BHK"
            return value.title()
        except Exception as e:
            self.logger.warning(f"Error cleaning property type value '{value}': {str(e)}")
            return value

    def clean_text_column(self, value):
        """Clean text column values."""
        try:
            value = str(value).strip().title()
            value = re.sub(r'\s+', ' ', value)
            return value
        except Exception as e:
            self.logger.warning(f"Error cleaning text value '{value}': {str(e)}")
            return value

    def convert_binary_features(self, df):
        """Convert binary features to 0/1."""
        self.logger.info("Converting binary features")
        for column in self.binary_columns:
            if column in df.columns:
                df[column] = df[column].str.strip().str.lower().map({'yes': 1, 'no': 0})
                self.logger.debug(f"Converted binary column: {column}")
        return df

    def handle_outliers(self, df, columns_to_treat, lower_percentile=0, upper_percentile=95):
        """Handle outliers using percentile-based clipping."""
        self.logger.info("Handling outliers")
        df_treated = df.copy()
        
        for column in columns_to_treat:
            self.logger.info(f"Processing outliers for column: {column}")
            
            # Ensure column is numeric
            df_treated[column] = pd.to_numeric(df_treated[column], errors='coerce')
            
            # Calculate bounds only on non-null values
            non_null_values = df_treated[column].dropna()
            if len(non_null_values) > 0:
                lower_bound = non_null_values.quantile(lower_percentile/100)
                upper_bound = non_null_values.quantile(upper_percentile/100)
                self.logger.info(f"{column} bounds - Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f}")
                
                # Count outliers before clipping
                outliers_count = ((df_treated[column] < lower_bound) | (df_treated[column] > upper_bound)).sum()
                self.logger.info(f"Number of outliers in {column}: {outliers_count}")
                
                df_treated[column] = df_treated[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df_treated

    def preprocess_data(self, file_path):
        """Main preprocessing pipeline."""
        self.logger.info("Starting data preprocessing pipeline")
        
        try:
            # Load data
            df = self.load_data(file_path)
            
            # Remove unnecessary features
            df = self.remove_unnecessary_features(df)
            
            # Convert Price to numeric and handle missing values
            self.logger.info("Processing Price column")
            # Handle price column carefully
            df['Price in lakhs'] = pd.to_numeric(df['Price in lakhs'], errors='coerce')
            self.logger.info(f"NaN values in Price column: {df['Price in lakhs'].isna().sum()}")
            
            # Only fill NaN if we have some valid prices
            if df['Price in lakhs'].notna().sum() > 0:
                price_median = df['Price in lakhs'].median()
                df['Price in lakhs'] = df['Price in lakhs'].fillna(price_median)
                self.logger.info(f"Filled NaN prices with median: {price_median}")
            else:
                self.logger.error("No valid prices found in the dataset")
            
            # Clean property area
            self.logger.info("Cleaning Property Area column")
            df['Property Area in Sq. Ft.'] = df['Property Area in Sq. Ft.'].apply(self.clean_property_area)
            df['Property Area in Sq. Ft.'] = pd.to_numeric(df['Property Area in Sq. Ft.'], errors='coerce')
            
            # Clean property type
            self.logger.info("Cleaning Property Type column")
            df['Propert Type'] = df['Propert Type'].apply(self.clean_property_type)
            
            # Clean text columns
            self.logger.info("Cleaning text columns")
            text_columns = ['Company Name', 'Sub-Area', 'TownShip Name/ Society Name', 'Description']
            for col in text_columns:
                self.logger.debug(f"Cleaning column: {col}")
                df[col] = df[col].apply(self.clean_text_column)
            
            # Convert binary features
            df = self.convert_binary_features(df)
            
            # Handle outliers for numeric columns
            columns_to_treat = ['Property Area in Sq. Ft.', 'Price in lakhs']
            df = self.handle_outliers(df, columns_to_treat)
            
            self.logger.info("Data preprocessing completed successfully")
            self.logger.info(f"Final preprocessed data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            self.logger.error("Full error traceback:", exc_info=True)
            raise