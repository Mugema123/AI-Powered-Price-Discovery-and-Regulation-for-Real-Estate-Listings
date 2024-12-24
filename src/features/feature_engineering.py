# src/features/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger('RealEstateAnalysis.FeatureEngineer')
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.tfidf = TfidfVectorizer(ngram_range=(2,2), max_features=10, 
                                   lowercase=True, stop_words='english')

    def create_price_subarea_feature(self, df):
        """Create average price by sub-area feature."""
        self.logger.info("Creating price by sub-area feature")
        try:
            df['Price_sub-area'] = df.groupby('Sub-Area')['Price in lakhs'].transform('mean')
            self.logger.debug("Successfully created price_sub-area feature")
            return df
        except Exception as e:
            self.logger.error(f"Error creating price_sub-area feature: {str(e)}")
            raise

    def extract_bhk_number(self, property_type):
        """Extract number of BHK from property type."""
        try:
            prop_type = str(property_type).lower()
            if prop_type == 'shop':
                return 0
            numbers = re.findall(r'\d+\.?\d*', prop_type.split('bhk')[0])
            if numbers:
                return float(numbers[0])
            return None
        except Exception as e:
            self.logger.warning(f"Error extracting BHK number from '{property_type}': {str(e)}")
            return None

    def process_description(self, df):
        """Process description text using TF-IDF."""
        self.logger.info("Processing description text with TF-IDF")
        try:
            X = self.tfidf.fit_transform(df['Description'])
            df_tfidf = pd.DataFrame(X.toarray(), 
                                  columns=self.tfidf.get_feature_names_out())
            self.logger.info(f"Created {len(self.tfidf.get_feature_names_out())} TF-IDF features")
            return pd.concat([df.reset_index(drop=True), 
                            df_tfidf.reset_index(drop=True)], axis=1)
        except Exception as e:
            self.logger.error(f"Error in TF-IDF processing: {str(e)}")
            raise

    def handle_categorical_features(self, df):
        """Handle categorical features using appropriate encoding."""
        self.logger.info("Handling categorical features")
        encoding_columns = ['Sub-Area', 'Propert Type', 'Company Name', 
                          'TownShip Name/ Society Name']
        
        try:
            for column in encoding_columns:
                unique_values = df[column].nunique()
                self.logger.debug(f"Processing column {column} with {unique_values} unique values")
                
                if unique_values < 10:
                    # Apply one-hot encoding
                    self.logger.debug(f"Applying one-hot encoding to {column}")
                    encoded_data = self.one_hot_encoder.fit_transform(df[[column]])
                    encoded_df = pd.DataFrame(
                        encoded_data, 
                        columns=self.one_hot_encoder.get_feature_names_out([column])
                    )
                    df = pd.concat([df.reset_index(drop=True), 
                                  encoded_df.reset_index(drop=True)], axis=1)
                    df.drop(columns=[column], inplace=True)
                else:
                    # Apply label encoding
                    self.logger.debug(f"Applying label encoding to {column}")
                    df[column] = self.label_encoder.fit_transform(df[column])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in categorical feature handling: {str(e)}")
            raise

    def engineer_features(self, df):
        """Main feature engineering pipeline."""
        self.logger.info("Starting feature engineering pipeline")
        try:
            # Create price by sub-area feature
            df = self.create_price_subarea_feature(df)
            
            # Extract BHK number
            self.logger.info("Extracting BHK numbers")
            df['BHK_number'] = df['Propert Type'].apply(self.extract_bhk_number)
            
            initial_shape = df.shape
            self.logger.info(f"Initial shape: {initial_shape}")
            
            # Process description
            df = self.process_description(df)
            self.logger.info(f"Shape after description processing: {df.shape}")
            
            # Handle categorical features
            df = self.handle_categorical_features(df)
            self.logger.info(f"Shape after categorical handling: {df.shape}")
            
            # Drop description column
            df.drop(columns=['Description'], inplace=True)
            
            # Check for any completely empty columns
            empty_cols = df.columns[df.isna().all()].tolist()
            if empty_cols:
                self.logger.warning(f"Found completely empty columns: {empty_cols}")
                df = df.drop(columns=empty_cols)
                
            # Fill any remaining NaN values with appropriate defaults
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.logger.info(f"Filled NaN values in {col} with median: {median_val}")
            
            self.logger.info("Feature engineering completed successfully")
            self.logger.info(f"Final engineered data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            self.logger.error("Full error traceback:", exc_info=True)
            raise