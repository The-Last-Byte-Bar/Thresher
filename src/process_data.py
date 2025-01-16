import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import RobustScaler
import logging

class DEXDataPreprocessor:
    """Enhanced data preprocessor for DEX trading data"""
    
    def __init__(self, validation_threshold: float = 1e6):
        self.scaler = RobustScaler()
        self.validation_threshold = validation_threshold
        self.logger = logging.getLogger("process_data")
        
        # Columns to exclude from normalization
        self.exclude_columns = [
            'timestamp', 'datetime', 'tx_id', 'height',
            'inclusionHeight', 'id', 'token_id', 'input_token',
            'output_token', 'input_token_name', 'output_token_name'
        ]
        
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data for common issues"""
        issues = []
        
        # Check for NaN values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_cols = df[numeric_cols].isna().sum()[df[numeric_cols].isna().sum() > 0]
        if not nan_cols.empty:
            issues.extend([f"NaN values found in column {col}" for col in nan_cols.index])
            
        # Check for infinite values in numeric columns
        inf_cols = df[numeric_cols].isin([np.inf, -np.inf]).sum()[df[numeric_cols].isin([np.inf, -np.inf]).sum() > 0]
        if not inf_cols.empty:
            issues.extend([f"Infinite values found in column {col}" for col in inf_cols.index])
            
        # Check for extreme values in numeric columns
        for col in numeric_cols:
            if df[col].abs().max() > self.validation_threshold:
                issues.append(f"Extreme values found in column {col}")
                
        return len(issues) == 0, issues
        
    def process_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Process outliers using IQR method with custom multiplier"""
        df_processed = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col not in self.exclude_columns]
            
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Clip values outside bounds
            df_processed[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            
        return df_processed
        
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features using RobustScaler"""
        df_normalized = df.copy()
        
        # Select only numeric columns for normalization, excluding specified columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in self.exclude_columns]
        
        if fit:
            normalized_data = self.scaler.fit_transform(df[cols_to_normalize])
        else:
            normalized_data = self.scaler.transform(df[cols_to_normalize])
            
        df_normalized[cols_to_normalize] = normalized_data
        return df_normalized
        
    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Complete data preparation pipeline"""
        # Initial validation
        is_valid, issues = self.validate_data(df)
        if not is_valid:
            self.logger.warning("Data validation issues found:")
            for issue in issues:
                self.logger.warning(f"- {issue}")
                
        # Convert timestamp to datetime if present
        df_processed = df.copy()
        if 'timestamp' in df_processed.columns:
            try:
                # If timestamp is in milliseconds (13 digits), convert to seconds
                if df_processed['timestamp'].astype(str).str.len().max() >= 13:
                    df_processed['timestamp'] = df_processed['timestamp'].astype(float) / 1000
                df_processed['datetime'] = pd.to_datetime(df_processed['timestamp'], unit='s')
            except Exception as e:
                self.logger.warning(f"Error converting timestamp to datetime: {e}")
        
        # Handle missing values in numeric columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].ffill().bfill()
        
        # Replace infinite values with large finite values
        df_processed = df_processed.replace([np.inf, -np.inf], [self.validation_threshold, -self.validation_threshold])
        
        # Process outliers
        df_processed = self.process_outliers(df_processed)
        
        # Normalize features
        df_processed = self.normalize_features(df_processed, fit=is_training)
        
        # Final validation
        is_valid, issues = self.validate_data(df_processed)
        if not is_valid:
            self.logger.error("Processed data still has validation issues:")
            for issue in issues:
                self.logger.error(f"- {issue}")
            raise ValueError("Data preprocessing failed to resolve all issues")
            
        return df_processed
        
    def prepare_train_val_data(self, 
                              df: pd.DataFrame, 
                              train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare and split data into training and validation sets"""
        # Calculate split index
        split_idx = int(len(df) * train_ratio)
        
        # Split data
        train_data = df.iloc[:split_idx].copy()
        val_data = df.iloc[split_idx:].copy()
        
        # Prepare training data
        train_processed = self.prepare_data(train_data, is_training=True)
        
        # Prepare validation data using training data statistics
        val_processed = self.prepare_data(val_data, is_training=False)
        
        return train_processed, val_processed

def test_preprocessor():
    """Test function for the preprocessor"""
    # Create sample data with various issues
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-01-01', periods=n_samples, freq='1min').astype(np.int64) // 10**9,
        'price': np.random.normal(100, 10, n_samples),
        'volume': np.random.exponential(1000, n_samples),
        'returns': np.random.normal(0, 0.02, n_samples),
    })
    
    # Add some problematic values
    data.loc[0, 'price'] = np.nan
    data.loc[1, 'volume'] = np.inf
    data.loc[2, 'returns'] = -np.inf
    data.loc[3:5, 'price'] = 1e10  # Extreme values
    
    # Initialize and test preprocessor
    preprocessor = DEXDataPreprocessor()
    
    # Test validation
    is_valid, issues = preprocessor.validate_data(data)
    print("Initial validation issues:", issues)
    
    # Test complete pipeline
    processed_data = preprocessor.prepare_data(data)
    
    # Verify results
    print("\nProcessed data statistics:")
    print(processed_data.describe())
    
    # Test train/val split
    train_data, val_data = preprocessor.prepare_train_val_data(data)
    print("\nTraining data shape:", train_data.shape)
    print("Validation data shape:", val_data.shape)
    
    return processed_data, train_data, val_data

if __name__ == "__main__":
    test_preprocessor()