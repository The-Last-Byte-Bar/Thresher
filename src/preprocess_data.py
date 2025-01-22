import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
from typing import Tuple, Optional, List
from process_data import DEXDataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLDataPreprocessor:
    """Preprocessor specifically designed for RL training data."""
    
    def __init__(self, validation_threshold: float = 1e4):
        self.validation_threshold = validation_threshold
        self.columns_to_drop = ['timestamp', 'datetime', 'tx_id', 'height']
        self.categorical_columns = ['is_buy']
        
    def _convert_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to numeric."""
        df_processed = df.copy()
        
        # Convert boolean/categorical columns to numeric
        if 'is_buy' in df_processed.columns:
            df_processed['is_buy'] = df_processed['is_buy'].astype(int)
            logger.info("Converted 'is_buy' to numeric")
            
        return df_processed
        
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate or adjust technical indicators after cleaning."""
        df_processed = df.copy()
        
        # List of columns that should be positive
        positive_columns = ['price', 'volume_erg', 'token_liquidity', 'erg_liquidity']
        
        # Ensure positive values where needed
        for col in positive_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].abs()
                
        # Calculate percentage changes for relevant columns
        for col in ['price', 'volume_erg']:
            if col in df_processed.columns:
                df_processed[f'{col}_pct_change'] = df_processed[col].pct_change()
                
        return df_processed
        
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to a reasonable range."""
        df_processed = df.copy()
        
        # Get numeric columns excluding target variables
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        # Scale features to reasonable ranges
        for col in numeric_cols:
            max_val = df_processed[col].abs().max()
            if max_val > self.validation_threshold:
                scale_factor = self.validation_threshold / max_val
                df_processed[col] = df_processed[col] * scale_factor
                logger.info(f"Scaled {col} by factor {scale_factor:.2e}")
                
        return df_processed
        
    def _remove_outliers(self, df: pd.DataFrame, n_std: float = 3) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        df_processed = df.copy()
        
        # Get numeric columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        # Remove outliers for each numeric column
        for col in numeric_cols:
            z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
            df_processed = df_processed[z_scores < n_std]
            
        removed_rows = len(df) - len(df_processed)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} outlier rows")
            
        return df_processed
        
    def prepare_rl_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data specifically for RL training."""
        try:
            logger.info(f"Initial data shape: {df.shape}")
            
            # Step 1: Drop unnecessary columns
            df_processed = df.drop(columns=[col for col in self.columns_to_drop if col in df.columns])
            logger.info("Dropped timestamp and identifier columns")
            
            # Step 2: Drop initial NaN values
            initial_rows = len(df_processed)
            df_processed = df_processed.dropna()
            rows_dropped = initial_rows - len(df_processed)
            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} rows with NaN values")
            
            # Step 3: Convert categorical variables
            df_processed = self._convert_categorical_columns(df_processed)
            
            # Step 4: Calculate technical indicators
            df_processed = self._calculate_technical_indicators(df_processed)
            
            # Step 5: Remove outliers
            df_processed = self._remove_outliers(df_processed)
            
            # Step 6: Normalize features
            df_processed = self._normalize_features(df_processed)
            
            # Final NaN check and cleanup
            df_processed = df_processed.dropna()
            
            # Get feature names for the RL environment
            feature_names = df_processed.columns.tolist()
            
            logger.info(f"Final data shape: {df_processed.shape}")
            logger.info(f"Features prepared for RL: {feature_names}")
            
            return df_processed, feature_names
            
        except Exception as e:
            logger.error(f"Error preparing RL data: {str(e)}")
            raise

def main():
    # Configuration
    INPUT_FILE = "data/metrics/CYPX_swaps_20250120_105113.csv"
    OUTPUT_DIR = "processed_data"
    
    try:
        # Load data
        df = pd.read_csv(INPUT_FILE)
        
        # Initialize RL preprocessor
        preprocessor = RLDataPreprocessor()
        
        # Prepare data for RL
        processed_data, feature_names = preprocessor.prepare_rl_data(df)
        
        # Create output directory
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        # Split into train/val sets (80/20)
        train_size = int(len(processed_data) * 0.8)
        train_data = processed_data.iloc[:train_size]
        val_data = processed_data.iloc[train_size:]
        
        # Save processed data and feature names
        train_data.to_pickle(output_dir / "train_data.pkl")
        val_data.to_pickle(output_dir / "val_data.pkl")
        with open(output_dir / "feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
            
        # Print summary statistics
        logger.info("\nTraining Data Statistics:")
        logger.info(train_data.describe())
        
        logger.info("\nValidation Data Statistics:")
        logger.info(val_data.describe())
        
    except Exception as e:
        logger.error(f"Error in main preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()