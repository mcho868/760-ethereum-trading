"""
Data Processor for DRL Trading System

This module handles all data loading, preprocessing, feature engineering,
and temporal splitting for the Deep Reinforcement Learning trading system.

Author: DRL Trading Team
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
# Technical Analysis library (optional)
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("⚠️ Warning: 'ta' library not available. Some technical indicators may use fallback calculations.")

from .config_manager import ConfigManager


class DataProcessor:
    """
    Comprehensive data processor for DRL trading system.
    
    This class handles:
    - Raw data loading and preprocessing
    - Core feature calculation (Z-score, momentum, zones)
    - Technical indicators calculation (MACD, RSI, Bollinger Bands, OBV)
    - Sentiment data integration
    - Temporal data splitting
    - Data validation and quality checks
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the data processor.
        
        Args:
            config: ConfigManager instance with all configuration parameters
        """
        self.config = config
        self.processed_data = None
        self.feature_columns = None
        self.data_splits = None
        
        # Ensure output directories exist
        os.makedirs(self.config.data.output_dir, exist_ok=True)
        
    def load_raw_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and preprocess raw ETH/USDT data.
        
        Args:
            data_path: Path to data file (uses config default if None)
            
        Returns:
            Cleaned and preprocessed DataFrame
        """
        if data_path is None:
            data_path = self.config.data.data_path
            
        print(f"📁 Loading data from: {data_path}")
        
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        df = pd.read_parquet(data_path)
        
        # Handle timestamp column and reset index if needed
        timestamp_col = self.config.data.timestamp_col
        if (df.index.name == timestamp_col or 'ts' in str(df.index.name) or 
            isinstance(df.index, pd.DatetimeIndex)):
            print(f"   🔧 Found timestamp in index, converting to column")
            df = df.reset_index()
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check for timestamp column
        if timestamp_col not in df.columns:
            possible_ts_cols = ['timestamp', 'ts', 'time', 'date', 'datetime']
            for col in possible_ts_cols:
                if col in df.columns:
                    print(f"   🔧 Found timestamp column: '{col}', renaming to '{timestamp_col}'")
                    df = df.rename(columns={col: timestamp_col})
                    break
            else:
                raise ValueError(f"Could not find timestamp column. Available: {list(df.columns)}")
        
        # Basic preprocessing
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        # Ensure required columns are present
        price_col = self.config.data.price_col
        required_cols = [price_col, 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Sort by timestamp and clean price data
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df = df.dropna(subset=[price_col])
        # Safe price filtering to handle any potential None values
        df = df[df[price_col].notna() & (df[price_col] > 0)]
        
        print(f"   ✅ Loaded {len(df):,} rows of data")
        print(f"   📊 Columns: {list(df.columns)}")
        
        # Display date range if possible
        if timestamp_col in df.columns:
            if df[timestamp_col].dtype == 'int64':
                start_date = pd.to_datetime(df[timestamp_col].min(), unit='s')
                end_date = pd.to_datetime(df[timestamp_col].max(), unit='s')
                print(f"   📅 Date range: {start_date} to {end_date}")
            elif pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                print(f"   📅 Date range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
        
        return df
    
    def calculate_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate core features (4 dimensions calculated here, 2 added by environment).
        
        Core Features:
        - Z-score: Standardized mean-reversion signal
        - Normalized Zone: Discrete trading signal
        - Price Momentum: Last-minute price return
        - Z-score Momentum: One-minute change in Z-score
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with core features added
        """
        print("   🔬 Calculating core features (4D + 2D by environment)...")
        
        price_col = self.config.data.price_col
        state_config = self.config.state_space
        
        # Calculate pseudo-spread and Z-score
        df['ma_baseline'] = df[price_col].rolling(
            window=state_config.ma_period, min_periods=1
        ).mean()
        
        df['spread'] = df[price_col] - df['ma_baseline']
        
        # Z-score calculation
        spread_mean = df['spread'].rolling(
            window=state_config.z_score_window, min_periods=1
        ).mean()
        spread_std = df['spread'].rolling(
            window=state_config.z_score_window, min_periods=1
        ).std()
        df['z_score'] = (df['spread'] - spread_mean) / (spread_std + 1e-8)
        df['z_score'] = df['z_score'].fillna(0)
        
        # Normalized Zone calculation
        def calculate_trading_zone(z_score):
            # Handle None/NaN values to prevent comparison errors
            if z_score is None or pd.isna(z_score):
                return 0.0  # Return neutral for None/NaN values
                
            open_threshold = state_config.open_threshold
            close_threshold = state_config.close_threshold
            
            if z_score > open_threshold:
                return 1.0  # Strong sell signal
            elif z_score > close_threshold:
                return 0.5  # Weak sell signal
            elif z_score >= -close_threshold:
                return 0.0  # Neutral (close positions)
            elif z_score >= -open_threshold:
                return -0.5  # Weak buy signal
            else:
                return -1.0  # Strong buy signal
        
        df['zone_norm'] = df['z_score'].apply(calculate_trading_zone)
        
        # Price Momentum
        df['price_momentum'] = df[price_col].pct_change(1).fillna(0)
        df['price_momentum'] = df['price_momentum'].clip(-0.1, 0.1)
        
        # Z-score Momentum
        df['z_score_momentum'] = df['z_score'].diff(1).fillna(0)
        df['z_score_momentum'] = df['z_score_momentum'].clip(-2.0, 2.0)
        
        # Clean up intermediate columns
        df = df.drop(['ma_baseline', 'spread'], axis=1)
        
        print(f"      ✅ Z-score range: [{df['z_score'].min():.2f}, {df['z_score'].max():.2f}]")
        print(f"      ✅ Zone distribution: {df['zone_norm'].value_counts().to_dict()}")
        print(f"      ✅ Price momentum range: [{df['price_momentum'].min():.4f}, {df['price_momentum'].max():.4f}]")
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators (8 dimensions).
        
        Technical Indicators:
        - MACD: Line, Signal, Histogram (3D)
        - RSI: Relative Strength Index (1D)
        - Bollinger Bands: Mid, Upper, Lower (3D)
        - OBV: On-Balance Volume (1D)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        print("   📈 Calculating technical indicators (8D)...")
        
        price_col = self.config.data.price_col
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"      ⚠️ Missing columns: {missing_cols}")
            print("      📊 Available columns:", list(df.columns))
            
            # Fallback calculations using available data
            self._calculate_indicators_fallback(df, price_col)
        else:
            # Calculate all indicators using ta library
            print("      📊 Full OHLCV data available - calculating comprehensive indicators")
            self._calculate_indicators_full(df)
        
        # Normalize all technical indicators
        self._normalize_technical_indicators(df)
        
        print(f"      ✅ Technical indicators calculated and normalized")
        return df
    
    def _calculate_indicators_fallback(self, df: pd.DataFrame, price_col: str) -> None:
        """Calculate indicators using fallback methods when OHLCV data is incomplete."""
        # MACD using existing columns or calculated from close prices
        if all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_diff']):
            df['macd'] = df['MACD']
            df['macd_signal'] = df['MACD_signal']
            df['macd_histogram'] = df['MACD_diff']
            print("      ✅ Using existing MACD indicators")
        else:
            ema12 = df[price_col].ewm(span=12).mean()
            ema26 = df[price_col].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            print("      ✅ Calculated MACD from close prices")
        
        # RSI calculation
        if 'RSI' in df.columns:
            df['rsi'] = df['RSI'] / 100.0
            print("      ✅ Using existing RSI indicator")
        else:
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df['rsi'] = (1 - (1 / (1 + rs)))
            print("      ✅ Calculated RSI from close prices")
        
        # Bollinger Bands
        if all(col in df.columns for col in ['BB_mid', 'BB_high', 'BB_low']):
            df['bb_middle'] = df['BB_mid']
            df['bb_upper'] = df['BB_high']
            df['bb_lower'] = df['BB_low']
            print("      ✅ Using existing Bollinger Bands")
        else:
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df[price_col].rolling(window=bb_period).mean()
            bb_std_val = df[price_col].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            print("      ✅ Calculated Bollinger Bands from close prices")
        
        # OBV calculation
        if 'volume' in df.columns:
            price_change = df[price_col].diff()
            volume_direction = np.where(
                price_change > 0, df['volume'],
                np.where(price_change < 0, -df['volume'], 0)
            )
            df['obv'] = volume_direction.cumsum()
            print("      ✅ Calculated OBV from volume data")
        else:
            df['obv'] = 0
            print("      ⚠️ No volume data - using dummy OBV")
    
    def _calculate_indicators_full(self, df: pd.DataFrame) -> None:
        """Calculate indicators using full OHLCV data with ta library."""
        if not TA_AVAILABLE:
            print(f"      ⚠️ TA library not available - using fallback calculations")
            self._calculate_indicators_fallback(df, 'close')
            return
            
        try:
            # MACD (3 components)
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi() / 100.0
            
            # Bollinger Bands (3 components)
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            
            # OBV
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                df['close'], df['volume']
            ).on_balance_volume()
            
        except Exception as e:
            print(f"      ⚠️ Error with ta library: {e}")
            # Fallback to manual calculations
            self._calculate_indicators_fallback(df, 'close')
    
    def _normalize_technical_indicators(self, df: pd.DataFrame) -> None:
        """Normalize technical indicators to [-1, 1] range."""
        tech_indicators = ['macd', 'macd_signal', 'macd_histogram', 'rsi',
                          'bb_middle', 'bb_upper', 'bb_lower', 'obv']
        
        for indicator in tech_indicators:
            if indicator in df.columns:
                # Robust normalization (clip outliers, then normalize)
                q99 = df[indicator].quantile(0.99)
                q01 = df[indicator].quantile(0.01)
                df[indicator] = df[indicator].clip(q01, q99)
                
                # Min-max normalization to [-1, 1] range
                min_val = df[indicator].min()
                max_val = df[indicator].max()
                if max_val > min_val:
                    df[indicator] = 2 * (df[indicator] - min_val) / (max_val - min_val) - 1
                else:
                    df[indicator] = 0
                
                df[indicator] = df[indicator].fillna(0)
    
    def integrate_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate minute-level sentiment data into the feature set.
        
        Args:
            df: DataFrame with price and technical features
            
        Returns:
            DataFrame with sentiment_score feature added
        """
        print("   🔄 Integrating sentiment data (1D)...")
        
        if not self.config.sentiment.enabled:
            df["sentiment_score"] = 0.0
            print("      ⚠️ Sentiment disabled – using neutral value 0.0")
            return df
        
        data_path = self.config.sentiment.data_path
        if not data_path or not os.path.exists(data_path):
            print("      ⚠️ Sentiment file missing – using 0.0")
            df["sentiment_score"] = 0.0
            return df
        
        try:
            sentiment_df = pd.read_csv(data_path)
        except Exception as exc:
            print(f"      ⚠️ Failed to load sentiment data: {exc} - defaulting to 0.0")
            df['sentiment_score'] = 0.0
            return df
        
        timestamp_col = self.config.data.timestamp_col
        if timestamp_col not in sentiment_df.columns:
            raise ValueError(f"Sentiment file must include '{timestamp_col}' column")
        
        # Process sentiment data
        sentiment_df[timestamp_col] = pd.to_datetime(sentiment_df[timestamp_col])
        sentiment_cols = [c for c in sentiment_df.columns if c != timestamp_col]
        
        if not sentiment_cols:
            print("      ⚠️ No sentiment columns found - defaulting to 0.0")
            df['sentiment_score'] = 0.0
            return df
        
        # Aggregate to minute level
        sentiment_df = (
            sentiment_df.groupby(timestamp_col)[sentiment_cols]
            .mean()
            .reset_index()
        )
        sentiment_df["sentiment_score"] = sentiment_df[sentiment_cols].mean(axis=1)
        
        # Handle timestamp compatibility
        df_ts_dtype = df[timestamp_col].dtype
        
        if pd.api.types.is_numeric_dtype(df_ts_dtype):
            # Convert datetime to Unix timestamp
            sentiment_df[timestamp_col] = (
                sentiment_df[timestamp_col].view("int64") // 10**9
            ).astype(df_ts_dtype)
        else:
            # Both are datetime types - ensure compatibility
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            sentiment_df[timestamp_col] = pd.to_datetime(sentiment_df[timestamp_col])
            
            # Remove timezone info
            if hasattr(df[timestamp_col].dtype, 'tz') and df[timestamp_col].dtype.tz is not None:
                df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)
            if hasattr(sentiment_df[timestamp_col].dtype, 'tz') and sentiment_df[timestamp_col].dtype.tz is not None:
                sentiment_df[timestamp_col] = sentiment_df[timestamp_col].dt.tz_localize(None)
        
        # Merge sentiment data
        df = df.merge(
            sentiment_df[[timestamp_col, "sentiment_score"]],
            on=timestamp_col,
            how="left",
        )
        coverage = df["sentiment_score"].notna().mean()
        df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
        
        print(f"      ✅ Sentiment merged – coverage: {coverage:.1%}")
        return df
    
    def create_temporal_splits(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create temporal data splits following methodology.
        
        Args:
            df: Processed DataFrame with features
            
        Returns:
            Dictionary containing train, validation, test splits and metadata
        """
        print("🔄 Creating temporal data splits (70/15/15)...")
        
        total_rows = len(df)
        protocol_config = self.config.training_protocol
        
        # Calculate split indices
        train_end_idx = int(total_rows * protocol_config.train_ratio)
        val_end_idx = int(total_rows * (protocol_config.train_ratio + protocol_config.validation_ratio))
        
        # Create splits
        train_df = df.iloc[:train_end_idx].copy().reset_index(drop=True)
        val_df = df.iloc[train_end_idx:val_end_idx].copy().reset_index(drop=True)
        test_df = df.iloc[val_end_idx:].copy().reset_index(drop=True)
        
        # Calculate date ranges for metadata
        timestamp_col = self.config.data.timestamp_col
        if timestamp_col in df.columns:
            train_start = pd.to_datetime(train_df[timestamp_col].iloc[0], unit='s')
            train_end = pd.to_datetime(train_df[timestamp_col].iloc[-1], unit='s')
            val_start = pd.to_datetime(val_df[timestamp_col].iloc[0], unit='s')
            val_end = pd.to_datetime(val_df[timestamp_col].iloc[-1], unit='s')
            test_start = pd.to_datetime(test_df[timestamp_col].iloc[0], unit='s')
            test_end = pd.to_datetime(test_df[timestamp_col].iloc[-1], unit='s')
        else:
            train_start = train_end = val_start = val_end = test_start = test_end = None
        
        # Create metadata
        split_info = {
            'total_rows': total_rows,
            'train_rows': len(train_df),
            'val_rows': len(val_df),
            'test_rows': len(test_df),
            'train_ratio': len(train_df) / total_rows,
            'val_ratio': len(val_df) / total_rows,
            'test_ratio': len(test_df) / total_rows,
            'train_start_date': train_start.isoformat() if train_start else None,
            'train_end_date': train_end.isoformat() if train_end else None,
            'val_start_date': val_start.isoformat() if val_start else None,
            'val_end_date': val_end.isoformat() if val_end else None,
            'test_start_date': test_start.isoformat() if test_start else None,
            'test_end_date': test_end.isoformat() if test_end else None,
            'methodology_compliance': True,
            'chronological_order': True
        }
        
        print(f"   ✅ Training:   {len(train_df):,} rows ({len(train_df)/total_rows:.1%})")
        print(f"   ✅ Validation: {len(val_df):,} rows ({len(val_df)/total_rows:.1%})")
        print(f"   ✅ Test:       {len(test_df):,} rows ({len(test_df)/total_rows:.1%})")
        
        if train_start:
            print(f"   📅 Training period:   {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            print(f"   📅 Validation period: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
            print(f"   📅 Test period:       {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
        
        # Store splits
        splits = {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'metadata': split_info
        }
        
        self.data_splits = splits
        return splits
    
    def save_splits_to_files(self, splits: Dict[str, Any]) -> None:
        """Save data splits to parquet files."""
        output_dir = self.config.data.output_dir
        
        # Save data splits
        train_path = os.path.join(output_dir, 'train_data_15d.parquet')
        val_path = os.path.join(output_dir, 'val_data_15d.parquet')
        test_path = os.path.join(output_dir, 'test_data_15d.parquet')
        split_info_path = os.path.join(output_dir, 'temporal_split_info.json')
        
        splits['train'].to_parquet(train_path, index=False)
        splits['validation'].to_parquet(val_path, index=False)
        splits['test'].to_parquet(test_path, index=False)
        
        with open(split_info_path, 'w') as f:
            json.dump(splits['metadata'], f, indent=2)
        
        print(f"   💾 Data splits saved to: {output_dir}")
    
    def get_feature_columns(self) -> List[str]:
        """Get standard feature column names for 15D state space."""
        return [
            # Core features (4D calculated, 2D added by environment)
            'z_score', 'zone_norm', 'price_momentum', 'z_score_momentum',
            # Technical indicators (8D)
            'macd', 'macd_signal', 'macd_histogram',  # MACD (3D)
            'rsi',  # RSI (1D)
            'bb_middle', 'bb_upper', 'bb_lower',  # Bollinger Bands (3D)
            'obv',  # OBV (1D)
            # Sentiment (1D)
            'sentiment_score'
        ]
    
    def run_full_pipeline(self, data_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Run the complete data processing pipeline.
        
        Args:
            data_path: Path to raw data file (uses config default if None)
            
        Returns:
            Tuple of (processed_data, feature_columns, data_splits)
        """
        print("🚀 Starting complete data processing pipeline...")
        
        # 1. Load raw data
        df_raw = self.load_raw_data(data_path)
        
        # 2. Feature engineering pipeline
        print("\n🔧 Feature Engineering Pipeline (15D State Space):")
        df_processed = df_raw.copy()
        
        # Core features (4D calculated, 2D added by environment)
        df_processed = self.calculate_core_features(df_processed)
        
        # Technical indicators (8D)
        df_processed = self.calculate_technical_indicators(df_processed)
        
        # Sentiment data (1D)
        df_processed = self.integrate_sentiment_data(df_processed)
        
        # Get feature columns
        feature_columns = self.get_feature_columns()
        
        # Verify all features are present
        missing_features = [col for col in feature_columns if col not in df_processed.columns]
        if missing_features:
            print(f"\n❌ Missing features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
        else:
            print(f"\n✅ All 15D features successfully calculated!")
        
        # 3. Create temporal splits
        splits = self.create_temporal_splits(df_processed)
        
        # 4. Save processed data
        processed_data_path = os.path.join(self.config.data.output_dir, 'processed_eth_data_15d.parquet')
        df_processed.to_parquet(processed_data_path, index=False)
        print(f"   💾 Processed data saved: {processed_data_path}")
        
        # 5. Save splits
        self.save_splits_to_files(splits)
        
        # 6. Display summary
        print(f"\n📊 Data Processing Summary:")
        print(f"   📈 Raw data: {len(df_raw):,} rows")
        print(f"   🔧 Processed data: {len(df_processed):,} rows")
        print(f"   📋 Features: {len(feature_columns)} dimensions")
        print(f"   🎯 State space: 15D (as per methodology)")
        print(f"   💾 Memory usage: {df_processed.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Display sample features
        print(f"\n📋 Sample of 15D Feature Data:")
        sample_features = df_processed[feature_columns].tail()
        pd.set_option('display.max_columns', None)
        print(sample_features)
        pd.reset_option('display.max_columns')
        
        print(f"\n🎯 Data ready for 15D state space training!")
        
        # Store results
        self.processed_data = df_processed
        self.feature_columns = feature_columns
        
        return df_processed, feature_columns, splits
    
    def load_existing_splits(self) -> Optional[Dict[str, Any]]:
        """
        Load existing data splits from files.
        
        Returns:
            Dictionary with splits or None if not found
        """
        output_dir = self.config.data.output_dir
        train_path = os.path.join(output_dir, 'train_data_15d.parquet')
        val_path = os.path.join(output_dir, 'val_data_15d.parquet')
        test_path = os.path.join(output_dir, 'test_data_15d.parquet')
        split_info_path = os.path.join(output_dir, 'temporal_split_info.json')
        
        if not all(os.path.exists(p) for p in [train_path, val_path, test_path, split_info_path]):
            return None
        
        try:
            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)
            test_df = pd.read_parquet(test_path)
            
            with open(split_info_path, 'r') as f:
                metadata = json.load(f)
            
            splits = {
                'train': train_df,
                'validation': val_df,
                'test': test_df,
                'metadata': metadata
            }
            
            self.data_splits = splits
            print(f"✅ Loaded existing data splits from: {output_dir}")
            return splits
            
        except Exception as e:
            print(f"⚠️ Error loading existing splits: {e}")
            return None
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        print("🔍 Validating data quality...")
        
        quality_metrics = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }
        
        # Check for extreme values in price data
        price_col = self.config.data.price_col
        if price_col in df.columns:
            quality_metrics['price_stats'] = {
                'min': float(df[price_col].min()),
                'max': float(df[price_col].max()),
                'mean': float(df[price_col].mean()),
                'std': float(df[price_col].std()),
                'zero_values': int((df[price_col] == 0).sum()),
                'negative_values': int((df[price_col] < 0).sum())
            }
        
        # Check feature completeness
        if self.feature_columns:
            available_features = [col for col in self.feature_columns if col in df.columns]
            quality_metrics['feature_completeness'] = {
                'available_features': len(available_features),
                'total_features': len(self.feature_columns),
                'completion_rate': len(available_features) / len(self.feature_columns)
            }
        
        print(f"   ✅ Data quality validation complete")
        print(f"   📊 Total rows: {quality_metrics['total_rows']:,}")
        print(f"   🔍 Missing values: {sum(quality_metrics['missing_values'].values())}")
        print(f"   🔄 Duplicate rows: {quality_metrics['duplicate_rows']}")
        
        return quality_metrics


if __name__ == "__main__":
    # Example usage
    from .config_manager import ConfigManager
    
    config = ConfigManager()
    processor = DataProcessor(config)
    
    # Run full pipeline
    processed_data, features, splits = processor.run_full_pipeline()
    
    # Validate data quality
    quality = processor.validate_data_quality(processed_data)
    print(f"\nData Quality Report: {quality}")
