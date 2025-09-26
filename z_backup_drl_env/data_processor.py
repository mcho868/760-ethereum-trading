import asyncio
import json
import websockets
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging
import talib

class DataProcessor:
    """
    Real-time data processor for DRL trading
    Handles Binance WebSocket data and calculates technical indicators
    """
    
    def __init__(self, symbol: str = "ethusdt", buffer_size: int = 500):
        self.symbol = symbol.lower()
        self.buffer_size = buffer_size
        self.websocket_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_1m"
        
        # OHLCV data buffers
        self.timestamps = deque(maxlen=buffer_size)
        self.opens = deque(maxlen=buffer_size)
        self.highs = deque(maxlen=buffer_size) 
        self.lows = deque(maxlen=buffer_size)
        self.closes = deque(maxlen=buffer_size)
        self.volumes = deque(maxlen=buffer_size)
        
        # Current state
        self.current_price = None
        self.current_volume = None
        self.last_update = None
        
        # Technical indicators cache
        self.indicators = {}
        
        # WebSocket connection
        self.websocket = None
        self.is_running = False
        
        # Data callbacks
        self.callbacks: List[Callable] = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Start WebSocket data stream"""
        self.is_running = True
        self.logger.info(f"Starting data processor for {self.symbol}")
        
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                self.websocket = websocket
                self.logger.info("WebSocket connected")
                
                async for message in websocket:
                    if not self.is_running:
                        break
                    await self._process_kline_data(message)
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            self.is_running = False
            
    async def stop(self):
        """Stop data processing"""
        self.is_running = False
        if self.websocket:
            await self.websocket.close()
        self.logger.info("Data processor stopped")
        
    async def _process_kline_data(self, message: str):
        """Process incoming kline data"""
        try:
            data = json.loads(message)
            kline = data['k']
            
            # Only process closed candles for consistency
            if not kline['x']:  # x = is_closed
                return
                
            # Extract OHLCV
            timestamp = kline['t']
            open_price = float(kline['o'])
            high_price = float(kline['h'])
            low_price = float(kline['l'])
            close_price = float(kline['c'])
            volume = float(kline['v'])
            
            # Update buffers
            self.timestamps.append(timestamp)
            self.opens.append(open_price)
            self.highs.append(high_price)
            self.lows.append(low_price)
            self.closes.append(close_price)
            self.volumes.append(volume)
            
            # Update current state
            self.current_price = close_price
            self.current_volume = volume
            self.last_update = datetime.fromtimestamp(timestamp / 1000)
            
            # Calculate indicators
            self._calculate_indicators()
            
            # Notify callbacks
            await self._notify_callbacks()
            
        except Exception as e:
            self.logger.error(f"Error processing kline data: {e}")
            
    def _calculate_indicators(self):
        """Calculate technical indicators"""
        if len(self.closes) < 50:  # Need sufficient data
            return
            
        # Convert to numpy arrays for talib
        high_array = np.array(list(self.highs))
        low_array = np.array(list(self.lows)) 
        close_array = np.array(list(self.closes))
        volume_array = np.array(list(self.volumes))
        
        try:
            # Price-based indicators
            self.indicators.update({
                # Moving averages
                'sma_5': talib.SMA(close_array, timeperiod=5)[-1] if len(close_array) >= 5 else close_array[-1],
                'sma_10': talib.SMA(close_array, timeperiod=10)[-1] if len(close_array) >= 10 else close_array[-1],
                'sma_20': talib.SMA(close_array, timeperiod=20)[-1] if len(close_array) >= 20 else close_array[-1],
                'ema_12': talib.EMA(close_array, timeperiod=12)[-1] if len(close_array) >= 12 else close_array[-1],
                'ema_26': talib.EMA(close_array, timeperiod=26)[-1] if len(close_array) >= 26 else close_array[-1],
                
                # Momentum indicators
                'rsi_14': talib.RSI(close_array, timeperiod=14)[-1] if len(close_array) >= 14 else 50.0,
                'macd': talib.MACD(close_array)[0][-1] if len(close_array) >= 26 else 0.0,
                'macd_signal': talib.MACD(close_array)[1][-1] if len(close_array) >= 26 else 0.0,
                
                # Volatility indicators
                'atr_14': talib.ATR(high_array, low_array, close_array, timeperiod=14)[-1] if len(close_array) >= 14 else 0.0,
                'bb_upper': talib.BBANDS(close_array, timeperiod=20)[0][-1] if len(close_array) >= 20 else close_array[-1],
                'bb_lower': talib.BBANDS(close_array, timeperiod=20)[2][-1] if len(close_array) >= 20 else close_array[-1],
                
                # Volume indicators  
                'volume_sma_20': talib.SMA(volume_array, timeperiod=20)[-1] if len(volume_array) >= 20 else volume_array[-1],
            })
            
            # Custom indicators
            self._calculate_custom_indicators(close_array, volume_array)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            
    def _calculate_custom_indicators(self, prices: np.ndarray, volumes: np.ndarray):
        """Calculate custom indicators used in your training"""
        try:
            current_price = prices[-1]
            
            # Price changes
            if len(prices) >= 2:
                self.indicators['price_change_1m'] = (prices[-1] - prices[-2]) / prices[-2]
            else:
                self.indicators['price_change_1m'] = 0.0
                
            if len(prices) >= 6:
                self.indicators['price_change_5m'] = (prices[-1] - prices[-6]) / prices[-6]
            else:
                self.indicators['price_change_5m'] = 0.0
                
            # Volatility measures
            if len(prices) >= 20:
                returns = np.diff(np.log(prices[-20:]))
                self.indicators['volatility_20'] = np.std(returns)
                
                # Z-score (mean reversion signal)
                mean_price = np.mean(prices[-20:])
                std_price = np.std(prices[-20:])
                if std_price > 0:
                    self.indicators['z_score'] = (current_price - mean_price) / std_price
                else:
                    self.indicators['z_score'] = 0.0
            else:
                self.indicators['volatility_20'] = 0.0
                self.indicators['z_score'] = 0.0
                
            # Volume ratios
            current_volume = volumes[-1]
            if len(volumes) >= 20:
                avg_volume = np.mean(volumes[-20:])
                self.indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                self.indicators['volume_ratio'] = 1.0
                
            # VWAP deviation
            if len(prices) >= 20 and len(volumes) >= 20:
                vwap = np.sum(prices[-20:] * volumes[-20:]) / np.sum(volumes[-20:])
                self.indicators['vwap_deviation'] = (current_price - vwap) / vwap if vwap > 0 else 0.0
            else:
                self.indicators['vwap_deviation'] = 0.0
                
            # Trend strength
            if len(prices) >= 20:
                # Simple trend strength based on EMA deviation
                ema_12 = self.indicators.get('ema_12', current_price)
                ema_26 = self.indicators.get('ema_26', current_price)
                self.indicators['trend_strength'] = (ema_12 - ema_26) / current_price if current_price > 0 else 0.0
            else:
                self.indicators['trend_strength'] = 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating custom indicators: {e}")
            
    def get_feature_vector(self) -> Optional[np.ndarray]:
        """
        Get feature vector for DRL model input
        Should match the features used in training
        """
        if not self.indicators or len(self.closes) < 20:
            return None
            
        try:
            current_price = self.current_price
            
            # Create feature vector - adjust based on your training features
            features = [
                # Price features
                self.indicators.get('price_change_1m', 0.0),
                self.indicators.get('price_change_5m', 0.0),
                
                # Technical indicators (normalized)
                self.indicators.get('rsi_14', 50.0) / 100.0,  # Normalize RSI to [0,1]
                self.indicators.get('atr_14', 0.0) / current_price if current_price > 0 else 0.0,  # Normalized ATR
                
                # Mean reversion
                self.indicators.get('z_score', 0.0),
                
                # Trend indicators
                (self.indicators.get('ema_12', current_price) - current_price) / current_price if current_price > 0 else 0.0,
                (self.indicators.get('ema_26', current_price) - current_price) / current_price if current_price > 0 else 0.0,
                self.indicators.get('trend_strength', 0.0),
                
                # Volatility
                self.indicators.get('volatility_20', 0.0),
                
                # Volume
                self.indicators.get('volume_ratio', 1.0) - 1.0,  # Center around 0
                self.indicators.get('vwap_deviation', 0.0),
                
                # Bollinger Bands position
                self._get_bb_position(),
                
                # MACD
                self.indicators.get('macd', 0.0) / current_price if current_price > 0 else 0.0,
            ]
            
            # Convert to numpy array and handle any NaN/inf values
            feature_array = np.array(features, dtype=np.float32)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return feature_array
            
        except Exception as e:
            self.logger.error(f"Error creating feature vector: {e}")
            return None
            
    def _get_bb_position(self) -> float:
        """Calculate position within Bollinger Bands"""
        try:
            bb_upper = self.indicators.get('bb_upper', self.current_price)
            bb_lower = self.indicators.get('bb_lower', self.current_price)
            
            if bb_upper > bb_lower:
                return (self.current_price - bb_lower) / (bb_upper - bb_lower)
            else:
                return 0.5  # Middle position if bands are invalid
                
        except Exception as e:
            return 0.5
            
    async def _notify_callbacks(self):
        """Notify all registered callbacks"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.get_current_state())
                else:
                    callback(self.get_current_state())
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
                
    def register_callback(self, callback: Callable):
        """Register callback for data updates"""
        self.callbacks.append(callback)
        
    def get_current_state(self) -> Dict:
        """Get current market state"""
        return {
            'price': self.current_price,
            'volume': self.current_volume,
            'timestamp': self.last_update,
            'indicators': self.indicators.copy(),
            'feature_vector': self.get_feature_vector(),
            'data_points': len(self.closes)
        }
        
    def is_ready(self) -> bool:
        """Check if processor has sufficient data"""
        return (len(self.closes) >= 50 and 
                self.current_price is not None and 
                len(self.indicators) > 0)