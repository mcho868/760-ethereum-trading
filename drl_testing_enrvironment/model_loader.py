import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Optional, Dict, Any
from stable_baselines3 import A2C
import logging

class ModelLoader:
    """
    Simple model loader for trained DRL agents
    Handles loading A2C models and any associated preprocessors
    """
    
    def __init__(self, model_path: str, scaler_path: str = None):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        
        # Model components
        self.model: Optional[A2C] = None
        self.scaler: Optional[Any] = None
        
        # Model info
        self.observation_space_dim = 0
        self.action_space_dim = 0
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Load components
        self._load_model()
        self._load_scaler()
        
    def _load_model(self):
        """Load the trained DRL model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            self.logger.info(f"Loading A2C model from {self.model_path}")
            self.model = A2C.load(str(self.model_path))
            
            # Get dimensions
            self.observation_space_dim = self.model.observation_space.shape[0]
            if hasattr(self.model.action_space, 'shape'):
                self.action_space_dim = self.model.action_space.shape[0] if self.model.action_space.shape else 1
            else:
                self.action_space_dim = 1
                
            self.logger.info(f"Model loaded - Obs dim: {self.observation_space_dim}, Action dim: {self.action_space_dim}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def _load_scaler(self):
        """Load feature scaler if available"""
        if not self.scaler_path or not self.scaler_path.exists():
            self.logger.info("No scaler file provided or found")
            return
            
        try:
            self.logger.info(f"Loading scaler from {self.scaler_path}")
            
            if self.scaler_path.suffix == '.pkl':
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            elif self.scaler_path.suffix == '.joblib':
                self.scaler = joblib.load(self.scaler_path)
            else:
                # Try pickle as default
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    
            self.logger.info("Scaler loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load scaler: {e}")
            self.scaler = None
            
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> float:
        """
        Make prediction using the loaded model
        
        Args:
            observation: Feature vector matching training data format
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action value (typically in range [-1, 1] for trading)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # Ensure observation is correct shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            
        # Apply scaling if available
        if self.scaler is not None:
            observation = self.scaler.transform(observation)
            
        # Get prediction
        action, _states = self.model.predict(observation, deterministic=deterministic)
        
        # Return scalar action
        if isinstance(action, np.ndarray):
            return float(action[0]) if len(action) > 0 else 0.0
        else:
            return float(action)
            
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_type': 'A2C',
            'observation_dim': self.observation_space_dim,
            'action_dim': self.action_space_dim,
            'has_scaler': self.scaler is not None,
            'model_loaded': self.model is not None
        }
        
    def is_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return self.model is not None