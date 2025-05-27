"""
Professional Machine Learning Models Management
Handles ML model loading, prediction, and performance monitoring.
"""

import logging
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from threading import Lock
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseNeuralNetwork(nn.Module, ABC):
    """
    Base class for all neural network models
    """
    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__
        self.version = "1.0.0"
        self.trained_at = None
        
    @abstractmethod
    def forward(self, x):
        """Forward pass through the network"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': self.model_name,
            'version': self.version,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trained_at': self.trained_at,
            'device': next(self.parameters()).device.type
        }


class Model1Net(BaseNeuralNetwork):
    """
    Model 1: Basic statistical features neural network
    """
    def __init__(self):
        super().__init__()
        self.model_name = "BasicStatisticsModel"
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.age_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.gender_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.handedness_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.class_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        age = self.age_head(shared_features)
        gender = self.gender_head(shared_features)
        handedness = self.handedness_head(shared_features)
        class_output = self.class_head(shared_features)
        
        return age, gender, handedness, class_output


class Model2Net(BaseNeuralNetwork):
    """
    Model 2: Histogram-based features neural network
    """
    def __init__(self):
        super().__init__()
        self.model_name = "HistogramModel"
        
        # Shared layers for histogram features
        self.shared = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.age_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.gender_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.handedness_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.class_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        age = self.age_head(shared_features)
        gender = self.gender_head(shared_features)
        handedness = self.handedness_head(shared_features)
        class_output = self.class_head(shared_features)
        
        return age, gender, handedness, class_output


class Model3Net(BaseNeuralNetwork):
    """
    Model 3: Advanced combined features neural network
    """
    def __init__(self):
        super().__init__()
        self.model_name = "AdvancedCombinedModel"
        
        # Shared layers for combined features
        self.shared = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.age_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.gender_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.handedness_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.class_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        age = self.age_head(shared_features)
        gender = self.gender_head(shared_features)
        handedness = self.handedness_head(shared_features)
        class_output = self.class_head(shared_features)
        
        return age, gender, handedness, class_output


class FeatureExtractor:
    """
    Professional feature extraction with validation and caching
    """
    
    @staticmethod
    def validate_timing_data(timing_data: List[float]) -> Tuple[bool, str]:
        """
        Validate keystroke timing data
        
        Args:
            timing_data: List of timing values in microseconds
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not timing_data:
            return False, "No timing data provided"
        
        if len(timing_data) < 5:
            return False, "Insufficient keystrokes (minimum 5 required)"
        
        if len(timing_data) > 1000:
            return False, "Too many keystrokes (maximum 1000 allowed)"
        
        # Check for reasonable timing values (1ms to 5s)
        valid_timings = [t for t in timing_data if 1000 <= t <= 5000000]
        
        if len(valid_timings) < len(timing_data) * 0.8:
            return False, "Too many unreasonable timing values"
        
        return True, "Valid"
    
    @staticmethod
    def extract_basic_features(timing_data: List[float]) -> Optional[np.ndarray]:
        """
        Extract basic statistical features (Model 1)
        
        Args:
            timing_data: List of timing values in microseconds
            
        Returns:
            NumPy array of features or None if invalid
        """
        is_valid, error_msg = FeatureExtractor.validate_timing_data(timing_data)
        if not is_valid:
            logger.warning(f"Invalid timing data for basic features: {error_msg}")
            return None
        
        try:
            # Convert to milliseconds for processing
            timing_ms = np.array([t / 1000.0 for t in timing_data])
            
            # Calculate statistical features
            features = [
                np.mean(timing_ms),
                np.std(timing_ms) if np.std(timing_ms) > 0 else 1e-6,
                np.min(timing_ms),
                np.max(timing_ms),
                np.sum(timing_ms),
                np.median(timing_ms),
                float(scipy.stats.skew(timing_ms)) if len(timing_ms) > 2 else 0.0
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting basic features: {e}")
            return None
    
    @staticmethod
    def extract_histogram_features(timing_data: List[float], num_bins: int = 20) -> Optional[np.ndarray]:
        """
        Extract histogram features (Model 2)
        
        Args:
            timing_data: List of timing values in microseconds
            num_bins: Number of histogram bins
            
        Returns:
            NumPy array of features or None if invalid
        """
        is_valid, error_msg = FeatureExtractor.validate_timing_data(timing_data)
        if not is_valid:
            logger.warning(f"Invalid timing data for histogram features: {error_msg}")
            return None
        
        try:
            # Convert to milliseconds
            timing_ms = np.array([t / 1000.0 for t in timing_data])
            
            # Create adaptive bins
            min_time = np.min(timing_ms)
            max_time = np.max(timing_ms)
            
            if max_time - min_time < 20:  # Ensure minimum range
                max_time = min_time + 20
            
            # Create histogram
            bins = np.linspace(min_time, max_time, num_bins + 1)
            hist, _ = np.histogram(timing_ms, bins=bins, density=True)
            
            # Normalize
            if np.sum(hist) > 0:
                hist = hist / np.sum(hist)
            
            return hist.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting histogram features: {e}")
            return None
    
    @staticmethod
    def extract_advanced_features(timing_data: List[float]) -> Optional[np.ndarray]:
        """
        Extract advanced combined features (Model 3)
        
        Args:
            timing_data: List of timing values in microseconds
            
        Returns:
            NumPy array of features or None if invalid
        """
        is_valid, error_msg = FeatureExtractor.validate_timing_data(timing_data)
        if not is_valid:
            logger.warning(f"Invalid timing data for advanced features: {error_msg}")
            return None
        
        try:
            # Get histogram features (10 bins)
            hist_features = FeatureExtractor.extract_histogram_features(timing_data, num_bins=10)
            
            # Get basic statistical features (first 10)
            basic_features = FeatureExtractor.extract_basic_features(timing_data)
            
            if hist_features is None or basic_features is None:
                return None
            
            # Take only first 10 basic features and combine
            combined_features = np.concatenate([hist_features, basic_features[:10]])
            
            return combined_features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting advanced features: {e}")
            return None


class KeystrokePredictor:
    """
    Professional keystroke analysis predictor
    """
    
    def __init__(self, model_path: str, model_class: type, feature_extractor_method: str):
        """
        Initialize predictor
        
        Args:
            model_path: Path to model weights
            model_class: Model class to instantiate
            feature_extractor_method: Method name for feature extraction
        """
        self.model_path = Path(model_path)
        self.model_class = model_class
        self.feature_extractor_method = feature_extractor_method
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        self.load_lock = Lock()
        
        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.last_prediction_time = None
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_max_size = 1000
    
    def load_model(self) -> bool:
        """
        Load the model weights
        
        Returns:
            True if successful, False otherwise
        """
        with self.load_lock:
            if self.is_loaded:
                return True
            
            try:
                if not self.model_path.exists():
                    logger.error(f"Model file not found: {self.model_path}")
                    return False
                
                self.model = self.model_class()
                
                # Load weights
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                # Move to device and set eval mode
                self.model.to(self.device)
                self.model.eval()
                
                self.is_loaded = True
                logger.info(f"Successfully loaded model: {self.model.model_name}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error loading model {self.model_path}: {e}")
                return False
    
    def _generate_cache_key(self, timing_data: List[float]) -> str:
        """Generate cache key for timing data"""
        # Round to reduce cache misses from minor variations
        rounded_data = [round(t, -2) for t in timing_data]  # Round to nearest 100
        return str(hash(tuple(rounded_data)))
    
    def predict(self, timing_data: List[float]) -> Optional[Dict[str, Any]]:
        """
        Make prediction on keystroke timing data
        
        Args:
            timing_data: List of timing values in microseconds
            
        Returns:
            Dictionary with predictions or None if failed
        """
        if not self.is_loaded and not self.load_model():
            return None
        
        # Check cache first
        cache_key = self._generate_cache_key(timing_data)
        if cache_key in self.prediction_cache:
            logger.debug("Returning cached prediction")
            return self.prediction_cache[cache_key]
        
        start_time = time.time()
        
        try:
            # Extract features
            extractor_method = getattr(FeatureExtractor, self.feature_extractor_method)
            features = extractor_method(timing_data)
            
            if features is None:
                logger.warning("Feature extraction failed")
                return None
            
            # Normalize features (using model-specific normalization)
            features = self._normalize_features(features)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                age_pred, gender_pred, handedness_pred, class_pred = self.model(input_tensor)
            
            # Process predictions
            prediction_result = self._process_predictions(
                age_pred, gender_pred, handedness_pred, class_pred
            )
            
            # Update performance metrics
            prediction_time = time.time() - start_time
            self.prediction_count += 1
            self.total_prediction_time += prediction_time
            self.last_prediction_time = datetime.utcnow()
            
            prediction_result['prediction_time_ms'] = prediction_time * 1000
            
            # Cache result
            if len(self.prediction_cache) < self.cache_max_size:
                self.prediction_cache[cache_key] = prediction_result
            
            logger.debug(f"Prediction completed in {prediction_time:.3f}s")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using model-specific parameters
        
        Args:
            features: Raw features
            
        Returns:
            Normalized features
        """
        # This should be replaced with proper normalization parameters
        # that were used during training
        try:
            # Simple standardization - replace with actual training parameters
            mean = np.mean(features)
            std = np.std(features)
            
            if std > 0:
                normalized = (features - mean) / std
            else:
                normalized = features
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Feature normalization failed: {e}, using raw features")
            return features
    
    def _process_predictions(self, age_pred, gender_pred, handedness_pred, class_pred) -> Dict[str, Any]:
        """
        Process raw model predictions into human-readable format
        
        Args:
            age_pred: Age prediction tensor
            gender_pred: Gender prediction tensor
            handedness_pred: Handedness prediction tensor
            class_pred: Class prediction tensor
            
        Returns:
            Dictionary with processed predictions
        """
        # Age prediction (regression)
        age = max(18, min(80, int(age_pred.item() * 50 + 35)))  # Scale and bound
        
        # Gender prediction (classification)
        gender_probs = torch.softmax(gender_pred, dim=1)
        gender_idx = torch.argmax(gender_probs, dim=1).item()
        gender = "Male" if gender_idx == 1 else "Female"
        gender_confidence = float(gender_probs[0][gender_idx].item())
        
        # Handedness prediction (classification)
        hand_probs = torch.softmax(handedness_pred, dim=1)
        hand_idx = torch.argmax(hand_probs, dim=1).item()
        handedness = "Right-handed" if hand_idx == 1 else "Left-handed"
        handedness_confidence = float(hand_probs[0][hand_idx].item())
        
        # Class prediction (classification)
        class_probs = torch.softmax(class_pred, dim=1)
        class_idx = torch.argmax(class_probs, dim=1).item()
        user_class = "Professional" if class_idx == 1 else "Casual"
        class_confidence = float(class_probs[0][class_idx].item())
        
        return {
            'age': age,
            'gender': gender,
            'handedness': handedness,
            'class': user_class,
            'confidence_scores': {
                'gender': gender_confidence,
                'handedness': handedness_confidence,
                'class': class_confidence
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_prediction_time = (
            self.total_prediction_time / self.prediction_count 
            if self.prediction_count > 0 else 0
        )
        
        return {
            'total_predictions': self.prediction_count,
            'average_prediction_time_ms': avg_prediction_time * 1000,
            'cache_hit_ratio': len(self.prediction_cache) / max(self.prediction_count, 1),
            'last_prediction': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'model_info': self.model.get_model_info() if self.model else {},
            'cache_size': len(self.prediction_cache)
        }


class ModelManager:
    """
    Professional model management system
    """
    
    def __init__(self, models_dir: Path):
        """
        Initialize model manager
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.predictors = {}
        self._setup_models()
    
    def _setup_models(self):
        """Setup all available models"""
        model_configs = [
            {
                'name': 'model1',
                'class': Model1Net,
                'weights': self.models_dir / 'model1_weights.pth',
                'extractor': 'extract_basic_features'
            },
            {
                'name': 'model2', 
                'class': Model2Net,
                'weights': self.models_dir / 'model2_weights.pth',
                'extractor': 'extract_histogram_features'
            },
            {
                'name': 'model3',
                'class': Model3Net,
                'weights': self.models_dir / 'model3_weights.pth', 
                'extractor': 'extract_advanced_features'
            }
        ]
        
        for config in model_configs:
            try:
                predictor = KeystrokePredictor(
                    model_path=config['weights'],
                    model_class=config['class'],
                    feature_extractor_method=config['extractor']
                )
                
                if predictor.load_model():
                    self.predictors[config['name']] = predictor
                    logger.info(f"Loaded model: {config['name']}")
                else:
                    logger.warning(f"Failed to load model: {config['name']}")
                    
            except Exception as e:
                logger.error(f"Error setting up model {config['name']}: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.predictors.keys())
    
    def predict(self, model_name: str, timing_data: List[float]) -> Optional[Dict[str, Any]]:
        """
        Make prediction using specified model
        
        Args:
            model_name: Name of model to use
            timing_data: Keystroke timing data
            
        Returns:
            Prediction results or None if failed
        """
        if model_name not in self.predictors:
            logger.error(f"Model not found: {model_name}")
            return None
        
        return self.predictors[model_name].predict(timing_data)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics for all models"""
        stats = {}
        
        for name, predictor in self.predictors.items():
            stats[name] = predictor.get_performance_stats()
        
        return stats
    
    def clear_caches(self):
        """Clear prediction caches for all models"""
        for predictor in self.predictors.values():
            predictor.prediction_cache.clear()
        
        logger.info("Cleared all model caches")


# Import scipy for skewness calculation
try:
    import scipy.stats
except ImportError:
    logger.warning("scipy not available, skewness calculation will be disabled")
    
    # Fallback implementation
    class MockScipy:
        class stats:
            @staticmethod
            def skew(data):
                return 0.0
    
    scipy = MockScipy() 