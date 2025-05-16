import numpy as np
import torch
from scipy.stats import skew

class Model1Processor:
    """Process keystroke data using basic statistical features (Model 1)"""
    
    def __init__(self, model_path='model_weights.pth'):
        self.model_path = model_path
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = ['mean', 'std', 'min', 'max', 'total', 'median', 'skew']
        
        # Pre-initialize scaler with fixed values to ensure deterministic predictions
        self.scaler.means_ = np.array([300.0, 150.0, 50.0, 600.0, 5000.0, 250.0, 0.5])
        self.scaler.stds_ = np.array([100.0, 75.0, 25.0, 200.0, 2000.0, 100.0, 1.0])
        
        # Cache for deterministic predictions
        self.prediction_cache = {}
    
    def load_model(self, model_class):
        """Load the PyTorch model"""
        try:
            self.model = model_class()
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading Model 1: {e}")
            return False
    
    def extract_features(self, keystroke_timings):
        """Extract basic statistical features from keystroke timings"""
        if len(keystroke_timings) < 5:
            return None
        
        # Convert to milliseconds
        timing_ms = [t / 1000 for t in keystroke_timings]
        
        # Generate a cache key for deterministic results
        cache_key = self._generate_cache_key(timing_ms)
        
        # Check if we have a cached prediction
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]['features']
        
        # Calculate the statistical features
        try:
            features = [
                np.mean(timing_ms),
                np.std(timing_ms) if np.std(timing_ms) != 0 else 1e-6,
                np.min(timing_ms),
                np.max(timing_ms),
                np.sum(timing_ms),
                np.median(timing_ms),
                skew(timing_ms)
            ]
            
            # Make sure we have all 7 features
            if len(features) < 7:
                features.extend([0] * (7 - len(features)))
            
            # Store in cache
            if cache_key not in self.prediction_cache:
                self.prediction_cache[cache_key] = {'features': features}
                
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def normalize_features(self, features):
        """Normalize features using RobustScaler with fixed parameters"""
        # Get cache key to use the same normalized features for identical inputs
        for key, data in self.prediction_cache.items():
            if data['features'] == features:
                if 'normalized' in data:
                    return data['normalized']
                break
                
        # Apply fixed normalization
        features_array = np.array(features).reshape(1, -1)
        normalized = self.scaler.transform(features_array).flatten()
        
        # Store in cache
        for key, data in self.prediction_cache.items():
            if data['features'] == features:
                data['normalized'] = normalized
                break
                
        return normalized
    
    def predict(self, features):
        """Make prediction using the loaded model"""
        if self.model is None:
            return None
            
        # Check cache for previous prediction
        for key, data in self.prediction_cache.items():
            if 'normalized' in data and np.array_equal(data['normalized'], features):
                if 'prediction' in data:
                    return data['prediction']
                break
                
        # Convert features to torch tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            age_pred, gender_pred, handedness_pred, class_pred = self.model(input_tensor)
        
        # Get predicted values
        age = int(age_pred.item() * 50 + 25)  # Adjust scaling based on your model
        gender = "Male" if torch.argmax(gender_pred, dim=1).item() == 1 else "Female"
        handedness = "Right-handed" if torch.argmax(handedness_pred, dim=1).item() == 1 else "Left-handed"
        user_class = "Class A" if torch.argmax(class_pred, dim=1).item() == 1 else "Class B"
        
        result = {
            'age': age,
            'gender': gender,
            'handedness': handedness,
            'class': user_class
        }
        
        # Store in cache
        for key, data in self.prediction_cache.items():
            if 'normalized' in data and np.array_equal(data['normalized'], features):
                data['prediction'] = result
                break
                
        return result
        
    def _generate_cache_key(self, timings):
        """Generate a cache key for a given set of keystroke timings"""
        # Round to reduce minor variations and improve cache hits
        rounded = [round(t, 2) for t in timings]
        return hash(str(rounded))


class RobustScaler:
    """Scaling class that matches the one used in training"""
    def __init__(self):
        self.means_ = None
        self.stds_ = None
    
    def fit(self, X):
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        self.stds_[self.stds_ == 0] = 1e-6
        return self
        
    def transform(self, X):
        if self.means_ is None or self.stds_ is None:
            return X
        return (X - self.means_) / self.stds_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X):
        return X * self.stds_ + self.means_ 