import numpy as np
import torch
from scipy.stats import skew

class Model3Processor:
    """Process keystroke data using advanced combined features (Model 3)"""
    
    def __init__(self, model_path='model3_weights.pth'):
        self.model_path = model_path
        self.model = None
        self.scaler = RobustScaler()
        self.hist_bins = 10  # Use 10 bins for histogram part
        self.min_range = 0
        self.max_range = 1000  # milliseconds
        
        # Pre-initialize scaler with fixed values for deterministic predictions
        # 10 histogram bins + 10 statistical features = 20 total features
        hist_means = [0.05] * 10  # Typical histogram bin values (uniform distribution)
        stat_means = [300.0, 150.0, 50.0, 600.0, 5000.0, 250.0, 0.5, 150.0, 450.0, 300.0]
        
        hist_stds = [0.07] * 10  # Typical standard deviations for histogram bins
        stat_stds = [100.0, 75.0, 25.0, 200.0, 2000.0, 100.0, 1.0, 50.0, 150.0, 100.0]
        
        self.scaler.means_ = np.array(hist_means + stat_means[:10])
        self.scaler.stds_ = np.array(hist_stds + stat_stds[:10])
        
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
            print(f"Error loading Model 3: {e}")
            return False
    
    def extract_features(self, keystroke_timings):
        """Extract combined features from keystroke timings"""
        if len(keystroke_timings) < 5:
            return None
        
        # Convert to milliseconds
        timing_ms = [t / 1000 for t in keystroke_timings]
        
        # Generate a cache key for deterministic results
        cache_key = self._generate_cache_key(timing_ms)
        
        # Check if we have a cached prediction
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]['features']
        
        try:
            # Compute histogram features (10 bins)
            hist_features = self.compute_histogram_features(timing_ms, bins=self.hist_bins)
            
            # Compute basic statistical features
            basic_features = self.compute_basic_features(timing_ms)
            
            # Combine both feature sets (10+10=20 features total)
            combined_features = np.concatenate([hist_features, basic_features[:10]])
            
            # Store in cache
            if cache_key not in self.prediction_cache:
                self.prediction_cache[cache_key] = {'features': combined_features}
                
            return combined_features
        except Exception as e:
            print(f"Error extracting combined features: {e}")
            return None
    
    def compute_histogram_features(self, timings, bins=10, min_range=None, max_range=None):
        """Compute histogram features exactly as in the notebook"""
        if min_range is None:
            min_range = self.min_range
        if max_range is None:
            max_range = self.max_range
            
        if np.any(np.isnan(timings)) or np.any(np.isinf(timings)) or np.max(timings) == np.min(timings):
            return np.zeros(bins)
            
        hist, _ = np.histogram(timings, bins=bins, range=(min_range, max_range))
        hist_sum = hist.sum()
        
        if hist_sum == 0:
            return np.zeros(bins)
            
        return hist / hist_sum
    
    def compute_basic_features(self, timings):
        """Compute basic statistical features"""
        features = [
            np.mean(timings),
            np.std(timings) if np.std(timings) != 0 else 1e-6,
            np.min(timings),
            np.max(timings),
            np.sum(timings),
            np.median(timings),
            skew(timings),
            np.percentile(timings, 25),  # Q1
            np.percentile(timings, 75),  # Q3
            np.percentile(timings, 75) - np.percentile(timings, 25),  # IQR
            # Additional features if needed to reach 10 total
            np.std(timings) / np.mean(timings) if np.mean(timings) > 0 else 0  # coef of variation
        ]
        
        # Ensure we have at least 10 features for the basic part
        if len(features) < 10:
            features.extend([0] * (10 - len(features)))
            
        return features
    
    def normalize_features(self, features):
        """Normalize features using RobustScaler with fixed parameters"""
        # Get cache key to use the same normalized features for identical inputs
        for key, data in self.prediction_cache.items():
            if np.array_equal(data['features'], features):
                if 'normalized' in data:
                    return data['normalized']
                break
                
        # Apply fixed normalization
        features_array = np.array(features).reshape(1, -1)
        normalized = self.scaler.transform(features_array).flatten()
        
        # Store in cache
        for key, data in self.prediction_cache.items():
            if np.array_equal(data['features'], features):
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
        
        # Process predictions
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