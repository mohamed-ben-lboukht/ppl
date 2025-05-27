"""
Professional Configuration Management System
Handles environment-specific settings, logging, and application configuration.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path


class DatabaseConfig:
    """Database configuration settings"""
    def __init__(self):
        self.type = "sqlite"
        self.url = "sqlite:///keystroke_analytics.db"
        self.pool_size = 10
        self.max_overflow = 20
        self.echo = False


class ModelConfig:
    """Machine learning model configuration"""
    def __init__(self):
        self.model1_path = "models/model1_weights.pth"
        self.model2_path = "models/model2_weights.pth"
        self.model3_path = "models/model3_weights.pth"
        self.min_keystrokes = 5
        self.max_keystrokes = 1000
        self.timing_precision = "microseconds"
        self.cache_predictions = True
        self.batch_size = 32


class SecurityConfig:
    """Security and authentication settings"""
    def __init__(self):
        self.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
        self.admin_password = os.getenv('ADMIN_PASSWORD', 'admin123')
        self.rate_limit = "100 per hour"
        self.cors_origins = None
        self.csrf_enabled = True


class AppConfig:
    """Main application configuration"""
    
    def __init__(self, config_name: str = None):
        """Initialize configuration"""
        # Set environment based on config_name or environment variable
        if config_name:
            self.environment = config_name
        else:
            self.environment = os.getenv('FLASK_ENV', 'development')
        
        # Initialize other attributes
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true' or self.environment == 'development'
        self.testing = False
        
        # Server settings
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', 5000))
        
        # Directories
        self.base_dir = Path(__file__).parent.absolute()
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.static_dir = self.base_dir / "static"
        self.templates_dir = self.base_dir / "templates"
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.log_file = 'keystroke_analytics.log'
        
        # API settings
        self.api_title = "Keystroke Analytics API"
        self.api_version = "1.0.0"
        self.api_description = "Professional keystroke timing analysis and user profiling system"
        
        # Data collection
        self.max_contributions_per_ip = 10
        self.data_retention_days = 365
        
        # CORS settings
        cors_origins_env = os.getenv('CORS_ORIGINS', '*')
        if cors_origins_env == '*':
            self.cors_origins = ['*']
        else:
            self.cors_origins = cors_origins_env.split(',')
        
        # Rate limiting
        self.rate_limits = ["100 per hour"]
        
        # Sub-configurations
        self.database_config = DatabaseConfig()
        self.model_config = ModelConfig()
        self.security_config = SecurityConfig()
        
        # Initialize
        self._create_directories()
        self._setup_logging()
        self._validate_config()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.data_dir,
            self.models_dir,
            self.logs_dir,
            self.static_dir,
            self.templates_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure professional logging system"""
        log_file_path = self.logs_dir / self.log_file
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format=self.log_format,
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )
        
        # Configure specific loggers
        app_logger = logging.getLogger('keystroke_analytics')
        app_logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Suppress noisy third-party loggers in production
        if self.environment == 'production':
            logging.getLogger('werkzeug').setLevel(logging.ERROR)
            logging.getLogger('urllib3').setLevel(logging.ERROR)
    
    def _validate_config(self):
        """Validate configuration settings"""
        if self.environment == 'production':
            if self.security_config.secret_key == 'your-secret-key-change-in-production':
                raise ValueError("Secret key must be changed in production!")
            
            if self.security_config.admin_password == 'admin123':
                raise ValueError("Admin password must be changed in production!")
        
        # Validate paths
        required_files = [
            self.models_dir / "model1_weights.pth",
            self.models_dir / "model2_weights.pth", 
            self.models_dir / "model3_weights.pth"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            logging.warning(f"Missing model files: {missing_files}")
    
    def get_database_url(self) -> str:
        """Get properly formatted database URL"""
        return self.database_config.url
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == 'production'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'api_title': self.api_title,
            'api_version': self.api_version,
            'host': self.host,
            'port': self.port,
            'log_level': self.log_level
        }


# Global configuration instance
config = AppConfig()

# Convenience function for getting configuration
def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config


# Environment-specific configurations
class DevelopmentConfig(AppConfig):
    """Development environment configuration"""
    def __init__(self):
        super().__init__()
        self.debug = True
        self.database.echo = True
        self.log_level = 'DEBUG'


class ProductionConfig(AppConfig):
    """Production environment configuration"""
    def __init__(self):
        super().__init__()
        self.debug = False
        self.testing = False
        self.database.echo = False
        self.log_level = 'INFO'
        

class TestingConfig(AppConfig):
    """Testing environment configuration"""
    def __init__(self):
        super().__init__()
        self.testing = True
        self.debug = True
        self.database.url = "sqlite:///:memory:"
        self.log_level = 'DEBUG'


# Configuration factory
def get_config_by_environment(env: str) -> AppConfig:
    """Get configuration based on environment name"""
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    config_class = configs.get(env, AppConfig)
    return config_class() 