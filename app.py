#!/usr/bin/env python3
"""
Professional Keystroke Analytics Application
Advanced keystroke timing analysis and user profiling system with ML models.

Author: Keystroke Analytics Team
Version: 2.0.0
License: MIT
"""

import os
import sys
import logging
import traceback
import uuid
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Flask imports
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

# Application imports
from config import AppConfig
from models import db, init_db, KeystrokeData, Prediction, UserContribution
from models.ml_models import ModelManager
from api import APIManager
from api.schemas import SchemaValidator, PredictionRequest, ContributionRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables
model_manager: Optional[ModelManager] = None
app_start_time = datetime.utcnow()


def create_app(config_name: str = None) -> Flask:
    """
    Application factory pattern for creating Flask app
    
    Args:
        config_name: Configuration environment name
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    config = AppConfig(config_name)
    app.config.update(config.to_dict())
    
    # Configure app for proxy (if behind nginx/load balancer)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Initialize extensions
    setup_extensions(app, config)
    
    # Initialize models and database
    init_db(app, config)
    
    # Setup error handlers
    setup_error_handlers(app)
    
    # Register blueprints and routes
    register_routes(app)
    
    # Initialize ML models
    initialize_ml_models(app, config)
    
    logger.info(f"Application created with config: {config_name or 'default'}")
    return app


def setup_extensions(app: Flask, config: AppConfig) -> None:
    """Setup Flask extensions"""
    
    # CORS configuration
    CORS(app, 
         origins=config.cors_origins,
         methods=['GET', 'POST', 'PUT', 'DELETE'],
         allow_headers=['Content-Type', 'Authorization'],
         supports_credentials=True)
    
    # Rate limiting
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=config.rate_limits,
        storage_uri="memory://"
    )
    
    # Store in app context for access in routes
    app.limiter = limiter
    
    logger.info("Extensions configured successfully")


def setup_error_handlers(app: Flask) -> None:
    """Setup comprehensive error handling"""
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        """Handle HTTP exceptions"""
        logger.warning(f"HTTP Exception: {e.code} - {e.description}")
        return jsonify({
            'error': {
                'code': e.code,
                'name': e.name,
                'description': e.description
            },
            'timestamp': datetime.utcnow().isoformat()
        }), e.code
    
    @app.errorhandler(Exception)
    def handle_general_exception(e):
        """Handle general exceptions"""
        error_id = str(uuid.uuid4())
        logger.error(f"Unhandled exception {error_id}: {str(e)}\n{traceback.format_exc()}")
        
        return jsonify({
            'error': {
                'code': 500,
                'name': 'Internal Server Error',
                'description': 'An unexpected error occurred',
                'error_id': error_id
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 500
    
    @app.errorhandler(429)
    def handle_rate_limit(e):
        """Handle rate limit exceeded"""
        logger.warning(f"Rate limit exceeded: {request.remote_addr}")
        return jsonify({
            'error': {
                'code': 429,
                'name': 'Too Many Requests',
                'description': 'Rate limit exceeded. Please try again later.',
                'retry_after': e.retry_after
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 429


def initialize_ml_models(app: Flask, config: AppConfig) -> None:
    """Initialize machine learning models"""
    global model_manager
    
    try:
        models_dir = Path(config.model_config.model1_path).parent
        model_manager = ModelManager(models_dir)
        
        available_models = model_manager.get_available_models()
        logger.info(f"Loaded ML models: {available_models}")
        
        if not available_models:
            logger.warning("No ML models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize ML models: {e}")
        model_manager = None


def register_routes(app: Flask) -> None:
    """Register all application routes"""
    
    # API routes
    api_manager = APIManager(app)
    api_manager.register_api_routes()
    
    # Web interface routes
    register_web_routes(app)
    
    # Health and monitoring routes
    register_health_routes(app)


def register_web_routes(app: Flask) -> None:
    """Register web interface routes"""
    
    @app.route('/')
    def index():
        """Main application page"""
        try:
            return render_template('index.html',
                                 title="Keystroke Analytics",
                                 version=app.config.get('VERSION', '2.0.0'))
        except Exception as e:
            logger.error(f"Error rendering index: {e}")
            return f"Application Error: {str(e)}", 500
    
    @app.route('/contribute')
    def contribute():
        """Data contribution page"""
        try:
            return render_template('contribute.html',
                                 title="Contribute Data")
        except Exception as e:
            logger.error(f"Error rendering contribute page: {e}")
            return f"Application Error: {str(e)}", 500
    
    @app.route('/admin')
    def admin():
        """Admin dashboard page"""
        try:
            return render_template('admin.html',
                                 title="Admin Dashboard")
        except Exception as e:
            logger.error(f"Error rendering admin page: {e}")
            return f"Application Error: {str(e)}", 500
    
    @app.route('/docs')
    def api_docs():
        """API documentation page"""
        return render_template('api_docs.html', title="API Documentation")
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files"""
        return send_from_directory(app.static_folder, filename)


def register_health_routes(app: Flask) -> None:
    """Register health check and monitoring routes"""
    
    @app.route('/api/health')
    def health_check():
        """Comprehensive health check endpoint"""
        try:
            # Basic health indicators
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': app.config.get('VERSION', '2.0.0'),
                'environment': app.config.get('ENVIRONMENT', 'development'),
                'uptime_seconds': (datetime.utcnow() - app_start_time).total_seconds()
            }
            
            # Database health
            try:
                db.session.execute('SELECT 1')
                health_status['database_status'] = 'healthy'
            except Exception as e:
                health_status['database_status'] = f'unhealthy: {str(e)}'
                health_status['status'] = 'degraded'
            
            # ML models health
            if model_manager:
                available_models = model_manager.get_available_models()
                health_status['models_loaded'] = available_models
                if not available_models:
                    health_status['status'] = 'degraded'
            else:
                health_status['models_loaded'] = []
                health_status['status'] = 'degraded'
            
            # System metrics
            process = psutil.Process()
            health_status['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            health_status['cpu_usage_percent'] = process.cpu_percent()
            
            # Set appropriate HTTP status code
            status_code = 200 if health_status['status'] == 'healthy' else 503
            
            return jsonify(health_status), status_code
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 503
    
    @app.route('/api/metrics')
    def metrics():
        """Application metrics endpoint"""
        try:
            # Database metrics
            total_predictions = db.session.query(Prediction).count()
            total_contributions = db.session.query(UserContribution).count()
            total_sessions = db.session.query(KeystrokeData).distinct(KeystrokeData.session_id).count()
            
            # Recent activity (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_predictions = db.session.query(Prediction).filter(
                Prediction.created_at >= yesterday
            ).count()
            
            # Model performance metrics
            model_stats = {}
            if model_manager:
                model_stats = model_manager.get_model_stats()
            
            # System metrics
            process = psutil.Process()
            system_stats = {
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_usage_percent': process.cpu_percent(),
                'disk_usage': psutil.disk_usage('/'),
                'uptime_seconds': (datetime.utcnow() - app_start_time).total_seconds()
            }
            
            metrics_data = {
                'database_metrics': {
                    'total_predictions': total_predictions,
                    'total_contributions': total_contributions,
                    'total_sessions': total_sessions,
                    'recent_predictions_24h': recent_predictions
                },
                'model_metrics': model_stats,
                'system_metrics': system_stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return jsonify(metrics_data)
            
        except Exception as e:
            logger.error(f"Metrics endpoint failed: {e}")
            return jsonify({'error': str(e)}), 500


# API Routes (moved to separate module for organization)
def create_api_routes(app: Flask) -> None:
    """Create API routes - now handled by APIManager"""
    pass


# Application startup and configuration
def ensure_directories():
    """Ensure required directories exist"""
    directories = ['logs', 'data', 'models', 'static', 'templates']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def setup_logging(config: AppConfig):
    """Setup application logging"""
    log_level = getattr(logging, config.log_level.upper())
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


# Application factory and entry point
app = None

def get_app() -> Flask:
    """Get or create application instance"""
    global app
    if app is None:
        ensure_directories()
        app = create_app()
    return app


if __name__ == '__main__':
    try:
        # Setup
        ensure_directories()
        
        # Create application
        app = create_app()
        
        # Get configuration
        config = AppConfig()
        setup_logging(config)
        
        logger.info("="*60)
        logger.info("KEYSTROKE ANALYTICS APPLICATION STARTING")
        logger.info("="*60)
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Debug mode: {config.debug}")
        logger.info(f"Host: {config.host}")
        logger.info(f"Port: {config.port}")
        logger.info("="*60)
        
        # Create database tables
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully")
        
        # Start application
        app.run(
            host=config.host,
            port=config.port,
            debug=config.debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")


# For WSGI servers (Gunicorn, uWSGI, etc.)
application = get_app() 