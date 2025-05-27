"""
Professional API Manager
Handles all REST API endpoints with comprehensive documentation and validation.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, g
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.exceptions import BadRequest, NotFound

from models import db, KeystrokeData, Prediction, UserContribution
from api.schemas import (
    SchemaValidator, 
    PredictionRequest, 
    ContributionRequest,
    PredictionRequestSchema,
    ContributionRequestSchema,
    PredictionResponseSchema
)

logger = logging.getLogger(__name__)


class APIManager:
    """
    Professional API management system with Flask-RESTx integration
    """
    
    def __init__(self, app: Flask):
        """
        Initialize API manager
        
        Args:
            app: Flask application instance
        """
        self.app = app
        self.api = None
        self._setup_api()
        self._setup_namespaces()
    
    def _setup_api(self):
        """Setup Flask-RESTx API with documentation"""
        self.api = Api(
            self.app,
            version='2.0',
            title='Keystroke Analytics API',
            description='Professional keystroke timing analysis and user profiling API',
            doc='/api/docs/',
            prefix='/api/v2',
            validate=True,
            catch_all_404s=True
        )
        
        # Error handlers for API
        @self.api.errorhandler(BadRequest)
        def handle_bad_request(error):
            """Handle bad request errors"""
            logger.warning(f"Bad request: {error}")
            return {'error': str(error), 'timestamp': datetime.utcnow().isoformat()}, 400
        
        @self.api.errorhandler(NotFound)
        def handle_not_found(error):
            """Handle not found errors"""
            logger.warning(f"Not found: {error}")
            return {'error': 'Resource not found', 'timestamp': datetime.utcnow().isoformat()}, 404
        
        @self.api.errorhandler(Exception)
        def handle_exception(error):
            """Handle general exceptions"""
            error_id = str(uuid.uuid4())
            logger.error(f"API Exception {error_id}: {str(error)}")
            return {
                'error': 'Internal server error',
                'error_id': error_id,
                'timestamp': datetime.utcnow().isoformat()
            }, 500
    
    def _setup_namespaces(self):
        """Setup API namespaces"""
        # Prediction namespace
        self.prediction_ns = Namespace('predictions', description='Keystroke analysis and prediction operations')
        self.api.add_namespace(self.prediction_ns)
        
        # Contribution namespace
        self.contribution_ns = Namespace('contributions', description='User data contribution operations')
        self.api.add_namespace(self.contribution_ns)
        
        # Administration namespace
        self.admin_ns = Namespace('admin', description='Administrative operations')
        self.api.add_namespace(self.admin_ns)
        
        # Model namespace
        self.model_ns = Namespace('models', description='Model information and statistics')
        self.api.add_namespace(self.model_ns)
    
    def register_api_routes(self):
        """Register all API routes"""
        self._register_prediction_routes()
        self._register_contribution_routes()
        self._register_admin_routes()
        self._register_model_routes()
        logger.info("API routes registered successfully")
    
    def _register_prediction_routes(self):
        """Register prediction API routes"""
        
        # API models for documentation
        keystroke_data_model = self.api.model('KeystrokeData', {
            'timing_data': fields.List(fields.Float, required=True, 
                                     description='List of keystroke timings in microseconds'),
            'text_typed': fields.String(required=True, description='The text that was typed'),
            'session_id': fields.String(required=True, description='Unique session identifier')
        })
        
        prediction_request_model = self.api.model('PredictionRequest', {
            'keystroke_data': fields.Nested(keystroke_data_model, required=True),
            'model_name': fields.String(required=True, enum=['model1', 'model2', 'model3'],
                                      description='Name of the model to use'),
            'include_confidence': fields.Boolean(default=True, 
                                               description='Whether to include confidence scores'),
            'user_agent': fields.String(description='Browser user agent string')
        })
        
        prediction_response_model = self.api.model('PredictionResponse', {
            'prediction_id': fields.String(description='Unique prediction identifier'),
            'age': fields.Integer(description='Predicted age'),
            'gender': fields.String(description='Predicted gender'),
            'handedness': fields.String(description='Predicted handedness'),
            'class': fields.String(description='Predicted user class'),
            'confidence_scores': fields.Raw(description='Prediction confidence scores'),
            'prediction_time_ms': fields.Float(description='Prediction processing time'),
            'model_used': fields.String(description='Model used for prediction'),
            'data_quality_score': fields.Float(description='Input data quality score'),
            'timestamp': fields.DateTime(description='Prediction timestamp')
        })
        
        @self.prediction_ns.route('/')
        class PredictionResource(Resource):
            @self.prediction_ns.expect(prediction_request_model, validate=True)
            @self.prediction_ns.marshal_with(prediction_response_model)
            @self.prediction_ns.doc('make_prediction')
            @self.prediction_ns.response(200, 'Success')
            @self.prediction_ns.response(400, 'Bad Request')
            @self.prediction_ns.response(429, 'Too Many Requests')
            def post(self):
                """
                Make keystroke analysis prediction
                
                Analyzes keystroke timing patterns to predict user demographics and characteristics.
                """
                try:
                    # Apply rate limiting
                    self.app.limiter.limit("10 per minute")(lambda: None)()
                    
                    # Validate request data
                    request_data = request.get_json()
                    if not request_data:
                        return {'error': 'No JSON data provided'}, 400
                    
                    # Validate using Pydantic
                    is_valid, validated_data = SchemaValidator.validate_with_pydantic(
                        PredictionRequest, request_data
                    )
                    
                    if not is_valid:
                        logger.warning(f"Invalid prediction request: {validated_data}")
                        return {'error': 'Validation failed', 'details': validated_data}, 400
                    
                    # Get model manager from app context
                    from app import model_manager
                    if not model_manager:
                        return {'error': 'ML models not available'}, 503
                    
                    # Extract timing data and make prediction
                    timing_data = validated_data.keystroke_data.timing_data
                    model_name = validated_data.model_name
                    
                    prediction_result = model_manager.predict(model_name, timing_data)
                    
                    if not prediction_result:
                        return {'error': 'Prediction failed'}, 500
                    
                    # Generate prediction ID and prepare response
                    prediction_id = str(uuid.uuid4())
                    
                    # Calculate data quality score
                    data_quality_score = self._calculate_data_quality(timing_data)
                    
                    # Store prediction in database
                    prediction_record = Prediction(
                        id=prediction_id,
                        session_id=validated_data.keystroke_data.session_id,
                        model_name=model_name,
                        age=prediction_result['age'],
                        gender=prediction_result['gender'],
                        handedness=prediction_result['handedness'],
                        user_class=prediction_result['class'],
                        confidence_scores=prediction_result.get('confidence_scores', {}),
                        data_quality_score=data_quality_score,
                        user_agent=validated_data.user_agent
                    )
                    
                    # Store keystroke data
                    keystroke_record = KeystrokeData(
                        session_id=validated_data.keystroke_data.session_id,
                        timing_data=timing_data,
                        text_typed=validated_data.keystroke_data.text_typed,
                        prediction_id=prediction_id
                    )
                    
                    db.session.add(prediction_record)
                    db.session.add(keystroke_record)
                    db.session.commit()
                    
                    # Prepare response
                    response = {
                        'prediction_id': prediction_id,
                        'age': prediction_result['age'],
                        'gender': prediction_result['gender'],
                        'handedness': prediction_result['handedness'],
                        'class': prediction_result['class'],
                        'prediction_time_ms': prediction_result.get('prediction_time_ms', 0),
                        'model_used': model_name,
                        'data_quality_score': data_quality_score,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    if validated_data.include_confidence:
                        response['confidence_scores'] = prediction_result.get('confidence_scores', {})
                    
                    logger.info(f"Prediction completed: {prediction_id}")
                    return response, 200
                    
                except Exception as e:
                    logger.error(f"Prediction error: {str(e)}")
                    return {'error': 'Internal server error'}, 500
        
        @self.prediction_ns.route('/<string:prediction_id>')
        class PredictionDetailResource(Resource):
            @self.prediction_ns.doc('get_prediction')
            @self.prediction_ns.response(200, 'Success')
            @self.prediction_ns.response(404, 'Not Found')
            def get(self, prediction_id):
                """Get details of a specific prediction"""
                try:
                    prediction = Prediction.query.get(prediction_id)
                    if not prediction:
                        return {'error': 'Prediction not found'}, 404
                    
                    return prediction.to_dict(), 200
                    
                except Exception as e:
                    logger.error(f"Error retrieving prediction {prediction_id}: {e}")
                    return {'error': 'Internal server error'}, 500
    
    def _register_contribution_routes(self):
        """Register contribution API routes"""
        
        contribution_request_model = self.api.model('ContributionRequest', {
            'keystroke_data': fields.Nested(self.api.models['KeystrokeData'], required=True),
            'user_data': fields.Raw(required=True, description='User demographic data'),
            'prediction_id': fields.String(description='Associated prediction ID')
        })
        
        @self.contribution_ns.route('/')
        class ContributionResource(Resource):
            @self.contribution_ns.expect(contribution_request_model, validate=True)
            @self.contribution_ns.doc('contribute_data')
            @self.contribution_ns.response(201, 'Created')
            @self.contribution_ns.response(400, 'Bad Request')
            def post(self):
                """
                Contribute keystroke and demographic data for research
                
                Allows users to contribute their keystroke patterns along with demographic
                information to improve model accuracy.
                """
                try:
                    # Apply rate limiting
                    self.app.limiter.limit("5 per minute")(lambda: None)()
                    
                    request_data = request.get_json()
                    if not request_data:
                        return {'error': 'No JSON data provided'}, 400
                    
                    # Validate using Marshmallow for contributions
                    schema = ContributionRequestSchema()
                    is_valid, validated_data = SchemaValidator.validate_with_marshmallow(
                        schema, request_data
                    )
                    
                    if not is_valid:
                        logger.warning(f"Invalid contribution request: {validated_data}")
                        return {'error': 'Validation failed', 'details': validated_data}, 400
                    
                    # Generate contribution ID
                    contribution_id = str(uuid.uuid4())
                    
                    # Store contribution in database
                    contribution_record = UserContribution(
                        id=contribution_id,
                        session_id=validated_data['keystroke_data']['session_id'],
                        timing_data=validated_data['keystroke_data']['timing_data'],
                        text_typed=validated_data['keystroke_data']['text_typed'],
                        user_data=validated_data['user_data'],
                        prediction_id=validated_data.get('prediction_id')
                    )
                    
                    db.session.add(contribution_record)
                    db.session.commit()
                    
                    response = {
                        'contribution_id': contribution_id,
                        'status': 'accepted',
                        'message': 'Thank you for your contribution!',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"Contribution received: {contribution_id}")
                    return response, 201
                    
                except Exception as e:
                    logger.error(f"Contribution error: {str(e)}")
                    return {'error': 'Internal server error'}, 500
    
    def _register_admin_routes(self):
        """Register admin API routes"""
        
        @self.admin_ns.route('/stats')
        class AdminStatsResource(Resource):
            @self.admin_ns.doc('get_admin_stats')
            @self.admin_ns.response(200, 'Success')
            @self.admin_ns.response(401, 'Unauthorized')
            def get(self):
                """Get comprehensive admin statistics"""
                try:
                    # TODO: Add proper authentication
                    
                    # Database statistics
                    db_stats = {
                        'total_predictions': Prediction.query.count(),
                        'total_contributions': UserContribution.query.count(),
                        'total_sessions': KeystrokeData.query.distinct(KeystrokeData.session_id).count(),
                        'recent_activity': self._get_recent_activity()
                    }
                    
                    # Model statistics
                    from app import model_manager
                    model_stats = {}
                    if model_manager:
                        model_stats = model_manager.get_model_stats()
                    
                    # System statistics
                    import psutil
                    process = psutil.Process()
                    system_stats = {
                        'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                        'cpu_usage_percent': process.cpu_percent(),
                        'disk_usage': dict(psutil.disk_usage('/')._asdict())
                    }
                    
                    response = {
                        'database_stats': db_stats,
                        'model_stats': model_stats,
                        'system_stats': system_stats,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    return response, 200
                    
                except Exception as e:
                    logger.error(f"Admin stats error: {str(e)}")
                    return {'error': 'Internal server error'}, 500
        
        @self.admin_ns.route('/data/export')
        class DataExportResource(Resource):
            @self.admin_ns.doc('export_data')
            def get(self):
                """Export data for analysis"""
                # TODO: Implement data export functionality
                return {'message': 'Data export not yet implemented'}, 501
    
    def _register_model_routes(self):
        """Register model information routes"""
        
        @self.model_ns.route('/info')
        class ModelInfoResource(Resource):
            @self.model_ns.doc('get_model_info')
            @self.model_ns.response(200, 'Success')
            def get(self):
                """Get information about available models"""
                try:
                    from app import model_manager
                    if not model_manager:
                        return {'error': 'Model manager not available'}, 503
                    
                    available_models = model_manager.get_available_models()
                    model_stats = model_manager.get_model_stats()
                    
                    response = {
                        'available_models': available_models,
                        'model_statistics': model_stats,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    return response, 200
                    
                except Exception as e:
                    logger.error(f"Model info error: {str(e)}")
                    return {'error': 'Internal server error'}, 500
        
        @self.model_ns.route('/clear-cache')
        class ClearCacheResource(Resource):
            @self.model_ns.doc('clear_model_cache')
            @self.model_ns.response(200, 'Success')
            def post(self):
                """Clear model prediction caches"""
                try:
                    from app import model_manager
                    if not model_manager:
                        return {'error': 'Model manager not available'}, 503
                    
                    model_manager.clear_caches()
                    
                    return {
                        'message': 'Model caches cleared successfully',
                        'timestamp': datetime.utcnow().isoformat()
                    }, 200
                    
                except Exception as e:
                    logger.error(f"Cache clear error: {str(e)}")
                    return {'error': 'Internal server error'}, 500
    
    def _calculate_data_quality(self, timing_data: list) -> float:
        """
        Calculate data quality score for timing data
        
        Args:
            timing_data: List of timing values
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            import numpy as np
            
            if len(timing_data) < 5:
                return 0.0
            
            # Check for reasonable timing values (1ms to 5s)
            valid_timings = [t for t in timing_data if 1000 <= t <= 5000000]
            validity_ratio = len(valid_timings) / len(timing_data)
            
            # Check for variation (not all same values)
            std_dev = np.std(timing_data)
            mean_val = np.mean(timing_data)
            variation_score = min(1.0, std_dev / mean_val) if mean_val > 0 else 0.0
            
            # Check data length (more data = better quality)
            length_score = min(1.0, len(timing_data) / 50.0)  # 50 keystrokes = perfect score
            
            # Combine scores
            quality_score = (validity_ratio * 0.5 + variation_score * 0.3 + length_score * 0.2)
            
            return round(quality_score, 3)
            
        except Exception as e:
            logger.warning(f"Error calculating data quality: {e}")
            return 0.5  # Default moderate quality score
    
    def _get_recent_activity(self) -> list:
        """Get recent system activity"""
        try:
            from datetime import timedelta
            
            # Get recent predictions (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_predictions = Prediction.query.filter(
                Prediction.created_at >= yesterday
            ).order_by(Prediction.created_at.desc()).limit(10).all()
            
            activity = []
            for prediction in recent_predictions:
                activity.append({
                    'type': 'prediction',
                    'id': prediction.id,
                    'model': prediction.model_name,
                    'timestamp': prediction.created_at.isoformat()
                })
            
            return activity
            
        except Exception as e:
            logger.warning(f"Error getting recent activity: {e}")
            return [] 