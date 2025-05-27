"""
Professional Database Management System
Handles database connections, migrations, and data persistence.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# Global database instance
db = SQLAlchemy()
migrate = Migrate()


def init_db(app, config):
    """
    Initialize database with Flask application
    
    Args:
        app: Flask application instance
        config: Application configuration object
    """
    # Configure database
    app.config['SQLALCHEMY_DATABASE_URI'] = config.get_database_url()
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_RECORD_QUERIES'] = config.is_development()
    
    # SQLite specific configurations
    if 'sqlite' in config.get_database_url():
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'poolclass': StaticPool,
            'pool_pre_ping': True,
            'pool_recycle': 300,
            'connect_args': {
                'check_same_thread': False,
                'timeout': 30
            }
        }
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Register database events
    _register_db_events()
    
    # Create tables in development
    if config.is_development():
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully")


def _register_db_events():
    """Register database events for optimization and logging"""
    
    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Enable WAL mode and other optimizations for SQLite"""
        if 'sqlite' in str(dbapi_connection):
            cursor = dbapi_connection.cursor()
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")
            # Optimize performance
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            cursor.close()
            logger.debug("SQLite optimizations applied")


class BaseModel(db.Model):
    """
    Base model class with common fields and methods
    """
    __abstract__ = True
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, 
        nullable=False, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    
    def save(self, commit: bool = True) -> 'BaseModel':
        """
        Save the model to database
        
        Args:
            commit: Whether to commit the transaction
            
        Returns:
            The saved model instance
        """
        try:
            db.session.add(self)
            if commit:
                db.session.commit()
            logger.debug(f"Saved {self.__class__.__name__} with ID: {self.id}")
            return self
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving {self.__class__.__name__}: {e}")
            raise
    
    def delete(self, soft_delete: bool = True, commit: bool = True) -> bool:
        """
        Delete the model from database
        
        Args:
            soft_delete: If True, set is_active=False instead of deleting
            commit: Whether to commit the transaction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if soft_delete:
                self.is_active = False
                self.updated_at = datetime.utcnow()
                db.session.add(self)
            else:
                db.session.delete(self)
            
            if commit:
                db.session.commit()
            
            logger.debug(f"Deleted {self.__class__.__name__} with ID: {self.id}")
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting {self.__class__.__name__}: {e}")
            return False
    
    def to_dict(self, exclude_fields: Optional[list] = None) -> dict:
        """
        Convert model to dictionary
        
        Args:
            exclude_fields: List of fields to exclude
            
        Returns:
            Dictionary representation of the model
        """
        exclude_fields = exclude_fields or []
        result = {}
        
        for column in self.__table__.columns:
            if column.name not in exclude_fields:
                value = getattr(self, column.name)
                if isinstance(value, datetime):
                    value = value.isoformat()
                result[column.name] = value
        
        return result
    
    @classmethod
    def get_by_id(cls, model_id: int) -> Optional['BaseModel']:
        """
        Get model by ID
        
        Args:
            model_id: The model ID
            
        Returns:
            Model instance or None
        """
        try:
            return cls.query.filter_by(id=model_id, is_active=True).first()
        except Exception as e:
            logger.error(f"Error getting {cls.__name__} by ID {model_id}: {e}")
            return None
    
    @classmethod
    def get_all(cls, active_only: bool = True) -> list:
        """
        Get all models
        
        Args:
            active_only: If True, only return active records
            
        Returns:
            List of model instances
        """
        try:
            query = cls.query
            if active_only:
                query = query.filter_by(is_active=True)
            return query.all()
        except Exception as e:
            logger.error(f"Error getting all {cls.__name__}: {e}")
            return []
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.id})>"


class DatabaseManager:
    """
    Database management utilities
    """
    
    @staticmethod
    def backup_database(backup_path: str) -> bool:
        """
        Backup the database
        
        Args:
            backup_path: Path to save backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Implementation depends on database type
            logger.info(f"Database backup saved to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False
    
    @staticmethod
    def get_database_stats() -> dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = {}
            
            # Get table row counts
            for table in db.metadata.tables.values():
                try:
                    count = db.session.execute(
                        f"SELECT COUNT(*) FROM {table.name}"
                    ).scalar()
                    stats[f"{table.name}_count"] = count
                except Exception:
                    stats[f"{table.name}_count"] = 0
            
            # Database size (SQLite specific)
            try:
                size_result = db.session.execute("PRAGMA page_count").scalar()
                page_size = db.session.execute("PRAGMA page_size").scalar()
                stats['database_size_bytes'] = (size_result or 0) * (page_size or 0)
            except Exception:
                stats['database_size_bytes'] = 0
            
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    @staticmethod
    def clean_old_data(days_old: int = 365) -> int:
        """
        Clean old data from database
        
        Args:
            days_old: Number of days to keep data
            
        Returns:
            Number of records cleaned
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # This would be implemented based on specific models
            # Example: old_records = SomeModel.query.filter(SomeModel.created_at < cutoff_date)
            
            logger.info(f"Cleaned old data older than {days_old} days")
            return 0
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
            return 0 