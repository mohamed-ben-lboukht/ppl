#!/bin/bash

# =============================================================================
# Keystroke Analytics - Professional Setup Script
# Version 2.0.0
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="Keystroke Analytics"
PYTHON_VERSION="3.8"
VENV_NAME="venv"
LOG_FILE="setup.log"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "=============================================================="
    echo "🔐 Keystroke Analytics - Professional Setup Script"
    echo "=============================================================="
    echo -e "${NC}"
}

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1" >> "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARN] $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS] $1" >> "$LOG_FILE"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

prompt_yes_no() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

detect_environment() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    
    log "Detected OS: $OS"
}

# =============================================================================
# SYSTEM REQUIREMENTS CHECK
# =============================================================================

check_requirements() {
    log "Checking system requirements..."
    
    # Check Python
    if check_command python3; then
        PYTHON_CMD="python3"
        PYTHON_VER=$(python3 --version | cut -d' ' -f2)
        log "Python found: $PYTHON_VER"
    elif check_command python; then
        PYTHON_CMD="python"
        PYTHON_VER=$(python --version | cut -d' ' -f2)
        log "Python found: $PYTHON_VER"
    else
        error "Python not found! Please install Python $PYTHON_VERSION or higher."
        exit 1
    fi
    
    # Check pip
    if check_command pip3; then
        PIP_CMD="pip3"
    elif check_command pip; then
        PIP_CMD="pip"
    else
        error "pip not found! Please install pip."
        exit 1
    fi
    
    # Check Git (optional)
    if check_command git; then
        log "Git found: $(git --version)"
    else
        warn "Git not found. Some features may not work."
    fi
    
    # Check Docker (optional)
    if check_command docker; then
        log "Docker found: $(docker --version)"
        DOCKER_AVAILABLE=true
    else
        warn "Docker not found. Container deployment will not be available."
        DOCKER_AVAILABLE=false
    fi
    
    success "System requirements check completed"
}

# =============================================================================
# VIRTUAL ENVIRONMENT SETUP
# =============================================================================

setup_virtual_environment() {
    log "Setting up Python virtual environment..."
    
    if [ -d "$VENV_NAME" ]; then
        if prompt_yes_no "Virtual environment already exists. Do you want to recreate it?"; then
            rm -rf "$VENV_NAME"
        else
            log "Using existing virtual environment"
            return 0
        fi
    fi
    
    # Create virtual environment
    $PYTHON_CMD -m venv "$VENV_NAME"
    
    # Activate virtual environment
    if [[ "$OS" == "windows" ]]; then
        source "$VENV_NAME/Scripts/activate"
    else
        source "$VENV_NAME/bin/activate"
    fi
    
    # Upgrade pip
    python -m pip install --upgrade pip setuptools wheel
    
    success "Virtual environment created and activated"
}

# =============================================================================
# DEPENDENCY INSTALLATION
# =============================================================================

install_dependencies() {
    log "Installing Python dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        error "requirements.txt not found!"
        exit 1
    fi
    
    # Install requirements
    $PIP_CMD install -r requirements.txt
    
    # Install additional development dependencies if available
    if [ -f "requirements-dev.txt" ]; then
        if prompt_yes_no "Install development dependencies?"; then
            $PIP_CMD install -r requirements-dev.txt
        fi
    fi
    
    success "Dependencies installed successfully"
}

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

setup_environment() {
    log "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            log "Created .env from env.example"
        else
            error "env.example not found!"
            exit 1
        fi
    else
        if prompt_yes_no ".env already exists. Do you want to backup and recreate it?"; then
            cp .env .env.backup
            cp env.example .env
            log "Backed up existing .env and created new one"
        fi
    fi
    
    # Create logs directory
    mkdir -p logs
    
    # Set default environment values
    if prompt_yes_no "Do you want to use default configuration for development?"; then
        # Update .env with development defaults
        sed -i.bak 's/FLASK_ENV=development/FLASK_ENV=development/' .env 2>/dev/null || true
        sed -i.bak 's/DEBUG=True/DEBUG=True/' .env 2>/dev/null || true
        sed -i.bak 's/SECRET_KEY=your-super-secret-key-change-in-production/SECRET_KEY=dev-secret-key-'$(date +%s)'/' .env 2>/dev/null || true
        rm -f .env.bak
        
        success "Environment configured for development"
    else
        warn "Please edit .env file manually with your configuration"
        echo "Key settings to configure:"
        echo "  - SECRET_KEY: Change to a secure random string"
        echo "  - DATABASE_URL: Configure your database connection"
        echo "  - ADMIN_PASSWORD: Set admin dashboard password"
        echo "  - REDIS_URL: Configure Redis if using caching"
    fi
}

# =============================================================================
# DATABASE SETUP
# =============================================================================

setup_database() {
    log "Setting up database..."
    
    # Check if Flask-Migrate is available
    if python -c "import flask_migrate" 2>/dev/null; then
        # Initialize migration repository if it doesn't exist
        if [ ! -d "migrations" ]; then
            log "Initializing database migration repository..."
            python -c "
from app import create_app
from flask_migrate import init
app = create_app()
with app.app_context():
    init()
"
        fi
        
        # Create migration
        log "Creating database migration..."
        python -c "
from app import create_app
from flask_migrate import migrate
app = create_app()
with app.app_context():
    migrate(message='Initial migration')
" 2>/dev/null || log "Migration already exists or no changes detected"
        
        # Apply migration
        log "Applying database migration..."
        python -c "
from app import create_app
from flask_migrate import upgrade
app = create_app()
with app.app_context():
    upgrade()
"
    else
        # Fallback: direct database creation
        log "Creating database tables directly..."
        python -c "
from app import create_app
from models.database import db
app = create_app()
with app.app_context():
    db.create_all()
"
    fi
    
    success "Database setup completed"
}

# =============================================================================
# MODEL FILES CHECK
# =============================================================================

check_model_files() {
    log "Checking machine learning model files..."
    
    MODEL_DIR="models"
    REQUIRED_MODELS=("model1_weights.pth" "model2_weights.pth" "model3_weights.pth")
    MISSING_MODELS=()
    
    if [ ! -d "$MODEL_DIR" ]; then
        mkdir -p "$MODEL_DIR"
        log "Created models directory"
    fi
    
    for model in "${REQUIRED_MODELS[@]}"; do
        if [ ! -f "$MODEL_DIR/$model" ]; then
            MISSING_MODELS+=("$model")
        fi
    done
    
    if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
        warn "Missing model files:"
        for model in "${MISSING_MODELS[@]}"; do
            echo "  - $MODEL_DIR/$model"
        done
        
        echo
        echo "Note: The application will run in demo mode without trained models."
        echo "To add model files:"
        echo "  1. Place your trained PyTorch model files in the models/ directory"
        echo "  2. Ensure they are named: model1_weights.pth, model2_weights.pth, model3_weights.pth"
        echo "  3. Restart the application"
        
        if ! prompt_yes_no "Continue without model files?"; then
            error "Model files required. Exiting."
            exit 1
        fi
    else
        success "All model files found"
    fi
}

# =============================================================================
# APPLICATION TESTING
# =============================================================================

test_application() {
    log "Testing application startup..."
    
    # Test import
    if python -c "import app; print('Import test passed')" 2>/dev/null; then
        success "Application imports successfully"
    else
        error "Application import failed"
        return 1
    fi
    
    # Test configuration
    if python -c "from config import Config; print('Config test passed')" 2>/dev/null; then
        success "Configuration loads successfully"
    else
        error "Configuration loading failed"
        return 1
    fi
    
    # Test database connection
    if python -c "
from app import create_app
from models.database import db
app = create_app()
with app.app_context():
    db.engine.execute('SELECT 1')
print('Database test passed')
" 2>/dev/null; then
        success "Database connection test passed"
    else
        warn "Database connection test failed (this may be normal if database is not configured)"
    fi
    
    success "Application testing completed"
}

# =============================================================================
# DOCKER SETUP
# =============================================================================

setup_docker() {
    if [ "$DOCKER_AVAILABLE" = true ]; then
        if prompt_yes_no "Do you want to set up Docker environment?"; then
            log "Setting up Docker environment..."
            
            # Build Docker image
            docker build -t keystroke-analytics:latest .
            
            # Create Docker network if it doesn't exist
            docker network create keystroke-network 2>/dev/null || true
            
            success "Docker environment ready"
            
            if prompt_yes_no "Do you want to start the application with Docker Compose?"; then
                docker-compose up -d
                success "Application started with Docker Compose"
                log "Application available at http://localhost:5000"
                return 0
            fi
        fi
    fi
    return 1
}

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

start_application() {
    log "Starting Keystroke Analytics application..."
    
    echo
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Starting $PROJECT_NAME${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
    echo "🌐 Application will be available at: http://localhost:5000"
    echo "📚 API Documentation: http://localhost:5000/api/docs/"
    echo "🔧 Admin Dashboard: http://localhost:5000/admin"
    echo "📊 Health Check: http://localhost:5000/api/health"
    echo
    echo "Press Ctrl+C to stop the application"
    echo
    
    # Start the application
    if [ -f "app.py" ]; then
        python app.py
    else
        error "app.py not found!"
        exit 1
    fi
}

# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

cleanup() {
    log "Performing cleanup..."
    
    # Deactivate virtual environment if active
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate 2>/dev/null || true
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -q, --quick          Quick setup (skip prompts, use defaults)"
    echo "  -d, --docker         Use Docker for deployment"
    echo "  -t, --test           Run tests only"
    echo "  -c, --clean          Clean installation (remove existing venv)"
    echo "  --dev                Development mode setup"
    echo "  --prod               Production mode setup"
    echo
    echo "Examples:"
    echo "  $0                   Interactive setup"
    echo "  $0 --quick           Quick setup with defaults"
    echo "  $0 --docker          Setup and run with Docker"
    echo "  $0 --test            Test application only"
}

main() {
    # Initialize log file
    echo "Setup started at $(date)" > "$LOG_FILE"
    
    # Parse command line arguments
    QUICK_MODE=false
    DOCKER_MODE=false
    TEST_MODE=false
    CLEAN_MODE=false
    ENV_MODE="development"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -q|--quick)
                QUICK_MODE=true
                shift
                ;;
            -d|--docker)
                DOCKER_MODE=true
                shift
                ;;
            -t|--test)
                TEST_MODE=true
                shift
                ;;
            -c|--clean)
                CLEAN_MODE=true
                shift
                ;;
            --dev)
                ENV_MODE="development"
                shift
                ;;
            --prod)
                ENV_MODE="production"
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Start setup
    print_banner
    
    log "Starting setup in $ENV_MODE mode"
    if [ "$QUICK_MODE" = true ]; then
        log "Running in quick mode (minimal prompts)"
    fi
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Detect environment
    detect_environment
    
    # Check system requirements
    check_requirements
    
    # Docker mode
    if [ "$DOCKER_MODE" = true ]; then
        if setup_docker; then
            exit 0
        fi
    fi
    
    # Test mode
    if [ "$TEST_MODE" = true ]; then
        test_application
        exit 0
    fi
    
    # Clean mode
    if [ "$CLEAN_MODE" = true ]; then
        log "Cleaning previous installation..."
        rm -rf "$VENV_NAME"
        rm -f ".env"
        rm -rf "migrations"
        rm -f "*.db"
    fi
    
    # Setup steps
    setup_virtual_environment
    install_dependencies
    setup_environment
    setup_database
    check_model_files
    test_application
    
    success "Setup completed successfully!"
    
    echo
    echo -e "${GREEN}🎉 Setup Summary:${NC}"
    echo "✅ Virtual environment created"
    echo "✅ Dependencies installed"
    echo "✅ Environment configured"
    echo "✅ Database initialized"
    echo "✅ Application tested"
    echo
    
    if [ "$QUICK_MODE" = false ]; then
        if prompt_yes_no "Do you want to start the application now?"; then
            start_application
        else
            echo
            echo -e "${BLUE}To start the application later:${NC}"
            echo "  1. Activate virtual environment: source $VENV_NAME/bin/activate"
            echo "  2. Run the application: python app.py"
            echo
            echo -e "${BLUE}Or use this script:${NC}"
            echo "  ./setup.sh --quick"
        fi
    else
        start_application
    fi
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 