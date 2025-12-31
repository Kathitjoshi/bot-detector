# Bot Detection System

A production-ready machine learning system for detecting automated social media accounts using behavioral analysis, cryptographic content fingerprinting, and ensemble learning techniques.

## Table of Contents

- [Overview](#overview)
- [Technical Stack](#technical-stack)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)

## Overview

This system identifies bot accounts on social media platforms by analyzing behavioral patterns, account characteristics, and content similarity. It achieves 90-95% accuracy using ensemble machine learning models with comprehensive anti-overfitting measures.

### Key Features

- Multiple ML models with hyperparameter optimization (Logistic Regression, Random Forest, XGBoost)
- Cryptographic content fingerprinting using SHA-256 for bot network detection
- Real-time predictions via REST API
- Batch processing for high-throughput scenarios
- Comprehensive test suite with 36 unit and integration tests
- Automated CI/CD pipeline with GitHub Actions
- Docker containerization for consistent deployment
- Production-ready with proper error handling and logging

### What Makes This Project Unique

- Detects coordinated bot networks using cryptographic hashing
- Implements proper anti-overfitting measures (cross-validation, SMOTE, regularization)
- Complete production pipeline from data processing to deployment
- Comprehensive testing and continuous integration
- Easy deployment to multiple cloud platforms

## Technical Stack

### Core Technologies
- Python 3.9+
- scikit-learn 1.5.2 (Machine Learning)
- XGBoost 2.1.3 (Gradient Boosting)
- FastAPI 0.115.6 (REST API)
- Pandas 2.2.3 (Data Processing)
- NumPy 2.x (Numerical Computing)

### Development Tools
- pytest (Testing Framework)
- Docker (Containerization)
- GitHub Actions (CI/CD)
- Black & Flake8 (Code Quality)

### Deployment Platforms
- Docker containerized

## Project Architecture

### System Components

```
┌─────────────────┐
│  Data Layer     │  - Synthetic data generation
│                 │  - CSV loading and cleaning
└────────┬────────┘
         │
┌────────▼────────┐
│ Feature Layer   │  - 15+ engineered features
│                 │  - SHA-256 content hashing
│                 │  - Behavioral analysis
└────────┬────────┘
         │
┌────────▼────────┐
│  Model Layer    │  - 3 ML models
│                 │  - Hyperparameter tuning
│                 │  - Cross-validation
└────────┬────────┘
         │
┌────────▼────────┐
│   API Layer     │  - FastAPI REST endpoints
│                 │  - Real-time predictions
│                 │  - Batch processing
└─────────────────┘
```

### Data Flow

1. Raw user data (12 features)
2. Feature engineering (26 features)
3. Model prediction (bot probability)
4. Response with confidence score

### Key Integration Points

- Model trained offline, loaded at API startup
- Same FeatureEngineer used in training and inference
- Docker packages entire application stack
- CI/CD automates testing, training, and deployment

## Installation

### Prerequisites

- Python 3.9, 3.10, 3.11, or 3.13
- pip package manager
- Git
- Docker (optional, for containerized deployment)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bot-detector.git
cd bot-detector

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import sklearn, xgboost, fastapi; print('All dependencies installed')"
```

## Quick Start

### Step 1: Train the Model (2-3 minutes)

```bash
python train.py
```

This creates synthetic data, trains models, and saves the best one to `models/saved_models/best_model.joblib`.

### Step 2: Run Tests (Optional but Recommended)

```bash
# Windows PowerShell
$env:PYTHONPATH = "$PWD"
pytest 

or

python -m pytest --cov=src --cov-report=html --cov-report=term  # with coverage report in root dir in vscode

start htmlcov/index.html  # to see coverage report(>80%)

# Mac/Linux
export PYTHONPATH=$PWD
pytest
```

Expected: 36 tests passing

### Step 3: Start the API Server

```bash
python src/api/main.py 
or
python run_api.py
```

The API will be available at http://localhost:8000

### Step 4: Test the API

Open your browser and go to http://localhost:8000/docs for interactive API documentation.

Or test via command line:

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"followers_count":50,"following_count":3000,"tweet_count":10000,"account_age_days":90,"listed_count":1,"verified":0,"default_profile":1,"default_profile_image":1,"geo_enabled":0,"description_length":10,"avg_tweets_per_day":100.0,"avg_retweet_ratio":0.9}' `
  -UseBasicParsing
```

```bash
# Mac/Linux/Git Bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"followers_count":50,"following_count":3000,"tweet_count":10000,"account_age_days":90,"listed_count":1,"verified":0,"default_profile":1,"default_profile_image":1,"geo_enabled":0,"description_length":10,"avg_tweets_per_day":100.0,"avg_retweet_ratio":0.9}'
```

Expected response:
```json
{
  "is_bot": true,
  "confidence": 0.9999,
  "suspicion_score": 1.0,
  "message": "This account shows bot-like behavior"
}
```

## Usage Guide

### Training a New Model

```bash
python train.py
```

This script:
1. Generates 10,000 **synthetic** user accounts (30% bots, 70% humans)
2. Engineers 26 features including cryptographic fingerprints
3. Trains 3 models with GridSearchCV hyperparameter tuning
4. Performs 5-fold stratified cross-validation
5. Applies SMOTE for class balancing
6. Evaluates on held-out test set
7. Saves best model based on F1 score

Training output includes:
- Model comparison metrics
- Best hyperparameters
- Test set performance (Accuracy, Precision, Recall, F1, AUC-ROC)
- Feature importance (for tree-based models)

### Using the API

#### Single Prediction

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "followers_count": 150,
    "following_count": 2000,
    "tweet_count": 5000,
    "account_age_days": 180,
    "listed_count": 2,
    "verified": 0,
    "default_profile": 1,
    "default_profile_image": 1,
    "geo_enabled": 0,
    "description_length": 20,
    "avg_tweets_per_day": 50.5,
    "avg_retweet_ratio": 0.85
}

response = requests.post(url, json=data)
print(response.json())
```

#### Batch Prediction (up to 100 users)

```python
batch_data = {
    "users": [
        {
            "followers_count": 50,
            "following_count": 3000,
            # ... other fields
        },
        {
            "followers_count": 500,
            "following_count": 300,
            # ... other fields
        }
    ]
}

response = requests.post("http://localhost:8000/batch_predict", json=batch_data)
print(response.json())
```

### If Using with Custom Data

Replace the synthetic data generation in `train.py` with real data:

```python
# Instead of:
df = processor.create_sample_dataset(n_samples=10000, bot_ratio=0.3)

# Use:
df = processor.load_data("path/to/your/data.csv")
```

Ensure your CSV has these columns:
- followers_count, following_count, tweet_count
- account_age_days, listed_count, verified
- default_profile, default_profile_image, geo_enabled
- description_length, avg_tweets_per_day, avg_retweet_ratio
- label (0 for human, 1 for bot)

## API Documentation

### Base URL

Local: `http://localhost:8000`

### Endpoints

#### GET /
Root endpoint with API information.

Response:
```json
{
  "message": "Bot Detection API",
  "version": "1.0.0",
  "status": "active",
  "model_loaded": true
}
```

#### GET /health
Health check endpoint for monitoring.

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### POST /predict
Predict if a single user is a bot.

Request Body:
```json
{
  "followers_count": 150,
  "following_count": 2000,
  "tweet_count": 5000,
  "account_age_days": 180,
  "listed_count": 2,
  "verified": 0,
  "default_profile": 1,
  "default_profile_image": 1,
  "geo_enabled": 0,
  "description_length": 20,
  "avg_tweets_per_day": 50.5,
  "avg_retweet_ratio": 0.85
}
```

Response:
```json
{
  "is_bot": true,
  "confidence": 0.87,
  "suspicion_score": 0.72,
  "message": "This account shows bot-like behavior"
}
```

#### POST /batch_predict
Predict bot status for multiple users (max 100).

Request Body:
```json
{
  "users": [
    { /* user 1 data */ },
    { /* user 2 data */ }
  ]
}
```

Response:
```json
{
  "predictions": [
    {
      "is_bot": true,
      "confidence": 0.87,
      "suspicion_score": 0.72,
      "message": "Bot-like behavior detected"
    }
  ],
  "total_users": 2,
  "bot_count": 1,
  "human_count": 1
}
```

#### GET /model_info
Get information about the loaded model. Note: model_info differs from time to time for users

Response:
```json
{
  "model_name": "logisticregression",
  "model_type": "LogisticRegression",
  "feature_importance_available": true
}
```

### Interactive Documentation

FastAPI provides automatic interactive documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

### Running Tests

```bash
# Set Python path (required for imports)
# Windows PowerShell:
$env:PYTHONPATH = "$PWD"

# Mac/Linux:
export PYTHONPATH=$PWD

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -x               # Unit tests only
python -m pytest tests/integration/test_api.py -v --tb=short # Integration tests only
python -m pytest tests/unit/test_model.py # Specific test file
```

### Test Structure

- Unit Tests (26 tests): Test individual components
  - Data processing: Data loading, cleaning, synthetic generation
  - Feature engineering: Feature creation, cryptographic functions
  - Model training: Training pipeline, predictions, metrics

- Integration Tests (10 tests): Test end-to-end functionality
  - API endpoints: All HTTP endpoints
  - Full pipeline: Complete workflow from data to prediction
  - Model consistency: Reproducibility of predictions

### Test Coverage

Target: >80% code coverage
Current: 36/36 tests passing (API tests require model to be loaded)

### Continuous Testing

Tests automatically run on every push via GitHub Actions CI/CD pipeline.

## Deployment

### Docker Deployment

#### Build Image

```bash
docker build -t bot-detector .
```

#### Run Container

```bash
docker run -d -p 8000:8000 --name bot-detector bot-detector
```

#### Check Logs

```bash
docker logs bot-detector
```

#### Stop Container

```bash
docker stop bot-detector
docker rm bot-detector
```

#### Using Docker Compose

```bash
# Start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```



## CI/CD Pipeline

### GitHub Actions Workflow

The project includes automated CI/CD via `.github/workflows/ci-cd.yml`:

#### Stages

1. Test Stage (runs on every push/PR)
   - Installs dependencies
   - Runs pytest with coverage
   - Lints code with flake8
   - Checks formatting with black

2. Train Stage (runs on main branch push only)
   - Trains model with `python train.py`
   - Uploads model as artifact (retained 30 days)

3. Build Stage (runs on main branch push only)
   - Downloads trained model artifact
   - Builds Docker image
   - Tests Docker container health

4. Deploy Stage (runs on main branch push only)
   - Currently placeholder for deployment
   - Can be extended for automatic cloud deployment

#### Triggering CI/CD

```bash
# Push to trigger full pipeline
git push origin main

# Create PR to trigger tests only
git checkout -b feature-branch
git push origin feature-branch
# Create PR on GitHub
```



### Setting Up CI/CD

No additional setup required. The workflow automatically runs when code is pushed to GitHub.

## Project Structure

```
bot-detector/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions CI/CD pipeline
├── data/
│   ├── raw/                    # Raw input data (gitignored)
│   └── processed/              # Processed datasets (gitignored)
├── models/
│   ├── saved_models/           # Trained models (gitignored except .gitkeep)
│   └── training/               # Training artifacts (gitignored)
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── data_loader.py      # Data loading and preprocessing
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   └── features.py         # Feature engineering + crypto
│   ├── model/
│   │   ├── __init__.py
│   │   └── train.py            # Model training with anti-overfitting
│   └── api/
│       ├── __init__.py
│       └── main.py             # FastAPI application
├── tests/
│   ├── __init__.py
│   ├── unit/                   # Unit tests (26 tests)
│   │   ├── test_data_processing.py
│   │   ├── test_feature_engineering.py
│   │   └── test_model.py
│   └── integration/            # Integration tests (10 tests)
│       └── test_api.py
├── logs/                       # Application logs (gitignored)
├── .gitignore                  # Git ignore rules
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── train.py                    # Main training script
└── README.md                   # This file
```

## Model Performance

### Training Performance

Typical metrics on test set (synthetic data):

- Accuracy: 95-100%
- Precision: 95-100%
- Recall: 95-100%
- F1 Score: 95-100%
- AUC-ROC: 99-100%

Note: High accuracy is due to synthetic data with clear separation. Real-world data would show 85-95% accuracy with proper noise and edge cases.

### API Performance

- Single prediction: <100ms response time
- Batch prediction (100 users): <500ms response time
- Throughput: ~1000 requests/minute
- Memory usage: ~500MB with model loaded

### Anti-Overfitting Measures

1. Train/Validation/Test split (60/20/20)
2. Stratified 5-fold cross-validation
3. SMOTE applied only to training data
4. Regularization (L2 for Logistic Regression, reg_alpha/reg_lambda for XGBoost)
5. Early stopping for iterative models
6. Hyperparameter tuning with validation set
7. Model selection based on validation performance

### Feature Importance

Top features for bot detection:
1. Follower/following ratio
2. Tweet frequency (tweets per day)
3. Account age
4. Retweet ratio
5. Profile completeness
6. Cryptographic similarity scores

## Development Workflow

### Local Development Cycle

1. Make code changes
2. Run tests: `pytest`
3. Train model: `python train.py`
4. Test API: `python src/api/main.py` or `python run_api.py`
5. Commit changes
6. Push to GitHub (triggers CI/CD)

### Adding New Features

1. Add feature in `src/feature_engineering/features.py`
2. Update unit tests in `tests/unit/test_feature_engineering.py`
3. Retrain model: `python train.py`
4. Verify tests pass: `pytest`
5. Commit and push

### Modifying Models

1. Edit `src/model/train.py`
2. Update hyperparameter grid
3. Retrain: `python train.py`
4. Check performance metrics
5. Update tests if needed
6. Commit and push

### Best Practices

- Always run tests before committing
- Keep model files out of git (use .gitignore)
- Document significant changes in commit messages
- Use feature branches for major changes
- Review CI/CD results before merging to main

## Troubleshooting

### Common Issues and Solutions



#### Issue: Model Not Found Error

```bash
# Solution: Train model first
python train.py
```

#### Issue: API Returns 503 Service Unavailable

Cause: Model not loaded
Solution: Ensure `models/saved_models/best_model.joblib` exists and run `python train.py`



#### Issue: Docker Build Fails

```bash
# Solution: Ensure all files are present
git status
# Make sure requirements.txt, Dockerfile, and src/ are committed
docker build -t bot-detector .
```

#### Issue: High Memory Usage

Solution: Use model quantization or reduce batch size in API

#### Issue: Slow Predictions

Solutions:
1. Use smaller model (Logistic Regression instead of XGBoost)
2. Reduce number of features
3. Implement caching for repeated requests
4. Scale horizontally with multiple instances

### Getting Help

1. Check this README thoroughly
2. Review QUICKSTART.md for setup issues
3. Check DEPLOYMENT.md for deployment issues
4. Review test output for specific errors
5. Check GitHub Actions logs for CI/CD issues
6. Open an issue on GitHub with error details

### Logs Location

- Application logs: `logs/` directory
- API logs: stdout when running `python src/api/main.py`
- Test logs: pytest output
- CI/CD logs: GitHub Actions → Workflow run → Logs

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit with clear messages
7. Push to your fork
8. Create a Pull Request with description of changes

### Code Style

- Follow PEP 8 style guide
- Use Black for formatting: `black src/`
- Use Flake8 for linting: `flake8 src/`
- Add docstrings to all functions and classes
- Write unit tests for new features

## License

This project is licensed under the MIT License.

## Acknowledgments

This project demonstrates machine learning engineering best practices including data processing, feature engineering, model training with anti-overfitting measures, API development, comprehensive testing, CI/CD automation, and deployment strategies.

Developed as a portfolio project showcasing skills in:
- Machine Learning (scikit-learn, XGBoost)
- Data Science (Pandas, NumPy)
- API Development (FastAPI)
- Software Engineering (Testing, CI/CD, Docker)
- Cryptography (SHA-256 hashing)
- DevOps (Docker, GitHub Actions, Cloud Deployment)

## Contact

For questions, issues, or suggestions, please open an issue on GitHub or submit a pull request.

## Version History

- v1.0.0 (2025-01-30): Initial release
  - Complete ML pipeline
  - FastAPI REST API
  - Docker deployment
  - CI/CD with GitHub Actions
  - Comprehensive test suite
