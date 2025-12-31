#!/usr/bin/env python3
"""
Script to run the Bot Detection API
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import uvicorn
    from src.api.main import app

    print("Starting Bot Detection API...")
    print(f"Model loaded: {app.state.model_data is not None if hasattr(app.state, 'model_data') else 'Check /health endpoint'}")
    print("API will be available at: http://127.0.0.1:8000")
    print("Health check: http://127.0.0.1:8000/health")
    print("API docs: http://127.0.0.1:8000/docs")

    uvicorn.run(app, host="127.0.0.1", port=8000)

except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all dependencies are installed.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting API: {e}")
    sys.exit(1)