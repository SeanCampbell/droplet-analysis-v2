#!/usr/bin/env python3
"""
Startup script for the Hough Circle Detection Python server
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import cv2
        import numpy
        import PIL
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_server():
    """Start the Flask server"""
    if not check_dependencies():
        sys.exit(1)
    
    print("Starting Hough Circle Detection server...")
    print("Server will be available at: http://localhost:5001")
    print("Health check: http://localhost:5001/health")
    print("Press Ctrl+C to stop the server")
    
    try:
        from app import app
        app.run(host='0.0.0.0', port=5001, debug=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()

