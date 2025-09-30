#!/usr/bin/env python3
"""
Test script for the Python Hough Circle Detection API
This script tests the API endpoints to ensure they're working correctly
"""

import requests
import base64
import json
import sys
import time
from PIL import Image, ImageDraw
import io

# Configuration
API_BASE_URL = 'http://localhost:5001'
TEST_IMAGE_SIZE = (400, 300)

def create_test_image():
    """Create a simple test image with circles"""
    # Create a white background
    img = Image.new('RGB', TEST_IMAGE_SIZE, 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some test circles
    # Circle 1: Center at (100, 100), radius 30
    draw.ellipse([70, 70, 130, 130], outline='black', width=3)
    
    # Circle 2: Center at (300, 200), radius 25
    draw.ellipse([275, 175, 325, 225], outline='black', width=3)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_detect_circles_endpoint():
    """Test the basic circle detection endpoint"""
    print("🔍 Testing basic circle detection endpoint...")
    
    # Create test image
    test_image = create_test_image()
    
    # Prepare request
    request_data = {
        "image": test_image,
        "min_radius": 20,
        "max_radius": 50,
        "dp": 1,
        "min_dist": 50,
        "param1": 50,
        "param2": 85
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/detect-circles",
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                circles = data.get('circles', [])
                print(f"✅ Circle detection successful: Found {len(circles)} circles")
                for i, circle in enumerate(circles):
                    print(f"   Circle {i+1}: center=({circle['cx']}, {circle['cy']}), radius={circle['r']}")
                return True
            else:
                print(f"❌ Circle detection failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Circle detection failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Circle detection failed: {e}")
        return False

def test_advanced_detect_circles_endpoint():
    """Test the advanced circle detection endpoint"""
    print("🔍 Testing advanced circle detection endpoint...")
    
    # Create test image
    test_image = create_test_image()
    
    # Prepare request with advanced parameters
    request_data = {
        "image": test_image,
        "preprocessing": {
            "blur_kernel": 9,
            "blur_sigma": 2,
            "threshold_type": "none"
        },
        "hough_params": {
            "min_radius": 20,
            "max_radius": 50,
            "dp": 1,
            "min_dist": 50,
            "param1": 50,
            "param2": 85
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/detect-circles-advanced",
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                circles = data.get('circles', [])
                print(f"✅ Advanced circle detection successful: Found {len(circles)} circles")
                for i, circle in enumerate(circles):
                    print(f"   Circle {i+1}: center=({circle['cx']}, {circle['cy']}), radius={circle['r']}")
                return True
            else:
                print(f"❌ Advanced circle detection failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Advanced circle detection failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Advanced circle detection failed: {e}")
        return False

def wait_for_server(max_attempts=10, delay=2):
    """Wait for the server to become available"""
    print(f"⏳ Waiting for server at {API_BASE_URL}...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            print(f"   Attempt {attempt + 1}/{max_attempts} - waiting {delay}s...")
            time.sleep(delay)
    
    print("❌ Server did not become available within the timeout period")
    return False

def main():
    """Run all tests"""
    print("🚀 Starting Python API Tests")
    print("=" * 40)
    
    # Wait for server to be ready
    if not wait_for_server():
        print("❌ Cannot proceed without server")
        sys.exit(1)
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_detect_circles_endpoint,
        test_advanced_detect_circles_endpoint
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Python API is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the server logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()

