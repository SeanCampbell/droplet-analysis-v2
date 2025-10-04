#!/usr/bin/env python3
"""
Test script for droplet detection method toggle functionality

This script tests both v1 (Hough circles) and v2 (random values) detection methods
to verify the toggle functionality works correctly.
"""

import sys
import json
import requests
import base64
import cv2
import numpy as np
from pathlib import Path

# Add the python-server directory to the path
sys.path.append('python-server')

try:
    from app import analyze_frame_comprehensive, detect_circles_hough, detect_circles_v2, detect_circles_v3, detect_circles_v4
except ImportError:
    print("❌ Error: Could not import detection functions from python-server/app.py")
    print("💡 Make sure you're running this script from the project root directory")
    sys.exit(1)

def create_test_image():
    """Create a simple test image with some circles"""
    # Create a white image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw some test circles
    cv2.circle(image, (150, 150), 50, (0, 0, 0), 2)
    cv2.circle(image, (450, 250), 75, (0, 0, 0), 2)
    
    return image

def test_direct_function_calls():
    """Test the detection functions directly"""
    print("🧪 Testing Direct Function Calls")
    print("=" * 40)
    
    # Create test image
    test_image = create_test_image()
    
    # Test v1 (Hough circles)
    print("🔍 Testing v1 (Hough circles)...")
    v1_result = detect_circles_hough(test_image, min_radius=20, max_radius=100)
    print(f"   V1 detected {len(v1_result)} circles")
    for i, circle in enumerate(v1_result):
        if isinstance(circle, dict):
            print(f"   Circle {i+1}: center=({circle['cx']}, {circle['cy']}), radius={circle['r']}")
        else:
            print(f"   Circle {i+1}: center=({circle[0]}, {circle[1]}), radius={circle[2]}")
    
    # Test v2 (optimized template matching)
    print("\n🎲 Testing v2 (optimized template matching)...")
    v2_result = detect_circles_v2(test_image, min_radius=20, max_radius=100)
    print(f"   V2 detected {len(v2_result)} circles")
    for i, circle in enumerate(v2_result):
        if isinstance(circle, dict):
            print(f"   Circle {i+1}: center=({circle['cx']}, {circle['cy']}), radius={circle['r']}")
        else:
            print(f"   Circle {i+1}: center=({circle[0]}, {circle[1]}), radius={circle[2]}")
    
    # Test v3 (fast hybrid detection)
    print("\n🔬 Testing v3 (fast hybrid detection)...")
    v3_result = detect_circles_v3(test_image, min_radius=20, max_radius=100)
    print(f"   V3 detected {len(v3_result)} circles")
    for i, circle in enumerate(v3_result):
        if isinstance(circle, dict):
            print(f"   Circle {i+1}: center=({circle['cx']}, {circle['cy']}), radius={circle['r']}")
        else:
            print(f"   Circle {i+1}: center=({circle[0]}, {circle[1]}), radius={circle[2]}")
    
    # Test v4 (placeholder algorithm)
    print("\n🚀 Testing v4 (placeholder algorithm)...")
    v4_result = detect_circles_v4(test_image, min_radius=20, max_radius=100)
    print(f"   V4 detected {len(v4_result)} circles")
    for i, circle in enumerate(v4_result):
        if isinstance(circle, dict):
            print(f"   Circle {i+1}: center=({circle['cx']}, {circle['cy']}), radius={circle['r']}")
        else:
            print(f"   Circle {i+1}: center=({circle[0]}, {circle[1]}), radius={circle[2]}")
    
    # Test comprehensive analysis
    print("\n🔬 Testing Comprehensive Analysis...")
    v1_comprehensive = analyze_frame_comprehensive(test_image, method="v1")
    v2_comprehensive = analyze_frame_comprehensive(test_image, method="v2")
    v3_comprehensive = analyze_frame_comprehensive(test_image, method="v3")
    v4_comprehensive = analyze_frame_comprehensive(test_image, method="v4")
    
    print(f"   V1 comprehensive: {len(v1_comprehensive.get('droplets', []))} droplets")
    print(f"   V2 comprehensive: {len(v2_comprehensive.get('droplets', []))} droplets")
    print(f"   V3 comprehensive: {len(v3_comprehensive.get('droplets', []))} droplets")
    print(f"   V4 comprehensive: {len(v4_comprehensive.get('droplets', []))} droplets")
    
    return True

def test_api_endpoint():
    """Test the API endpoint with method parameter"""
    print("\n🌐 Testing API Endpoint")
    print("=" * 40)
    
    # Create test image
    test_image = create_test_image()
    
    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', test_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Test data
    test_data = {
        "image": image_base64,
        "min_radius": 20,
        "max_radius": 100
    }
    
    api_url = "http://localhost:5001/api/analyze-frame"
    
    try:
        # Test v1 method
        print("🔍 Testing API with v1 method...")
        v1_data = {**test_data, "method": "v1"}
        v1_response = requests.post(api_url, json=v1_data, timeout=10)
        
        if v1_response.status_code == 200:
            v1_result = v1_response.json()
            droplet_count = len(v1_result.get('droplets', []))
            print(f"   V1 API: {droplet_count} droplets detected")
        else:
            print(f"   V1 API failed: {v1_response.status_code}")
            return False
        
        # Test v2 method
        print("🎲 Testing API with v2 method...")
        v2_data = {**test_data, "method": "v2"}
        v2_response = requests.post(api_url, json=v2_data, timeout=10)
        
        if v2_response.status_code == 200:
            v2_result = v2_response.json()
            droplet_count = len(v2_result.get('droplets', []))
            print(f"   V2 API: {droplet_count} droplets detected")
        else:
            print(f"   V2 API failed: {v2_response.status_code}")
            return False
        
        # Test default method (should be v1)
        print("🔧 Testing API with default method...")
        default_response = requests.post(api_url, json=test_data, timeout=10)
        
        if default_response.status_code == 200:
            default_result = default_response.json()
            droplet_count = len(default_result.get('droplets', []))
            print(f"   Default API: {droplet_count} droplets detected")
        else:
            print(f"   Default API failed: {default_response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("💡 Make sure the server is running: python python-server/app.py")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_evaluation_script():
    """Test the evaluation script with method parameter"""
    print("\n📊 Testing Evaluation Script")
    print("=" * 40)
    
    # Check if evaluation directory exists
    eval_dir = Path("eval/droplet_frames_want/raw")
    if not eval_dir.exists():
        print("❌ Evaluation directory not found, skipping evaluation test")
        return True
    
    # Test v1 method
    print("🔍 Testing evaluation with v1 method...")
    import subprocess
    try:
        result = subprocess.run([
            "python", "eval_droplet_detection.py", 
            "--method", "v1", 
            "--output", "test_v1_results.json"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   V1 evaluation completed successfully")
        else:
            print(f"   V1 evaluation failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   V1 evaluation timed out")
        return False
    
    # Test v2 method
    print("🎲 Testing evaluation with v2 method...")
    try:
        result = subprocess.run([
            "python", "eval_droplet_detection.py", 
            "--method", "v2", 
            "--output", "test_v2_results.json"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   V2 evaluation completed successfully")
        else:
            print(f"   V2 evaluation failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   V2 evaluation timed out")
        return False
    
    # Compare results
    try:
        with open("test_v1_results.json", 'r') as f:
            v1_results = json.load(f)
        with open("test_v2_results.json", 'r') as f:
            v2_results = json.load(f)
        
        v1_avg_loss = v1_results['evaluation_summary']['avg_total_loss']
        v2_avg_loss = v2_results['evaluation_summary']['avg_total_loss']
        
        print(f"   V1 average loss: {v1_avg_loss:.2f}")
        print(f"   V2 average loss: {v2_avg_loss:.2f}")
        print(f"   Loss difference: {abs(v1_avg_loss - v2_avg_loss):.2f}")
        
        # Clean up test files
        Path("test_v1_results.json").unlink(missing_ok=True)
        Path("test_v2_results.json").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"   Error comparing results: {e}")
    
    return True

def main():
    print("🎯 Droplet Detection Method Toggle Test")
    print("=" * 50)
    
    # Test 1: Direct function calls
    if not test_direct_function_calls():
        print("❌ Direct function call test failed")
        sys.exit(1)
    
    # Test 2: API endpoint
    if not test_api_endpoint():
        print("❌ API endpoint test failed")
        sys.exit(1)
    
    # Test 3: Evaluation script
    if not test_evaluation_script():
        print("❌ Evaluation script test failed")
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
    print("✅ Method toggle functionality is working correctly")
    print("\n📋 Usage Examples:")
    print("   - API: POST to /api/analyze-frame with {\"method\": \"v1\"} or {\"method\": \"v2\"}")
    print("   - Evaluation: python eval_droplet_detection.py --method v1")
    print("   - Evaluation: python eval_droplet_detection.py --method v2")

if __name__ == "__main__":
    main()
