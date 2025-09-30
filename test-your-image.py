#!/usr/bin/env python3
"""
Test script to debug circle detection on your specific images
"""

import requests
import base64
import json
import sys

def test_image_with_debug(image_path):
    """Test an image file with the debug endpoint"""
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Prepare request
    request_data = {
        "image": image_base64,
        "min_radius": 10,  # Try smaller radius
        "max_radius": 200  # Try larger radius
    }
    
    try:
        print(f"🔍 Testing image: {image_path}")
        print("📡 Sending request to debug endpoint...")
        
        response = requests.post(
            'http://localhost:5001/debug-circles',
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print("✅ Debug analysis completed!")
                print(f"📊 Image info: {data['image_info']}")
                print(f"🎯 Best result: {data['best_count']} circles found")
                
                if data['best_result']:
                    print("🔍 Detected circles:")
                    for circle in data['best_result']:
                        print(f"   Circle {circle['id']+1}: center=({circle['cx']}, {circle['cy']}), radius={circle['r']}")
                else:
                    print("❌ No circles detected with any parameter combination")
                
                print("\n📋 Parameter test results:")
                for test in data['parameter_tests']:
                    status = "✅" if test['count'] > 0 else "❌"
                    print(f"   {status} {test['params']}: {test['count']} circles")
                
                return data['best_result']
            else:
                print(f"❌ Debug failed: {data.get('error')}")
                return None
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def test_image_with_regular(image_path):
    """Test an image file with the regular endpoint"""
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Prepare request with more relaxed parameters
    request_data = {
        "image": image_base64,
        "min_radius": 10,
        "max_radius": 200,
        "dp": 1,
        "min_dist": 20,
        "param1": 20,
        "param2": 30
    }
    
    try:
        print(f"🔍 Testing image with regular endpoint: {image_path}")
        
        response = requests.post(
            'http://localhost:5001/detect-circles',
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                circles = data.get('circles', [])
                print(f"✅ Regular endpoint: {len(circles)} circles found")
                for circle in circles:
                    print(f"   Circle {circle['id']+1}: center=({circle['cx']}, {circle['cy']}), radius={circle['r']}")
                return circles
            else:
                print(f"❌ Regular endpoint failed: {data.get('error')}")
                return None
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test-your-image.py <image_path>")
        print("Example: python3 test-your-image.py droplet_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("🚀 Testing circle detection on your image")
    print("=" * 50)
    
    # Test with debug endpoint first
    debug_result = test_image_with_debug(image_path)
    
    print("\n" + "=" * 50)
    
    # Test with regular endpoint
    regular_result = test_image_with_regular(image_path)
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"Debug endpoint: {len(debug_result) if debug_result else 0} circles")
    print(f"Regular endpoint: {len(regular_result) if regular_result else 0} circles")

if __name__ == "__main__":
    main()
