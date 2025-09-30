#!/usr/bin/env python3
"""
Test script for the comprehensive frame analysis endpoint
"""

import requests
import base64
import json
import sys
from PIL import Image, ImageDraw
import io

def create_test_image_with_all_elements():
    """Create a test image with droplets, scale bar, and timestamp"""
    # Create a white background
    img = Image.new('RGB', (600, 400), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw test circles (droplets)
    draw.ellipse([100, 100, 200, 200], outline='black', width=3)  # radius ~50
    draw.ellipse([350, 150, 450, 250], outline='black', width=3)  # radius ~50
    
    # Draw scale bar (horizontal line)
    draw.line([(450, 350), (550, 350)], fill='black', width=3)
    draw.text((500, 360), "50 µm", fill='black')
    
    # Draw timestamp
    draw.text((20, 20), "Live Time: 0:01:23.456", fill='black')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def test_comprehensive_analysis():
    """Test the comprehensive analysis endpoint"""
    print("🧪 Testing comprehensive frame analysis...")
    
    # Create test image
    test_image = create_test_image_with_all_elements()
    
    # Prepare request
    request_data = {
        "image": test_image,
        "min_radius": 20,
        "max_radius": 100,
        "dp": 1,
        "min_dist": 50,
        "param1": 50,
        "param2": 85
    }
    
    try:
        response = requests.post(
            'http://localhost:5001/analyze-frame',
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print("✅ Comprehensive analysis successful!")
                print(f"📊 Results:")
                print(f"   Timestamp: {data.get('timestamp')} (Found: {data.get('timestampFound')})")
                print(f"   Scale Found: {data.get('scaleFound')}")
                if data.get('scale'):
                    scale = data['scale']
                    print(f"   Scale: ({scale.get('x1')}, {scale.get('y1')}) to ({scale.get('x2')}, {scale.get('y2')}) - {scale.get('label')}")
                print(f"   Droplets Found: {data.get('dropletsFound')} ({len(data.get('droplets', []))} detected)")
                for i, droplet in enumerate(data.get('droplets', [])):
                    print(f"     Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
                
                return True
            else:
                print(f"❌ Analysis failed: {data.get('error')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get('http://localhost:5001/health', timeout=5)
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

def main():
    print("🚀 Testing Comprehensive Frame Analysis API")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("❌ Cannot proceed without healthy server")
        sys.exit(1)
    
    print()
    
    # Test comprehensive analysis
    if test_comprehensive_analysis():
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
