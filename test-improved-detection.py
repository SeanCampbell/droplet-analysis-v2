#!/usr/bin/env python3
"""
Test script for the improved detection algorithms
"""

import requests
import base64
import json
import sys
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image_with_touching_droplets():
    """Create a test image with touching droplets like in the descriptions"""
    # Create a grayscale-like background
    img = Image.new('RGB', (800, 600), (200, 200, 200))  # Light gray background
    draw = ImageDraw.Draw(img)
    
    # Draw two touching circular objects (droplets)
    # Left droplet
    left_center = (200, 300)
    left_radius = 80
    draw.ellipse([left_center[0]-left_radius, left_center[1]-left_radius, 
                  left_center[0]+left_radius, left_center[1]+left_radius], 
                 outline='black', width=2)
    draw.ellipse([left_center[0]-left_radius//2, left_center[1]-left_radius//2, 
                  left_center[0]+left_radius//2, left_center[1]+left_radius//2], 
                 fill='white', outline='black', width=1)
    
    # Right droplet (touching the left one)
    right_center = (320, 300)  # Touching the left droplet
    right_radius = 75
    draw.ellipse([right_center[0]-right_radius, right_center[1]-right_radius, 
                  right_center[0]+right_radius, right_center[1]+right_radius], 
                 outline='black', width=2)
    draw.ellipse([right_center[0]-right_radius//2, right_center[1]-right_radius//2, 
                  right_center[0]+right_radius//2, right_center[1]+right_radius//2], 
                 fill='white', outline='black', width=1)
    
    # Draw timestamp in bottom-left (black box with white text)
    timestamp_box = [20, 550, 200, 580]
    draw.rectangle(timestamp_box, fill='black')
    draw.text((30, 555), "Live Time: 218.017", fill='white')
    
    # Draw scale bar in bottom-right (black line with label above)
    scale_x1, scale_y1 = 650, 570
    scale_x2, scale_y2 = 750, 570
    draw.line([(scale_x1, scale_y1), (scale_x2, scale_y2)], fill='black', width=3)
    draw.text((690, 550), "50 µm", fill='black')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def test_improved_detection():
    """Test the improved detection algorithms"""
    print("🧪 Testing improved detection algorithms...")
    
    # Create test image
    test_image = create_test_image_with_touching_droplets()
    
    # Prepare request
    request_data = {
        "image": test_image,
        "min_radius": 30,
        "max_radius": 150,
        "dp": 1,
        "min_dist": 20,  # Smaller min_dist for touching droplets
        "param1": 30,
        "param2": 50
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
                print("✅ Analysis successful!")
                print(f"📊 Results:")
                print(f"   Timestamp: '{data.get('timestamp')}' (Found: {data.get('timestampFound')})")
                print(f"   Scale Found: {data.get('scaleFound')}")
                if data.get('scale'):
                    scale = data['scale']
                    print(f"   Scale: ({scale.get('x1')}, {scale.get('y1')}) to ({scale.get('x2')}, {scale.get('y2')}) - '{scale.get('label')}'")
                print(f"   Droplets Found: {data.get('dropletsFound')} ({len(data.get('droplets', []))} detected)")
                for i, droplet in enumerate(data.get('droplets', [])):
                    print(f"     Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
                
                # Check if timestamp was detected correctly
                if data.get('timestampFound') and "218.017" in str(data.get('timestamp')):
                    print("🎉 Timestamp detection working correctly!")
                    timestamp_ok = True
                else:
                    print("⚠️  Timestamp detection needs improvement")
                    timestamp_ok = False
                
                # Check if droplets were detected
                if data.get('dropletsFound') and len(data.get('droplets', [])) >= 2:
                    print("🎉 Droplet detection working correctly!")
                    droplets_ok = True
                else:
                    print("⚠️  Droplet detection needs improvement")
                    droplets_ok = False
                
                # Check if scale was detected
                if data.get('scaleFound'):
                    print("🎉 Scale detection working correctly!")
                    scale_ok = True
                else:
                    print("⚠️  Scale detection needs improvement")
                    scale_ok = False
                
                return timestamp_ok and droplets_ok and scale_ok
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
    print("🚀 Testing Improved Detection Algorithms")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("❌ Cannot proceed without healthy server")
        sys.exit(1)
    
    print()
    
    # Test improved detection
    if test_improved_detection():
        print("\n🎉 All detection algorithms working correctly!")
        sys.exit(0)
    else:
        print("\n⚠️  Some detection algorithms need further improvement")
        sys.exit(1)

if __name__ == "__main__":
    main()
