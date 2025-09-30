#!/usr/bin/env python3
"""
Test script specifically for timestamp detection improvements
"""

import requests
import base64
import json
import sys
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image_like_yours():
    """Create a test image similar to the described format"""
    # Create a grayscale-like background
    img = Image.new('RGB', (800, 600), (200, 200, 200))  # Light gray background
    draw = ImageDraw.Draw(img)
    
    # Draw two circular objects (droplets)
    # Larger droplet on the left
    draw.ellipse([100, 150, 300, 350], outline='black', width=2)  # Outer circle
    draw.ellipse([150, 200, 250, 300], fill='white', outline='black', width=1)  # Inner bright area
    
    # Smaller droplet on the right
    draw.ellipse([500, 200, 650, 350], outline='black', width=2)  # Outer circle
    draw.ellipse([530, 230, 620, 320], fill='white', outline='black', width=1)  # Inner bright area
    
    # Draw timestamp in bottom-left (black box with white text)
    timestamp_box = [20, 550, 200, 580]
    draw.rectangle(timestamp_box, fill='black')
    draw.text((30, 555), "Live Time: 40.456", fill='white')
    
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

def test_timestamp_detection():
    """Test the improved timestamp detection"""
    print("🧪 Testing improved timestamp detection...")
    
    # Create test image
    test_image = create_test_image_like_yours()
    
    # Prepare request
    request_data = {
        "image": test_image,
        "min_radius": 30,
        "max_radius": 150,
        "dp": 1,
        "min_dist": 100,
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
                if data.get('timestampFound') and "40.456" in str(data.get('timestamp')):
                    print("🎉 Timestamp detection working correctly!")
                    return True
                else:
                    print("⚠️  Timestamp detection needs improvement")
                    return False
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
    print("🚀 Testing Improved Timestamp Detection")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("❌ Cannot proceed without healthy server")
        sys.exit(1)
    
    print()
    
    # Test timestamp detection
    if test_timestamp_detection():
        print("\n🎉 Timestamp detection test passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Timestamp detection needs further improvement")
        sys.exit(1)

if __name__ == "__main__":
    main()
