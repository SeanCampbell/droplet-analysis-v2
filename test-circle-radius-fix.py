#!/usr/bin/env python3
"""
Test script for the improved circle radius detection and scale bar positioning
"""

import requests
import base64
import json
import sys
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image_with_precise_circles():
    """Create a test image with precisely sized circles"""
    # Create a grayscale-like background
    img = Image.new('RGB', (800, 600), (200, 200, 200))  # Light gray background
    draw = ImageDraw.Draw(img)
    
    # Draw two precisely sized circular objects (droplets)
    # Left droplet - radius 60
    left_center = (200, 300)
    left_radius = 60
    draw.ellipse([left_center[0]-left_radius, left_center[1]-left_radius, 
                  left_center[0]+left_radius, left_center[1]+left_radius], 
                 outline='black', width=3)
    draw.ellipse([left_center[0]-left_radius//2, left_center[1]-left_radius//2, 
                  left_center[0]+left_radius//2, left_center[1]+left_radius//2], 
                 fill='white', outline='black', width=2)
    
    # Right droplet - radius 55 (touching the left one)
    right_center = (315, 300)  # Touching the left droplet
    right_radius = 55
    draw.ellipse([right_center[0]-right_radius, right_center[1]-right_radius, 
                  right_center[0]+right_radius, right_center[1]+right_radius], 
                 outline='black', width=3)
    draw.ellipse([right_center[0]-right_radius//2, right_center[1]-right_radius//2, 
                  right_center[0]+right_radius//2, right_center[1]+right_radius//2], 
                 fill='white', outline='black', width=2)
    
    # Draw timestamp in bottom-left (black box with white text)
    timestamp_box = [20, 550, 200, 580]
    draw.rectangle(timestamp_box, fill='black')
    draw.text((30, 555), "Live Time: 271.302", fill='white')
    
    # Draw scale bar in bottom-right (yellow line with label above)
    scale_x1, scale_y1 = 650, 570
    scale_x2, scale_y2 = 750, 570
    # Draw a thick yellow line for the scale bar
    for i in range(5):
        draw.line([(scale_x1, scale_y1+i), (scale_x2, scale_y2+i)], fill=(255, 255, 0), width=1)
    draw.text((690, 550), "50 µm", fill='black')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def test_circle_radius_detection():
    """Test the improved circle radius detection"""
    print("🧪 Testing improved circle radius detection...")
    
    # Create test image with known circle sizes
    test_image = create_test_image_with_precise_circles()
    
    # Prepare request
    request_data = {
        "image": test_image,
        "min_radius": 30,
        "max_radius": 150,
        "dp": 1,
        "min_dist": 20,
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
                
                # Check circle radius accuracy
                expected_radii = [60, 55]  # Expected radii from our test image
                detected_droplets = data.get('droplets', [])
                
                radius_accuracy = []
                for i, droplet in enumerate(detected_droplets):
                    detected_radius = droplet['r']
                    expected_radius = expected_radii[i] if i < len(expected_radii) else 0
                    accuracy = 1.0 - abs(detected_radius - expected_radius) / expected_radius if expected_radius > 0 else 0
                    radius_accuracy.append(accuracy)
                    print(f"     Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']} (expected: {expected_radius}, accuracy: {accuracy:.2f})")
                
                # Check if timestamp was detected correctly
                if data.get('timestampFound') and "271.302" in str(data.get('timestamp')):
                    print("🎉 Timestamp detection working correctly!")
                    timestamp_ok = True
                else:
                    print("⚠️  Timestamp detection needs improvement")
                    timestamp_ok = False
                
                # Check if droplets were detected with reasonable accuracy
                if data.get('dropletsFound') and len(detected_droplets) >= 2:
                    avg_accuracy = sum(radius_accuracy) / len(radius_accuracy) if radius_accuracy else 0
                    if avg_accuracy > 0.7:  # 70% accuracy threshold
                        print(f"🎉 Droplet radius detection working well! (avg accuracy: {avg_accuracy:.2f})")
                        droplets_ok = True
                    else:
                        print(f"⚠️  Droplet radius detection needs improvement (avg accuracy: {avg_accuracy:.2f})")
                        droplets_ok = False
                else:
                    print("⚠️  Droplet detection needs improvement")
                    droplets_ok = False
                
                # Check if scale was detected in the right area (not in timestamp area)
                if data.get('scaleFound'):
                    scale = data.get('scale', {})
                    scale_x = (scale.get('x1', 0) + scale.get('x2', 0)) / 2
                    # Scale should be in the right half of the image
                    if scale_x > 400:  # Right half of 800px image
                        print("🎉 Scale detection in correct area!")
                        scale_ok = True
                    else:
                        print(f"⚠️  Scale detected in wrong area (x={scale_x})")
                        scale_ok = False
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
    print("🚀 Testing Circle Radius and Scale Bar Improvements")
    print("=" * 60)
    
    # Test health check first
    if not test_health_check():
        print("❌ Cannot proceed without healthy server")
        sys.exit(1)
    
    print()
    
    # Test improved detection
    if test_circle_radius_detection():
        print("\n🎉 All improvements working correctly!")
        sys.exit(0)
    else:
        print("\n⚠️  Some improvements need further work")
        sys.exit(1)

if __name__ == "__main__":
    main()
