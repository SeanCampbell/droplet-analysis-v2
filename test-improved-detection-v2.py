#!/usr/bin/env python3
"""
Test script for the improved detection algorithms
Tests both circle detection (full droplet size) and scale bar detection (avoiding timestamp area)
"""

import requests
import base64
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_image():
    """Create a test image with two droplets, timestamp, and scale bar"""
    # Create a larger image to better test the algorithms
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='lightgray')
    draw = ImageDraw.Draw(img)
    
    # Draw two droplets with dark outer rings and bright cores
    droplet1_center = (200, 300)
    droplet1_outer_radius = 80  # Full droplet size
    droplet1_inner_radius = 50  # Bright core size
    
    droplet2_center = (600, 300)
    droplet2_outer_radius = 75
    droplet2_inner_radius = 45
    
    # Draw outer dark rings (full droplet size)
    draw.ellipse([droplet1_center[0] - droplet1_outer_radius, droplet1_center[1] - droplet1_outer_radius,
                  droplet1_center[0] + droplet1_outer_radius, droplet1_center[1] + droplet1_outer_radius],
                 fill='darkgray', outline='black', width=3)
    
    draw.ellipse([droplet2_center[0] - droplet2_outer_radius, droplet2_center[1] - droplet2_outer_radius,
                  droplet2_center[0] + droplet2_outer_radius, droplet2_center[1] + droplet2_outer_radius],
                 fill='darkgray', outline='black', width=3)
    
    # Draw bright inner cores
    draw.ellipse([droplet1_center[0] - droplet1_inner_radius, droplet1_center[1] - droplet1_inner_radius,
                  droplet1_center[0] + droplet1_inner_radius, droplet1_center[1] + droplet1_inner_radius],
                 fill='white', outline='lightgray', width=2)
    
    draw.ellipse([droplet2_center[0] - droplet2_inner_radius, droplet2_center[1] - droplet2_inner_radius,
                  droplet2_center[0] + droplet2_inner_radius, droplet2_center[1] + droplet2_inner_radius],
                 fill='white', outline='lightgray', width=2)
    
    # Draw timestamp in bottom-left (should be avoided by scale detection)
    timestamp_box = [20, height - 60, 200, height - 20]
    draw.rectangle(timestamp_box, fill='black')
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((30, height - 50), "Live Time: 123.456", fill='white', font=font)
    
    # Draw scale bar in bottom-right (should be detected correctly)
    scale_start = (width - 120, height - 40)
    scale_end = (width - 20, height - 40)
    draw.line([scale_start, scale_end], fill='black', width=3)
    
    # Draw scale bar label
    draw.text((width - 100, height - 70), "50 µm", fill='black', font=font)
    
    # Add some noise/distractors
    # Draw some lines in the timestamp area that should be ignored
    draw.line([(50, height - 100), (150, height - 100)], fill='black', width=2)
    draw.line([(30, height - 80), (180, height - 80)], fill='black', width=2)
    
    return img

def test_detection():
    """Test the improved detection algorithms"""
    print("🧪 Testing Improved Detection Algorithms")
    print("=" * 50)
    
    # Create test image
    test_img = create_test_image()
    
    # Convert to base64
    import io
    buffer = io.BytesIO()
    test_img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Test the API
    api_url = "http://localhost:5001/analyze-frame"
    
    payload = {
        "image": img_base64
    }
    
    try:
        print("📤 Sending test image to API...")
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response received")
            print(f"📊 Analysis Results:")
            print(f"   Droplets detected: {len(result.get('droplets', []))}")
            
            # Check droplet detection
            droplets = result.get('droplets', [])
            if droplets:
                print(f"   🎯 Droplet Analysis:")
                for i, droplet in enumerate(droplets):
                    print(f"      Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
                    # Check if radius is closer to full droplet size (80, 75) vs inner core (50, 45)
                    expected_full_radius = 80 if i == 0 else 75
                    expected_inner_radius = 50 if i == 0 else 45
                    full_diff = abs(droplet['r'] - expected_full_radius)
                    inner_diff = abs(droplet['r'] - expected_inner_radius)
                    
                    if full_diff < inner_diff:
                        print(f"         ✅ GOOD: Detected full droplet size (radius {droplet['r']} vs expected {expected_full_radius})")
                    else:
                        print(f"         ⚠️  PARTIAL: Detected inner core size (radius {droplet['r']} vs expected full {expected_full_radius})")
            else:
                print("   ❌ No droplets detected")
            
            # Check scale detection
            scale = result.get('scale', {})
            if scale:
                print(f"   📏 Scale Analysis:")
                print(f"      Scale bar: ({scale['x1']}, {scale['y1']}) to ({scale['x2']}, {scale['y2']})")
                print(f"      Length: {scale['length']} pixels")
                print(f"      Label: {scale['label']}")
                
                # Check if scale is in the correct area (bottom-right, avoiding timestamp)
                scale_x = (scale['x1'] + scale['x2']) / 2
                scale_y = (scale['y1'] + scale['y2']) / 2
                
                if scale_x > 400 and scale_y > 450:  # Should be in bottom-right
                    print(f"         ✅ GOOD: Scale detected in correct area (x={scale_x:.1f}, y={scale_y:.1f})")
                else:
                    print(f"         ⚠️  WARNING: Scale detected in unexpected area (x={scale_x:.1f}, y={scale_y:.1f})")
            else:
                print("   ❌ No scale bar detected")
            
            # Check timestamp detection
            timestamp = result.get('timestamp', {})
            if timestamp:
                print(f"   ⏰ Timestamp Analysis:")
                if isinstance(timestamp, dict):
                    print(f"      Timestamp: {timestamp.get('value', 'N/A')}")
                    print(f"      Found: {timestamp.get('found', False)}")
                    if timestamp.get('found', False):
                        print(f"         ✅ GOOD: Timestamp detected correctly")
                    else:
                        print(f"         ❌ Timestamp not found")
                else:
                    print(f"      Timestamp: {timestamp}")
            else:
                print("   ❌ No timestamp information")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_detection()
