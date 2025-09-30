#!/usr/bin/env python3
"""
Debug script for Hough circle detection
This script helps diagnose why circles aren't being detected
"""

import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageDraw
import json

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_data))
    
    # Convert to OpenCV format (BGR)
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return opencv_image

def debug_circle_detection(image, min_radius=20, max_radius=150, dp=1, min_dist=50, param1=50, param2=85):
    """
    Debug version of circle detection with detailed output
    """
    print(f"🔍 Debug Circle Detection")
    print(f"Image shape: {image.shape}")
    print(f"Parameters: min_radius={min_radius}, max_radius={max_radius}, dp={dp}")
    print(f"Parameters: min_dist={min_dist}, param1={param1}, param2={param2}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale shape: {gray.shape}")
    print(f"Grayscale range: {gray.min()} - {gray.max()}")
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    print(f"Blurred range: {blurred.min()} - {blurred.max()}")
    
    # Try different parameter combinations
    param_combinations = [
        {"dp": 1, "min_dist": 50, "param1": 50, "param2": 85},
        {"dp": 1, "min_dist": 30, "param1": 30, "param2": 50},
        {"dp": 1, "min_dist": 20, "param1": 20, "param2": 30},
        {"dp": 2, "min_dist": 50, "param1": 50, "param2": 85},
        {"dp": 1, "min_dist": 10, "param1": 10, "param2": 20},
    ]
    
    best_result = None
    best_count = 0
    
    for i, params in enumerate(param_combinations):
        print(f"\n🧪 Testing parameter set {i+1}: {params}")
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=params["dp"],
            minDist=params["min_dist"],
            param1=params["param1"],
            param2=params["param2"],
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"✅ Found {len(circles)} circles with these parameters")
            for j, (x, y, r) in enumerate(circles):
                print(f"   Circle {j+1}: center=({x}, {y}), radius={r}")
            
            if len(circles) > best_count:
                best_count = len(circles)
                best_result = circles
        else:
            print("❌ No circles found with these parameters")
    
    # Return best result
    if best_result is not None:
        detected_circles = []
        for i, (x, y, r) in enumerate(best_result):
            detected_circles.append({
                "id": i,
                "cx": int(x),
                "cy": int(y),
                "r": int(r)
            })
        return detected_circles
    else:
        print("❌ No circles detected with any parameter combination")
        return []

def test_with_sample_image():
    """Test with a sample image"""
    print("🎨 Creating test image...")
    
    # Create a test image
    img = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw clear circles
    draw.ellipse([70, 70, 130, 130], outline='black', width=5)  # radius ~30
    draw.ellipse([275, 175, 325, 225], outline='black', width=5)  # radius ~25
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Convert to OpenCV
    image = base64_to_image(img_base64)
    
    # Test detection
    circles = debug_circle_detection(image)
    
    print(f"\n📊 Final Result: {len(circles)} circles detected")
    for circle in circles:
        print(f"   Circle {circle['id']+1}: center=({circle['cx']}, {circle['cy']}), radius={circle['r']}")
    
    return circles

if __name__ == "__main__":
    test_with_sample_image()
