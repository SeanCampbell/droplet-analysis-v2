#!/usr/bin/env python3
"""
Test script to debug circle detection issues
"""

import requests
import base64
import numpy as np
from PIL import Image, ImageDraw
import io

def create_test_image():
    """Create a simple test image with circles"""
    # Create a white image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some circles
    draw.ellipse([100, 100, 200, 200], outline='black', width=3)  # Circle 1
    draw.ellipse([300, 200, 450, 350], outline='black', width=3)  # Circle 2
    draw.ellipse([500, 100, 600, 200], outline='black', width=3)  # Circle 3
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def test_api():
    """Test the API with a simple image"""
    print("Creating test image...")
    image_data = create_test_image()
    
    print("Sending request to API...")
    url = "http://localhost:5001/analyze-frame"
    
    payload = {
        "image": image_data,
        "min_radius": 20,
        "max_radius": 200
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API Response:")
            print(f"  Success: {result.get('success', 'N/A')}")
            print(f"  Droplets found: {result.get('dropletsFound', 'N/A')}")
            print(f"  Number of droplets: {len(result.get('droplets', []))}")
            
            if result.get('droplets'):
                print("  Droplets:")
                for i, droplet in enumerate(result['droplets']):
                    print(f"    {i+1}: cx={droplet.get('cx')}, cy={droplet.get('cy')}, r={droplet.get('r')}")
            else:
                print("  No droplets detected")
                
            print(f"  Timestamp: {result.get('timestamp', 'N/A')}")
            print(f"  Scale: {result.get('scale', {}).get('label', 'N/A')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()
