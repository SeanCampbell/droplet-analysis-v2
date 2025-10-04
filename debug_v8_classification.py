#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os

# Add the python-server directory to the path
sys.path.append('python-server')

from app import classify_microscope, extract_image_features

def debug_v8_classification():
    """Debug V8 classification on specific frames"""
    
    eval_dir = "eval/droplet_frames_want/raw"
    
    # V3 performance by frame
    v3_performance = {
        'frame_0000.jpeg': 50.19,      # Excellent
        'frame_0001.jpeg': 17442.98,   # Good
        'frame_0002.jpeg': 137.76,     # Excellent
        'frame_0003.jpeg': 29.92,      # Excellent
        'frame_0007.jpeg': 183219.49,  # Poor
        'frame_0008.jpeg': 148772.86,  # Poor
        'frame_0009.jpeg': 156891.65,  # Poor
        'frame_0010.jpeg': 155.00,     # Excellent
        'frame_0011.jpeg': 709304.46,  # Very Poor
        'frame_0012.jpeg': 980067.58,  # Very Poor
        'frame_0013.jpeg': 857571.68,  # Very Poor
    }
    
    # V8 performance by frame
    v8_performance = {
        'frame_0000.jpeg': 47.62,
        'frame_0001.jpeg': 17441.10,
        'frame_0002.jpeg': 144.03,
        'frame_0003.jpeg': 15.46,
        'frame_0007.jpeg': 35611.31,
        'frame_0008.jpeg': 17888.72,
        'frame_0009.jpeg': 21334.40,
        'frame_0010.jpeg': 171.00,
        'frame_0011.jpeg': 156.29,
        'frame_0012.jpeg': 298.47,
        'frame_0013.jpeg': 86923.23,
    }
    
    print("🔍 V8 Classification Debug")
    print("=" * 80)
    
    for filename in sorted(os.listdir(eval_dir)):
        if filename.endswith('.jpeg'):
            filepath = os.path.join(eval_dir, filename)
            
            # Load image
            image = cv2.imread(filepath)
            if image is None:
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract features
            features = extract_image_features(gray)
            
            # Classify microscope
            microscope_type = classify_microscope(gray)
            
            # Get performance
            v3_loss = v3_performance.get(filename, 0)
            v8_loss = v8_performance.get(filename, 0)
            
            # Calculate improvement
            if v3_loss > 0:
                improvement = ((v3_loss - v8_loss) / v3_loss) * 100
            else:
                improvement = 0
            
            print(f"{filename:20} | {microscope_type:12} | V3: {v3_loss:10.2f} | V8: {v8_loss:10.2f} | {improvement:+6.1f}%")
            print(f"{'':20} | brightness={features['brightness']:.3f} (threshold=0.658)")
            print()

if __name__ == "__main__":
    debug_v8_classification()
