#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os

# Add the python-server directory to the path
sys.path.append('python-server')

from app import classify_microscope, extract_image_features

def test_microscope_classification():
    """Test microscope classification on all evaluation frames"""
    
    eval_dir = "eval/droplet_frames_want/raw"
    
    # V3 performance by frame (from the analysis above)
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
    
    print("🔍 Microscope Classification Analysis")
    print("=" * 60)
    
    excellent_frames = []
    poor_frames = []
    
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
            
            # Get V3 performance
            v3_loss = v3_performance.get(filename, 0)
            
            # Categorize performance
            if v3_loss < 1000:  # Excellent
                performance_category = "EXCELLENT"
                excellent_frames.append((filename, microscope_type, v3_loss, features))
            elif v3_loss < 100000:  # Good
                performance_category = "GOOD"
            else:  # Poor
                performance_category = "POOR"
                poor_frames.append((filename, microscope_type, v3_loss, features))
            
            print(f"{filename:20} | {microscope_type:12} | {v3_loss:10.2f} | {performance_category:10}")
            print(f"{'':20} | Features: contrast={features['contrast']:.3f}, noise={features['noise']:.3f}, brightness={features['brightness']:.3f}, edge_density={features['edge_density']:.3f}")
            print()
    
    print("\n📊 Analysis Summary")
    print("=" * 60)
    
    print(f"\n🎯 Excellent V3 Performance Frames ({len(excellent_frames)}):")
    for filename, microscope_type, v3_loss, features in excellent_frames:
        print(f"  {filename}: {microscope_type} (loss: {v3_loss:.2f})")
    
    print(f"\n❌ Poor V3 Performance Frames ({len(poor_frames)}):")
    for filename, microscope_type, v3_loss, features in poor_frames:
        print(f"  {filename}: {microscope_type} (loss: {v3_loss:.2f})")
    
    # Analyze classification patterns
    excellent_types = [item[1] for item in excellent_frames]
    poor_types = [item[1] for item in poor_frames]
    
    print(f"\n🔬 Classification Patterns:")
    print(f"  Excellent frames classified as: {set(excellent_types)}")
    print(f"  Poor frames classified as: {set(poor_types)}")
    
    # Check if classification is working correctly
    if set(excellent_types) == set(poor_types):
        print(f"\n⚠️  WARNING: Classification is not distinguishing between excellent and poor frames!")
        print(f"   All frames are being classified as: {set(excellent_types)}")
    else:
        print(f"\n✅ Classification is distinguishing between frame types")
        print(f"   Excellent frames: {set(excellent_types)}")
        print(f"   Poor frames: {set(poor_types)}")

if __name__ == "__main__":
    test_microscope_classification()
