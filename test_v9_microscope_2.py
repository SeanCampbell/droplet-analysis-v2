#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os
import json

# Add the python-server directory to the path
sys.path.append('python-server')

from app import detect_circles_v9, detect_circles_v8, classify_microscope, extract_image_features

def test_v9_microscope_2():
    """Test V9 performance specifically on microscope_2 frames (7-9)"""
    
    eval_dir = "eval/droplet_frames_want/raw"
    
    # Microscope_2 frames
    microscope_2_frames = ['frame_0007.jpeg', 'frame_0008.jpeg', 'frame_0009.jpeg']
    
    # Ground truth will be loaded per frame
    
    print("🔍 V9 Microscope_2 Performance Test (Frames 7-9)")
    print("=" * 60)
    
    v8_losses = []
    v9_losses = []
    
    for filename in microscope_2_frames:
        filepath = os.path.join(eval_dir, filename)
        
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            continue
            
        # Load ground truth for this frame
        gt_filepath = filepath.replace('.jpeg', '.json')
        with open(gt_filepath, 'r') as f:
            gt_data = json.load(f)
        gt_droplets = gt_data.get('droplets', [])
        gt_scale = gt_data.get('scale', {}).get('length', 1.0)
        
        # Test V8
        v8_droplets = detect_circles_v8(image, 20, 500, 1, 50, 50, 85)
        
        # Test V9
        v9_droplets = detect_circles_v9(image, 20, 500, 1, 50, 50, 85)
        
        # Calculate losses
        def calculate_loss(detected_droplets, gt_droplets, gt_scale):
            if len(detected_droplets) == 0:
                return float('inf')
            
            total_loss = 0
            
            # Droplet loss
            for i, gt_droplet in enumerate(gt_droplets):
                if i < len(detected_droplets):
                    detected = detected_droplets[i]
                    # Calculate distance between centers
                    center_loss = np.sqrt((detected['cx'] - gt_droplet['cx'])**2 + 
                                        (detected['cy'] - gt_droplet['cy'])**2)
                    # Calculate radius difference
                    radius_loss = abs(detected['r'] - gt_droplet['r'])
                    total_loss += center_loss + radius_loss
                else:
                    # Missing droplet penalty
                    total_loss += 1000
            
            # Scale loss (simplified)
            scale_loss = abs(gt_scale - 1.0) * 1000
            
            return total_loss + scale_loss
        
        v8_loss = calculate_loss(v8_droplets, gt_droplets, gt_scale)
        v9_loss = calculate_loss(v9_droplets, gt_droplets, gt_scale)
        
        v8_losses.append(v8_loss)
        v9_losses.append(v9_loss)
        
        improvement = ((v8_loss - v9_loss) / v8_loss) * 100 if v8_loss > 0 else 0
        
        print(f"{filename:20} | V8: {v8_loss:10.2f} | V9: {v9_loss:10.2f} | {improvement:+6.1f}%")
    
    print(f"\n📊 Microscope_2 Summary:")
    print(f"V8 Average Loss: {np.mean(v8_losses):.2f}")
    print(f"V9 Average Loss: {np.mean(v9_losses):.2f}")
    print(f"Overall Improvement: {((np.mean(v8_losses) - np.mean(v9_losses)) / np.mean(v8_losses)) * 100:.1f}%")

if __name__ == "__main__":
    test_v9_microscope_2()
