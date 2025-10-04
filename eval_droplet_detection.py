#!/usr/bin/env python3
"""
Evaluation script for droplet detection algorithm

This script runs the existing droplet detection algorithm on frames in the eval directory
and compares the results with ground truth data to calculate evaluation metrics.
"""

import os
import json
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse

# Add the python-server directory to the path so we can import the detection functions
sys.path.append('python-server')

try:
    from app import analyze_frame_comprehensive, base64_to_image
except ImportError:
    print("❌ Error: Could not import detection functions from python-server/app.py")
    print("💡 Make sure you're running this script from the project root directory")
    sys.exit(1)

def load_ground_truth(json_path: str) -> Dict[str, Any]:
    """Load ground truth data from JSON file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading ground truth from {json_path}: {e}")
        return {}

def run_detection_on_frame(image_path: str, method: str = "v1") -> Dict[str, Any]:
    """Run the droplet detection algorithm on a single frame"""
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image {image_path}")
            return {}
        
        # Run the comprehensive analysis
        result = analyze_frame_comprehensive(image, min_radius=20, max_radius=500, method=method)
        
        return result
    except Exception as e:
        print(f"❌ Error running detection on {image_path}: {e}")
        return {}

def calculate_droplet_loss(predicted_droplets: List[Dict], ground_truth_droplets: List[Dict]) -> float:
    """
    Calculate the loss for droplet detection.
    
    For each droplet, we calculate the sum of squared differences for:
    - cx (center x)
    - cy (center y) 
    - r (radius)
    
    We match droplets by finding the closest predicted droplet to each ground truth droplet.
    """
    if not ground_truth_droplets:
        # If no ground truth droplets, penalize all predicted droplets
        return sum(droplet['r']**2 for droplet in predicted_droplets) * 10  # Heavy penalty
    
    if not predicted_droplets:
        # If no predicted droplets, penalize all ground truth droplets
        return sum(droplet['r']**2 for droplet in ground_truth_droplets) * 10  # Heavy penalty
    
    total_loss = 0.0
    used_predictions = set()
    
    # For each ground truth droplet, find the closest predicted droplet
    for gt_droplet in ground_truth_droplets:
        best_loss = float('inf')
        best_pred_idx = -1
        
        for pred_idx, pred_droplet in enumerate(predicted_droplets):
            if pred_idx in used_predictions:
                continue
                
            # Calculate squared differences for each trait
            cx_diff = (pred_droplet['cx'] - gt_droplet['cx']) ** 2
            cy_diff = (pred_droplet['cy'] - gt_droplet['cy']) ** 2
            r_diff = (pred_droplet['r'] - gt_droplet['r']) ** 2
            
            # Sum of squared differences
            loss = cx_diff + cy_diff + r_diff
            
            if loss < best_loss:
                best_loss = loss
                best_pred_idx = pred_idx
        
        if best_pred_idx != -1:
            used_predictions.add(best_pred_idx)
            total_loss += best_loss
        else:
            # No prediction found for this ground truth droplet
            total_loss += gt_droplet['r'] ** 2 * 10  # Heavy penalty
    
    # Penalize unmatched predicted droplets
    for pred_idx, pred_droplet in enumerate(predicted_droplets):
        if pred_idx not in used_predictions:
            total_loss += pred_droplet['r'] ** 2 * 10  # Heavy penalty
    
    return total_loss

def calculate_scale_loss(predicted_scale: Dict, ground_truth_scale: Dict) -> float:
    """
    Calculate the loss for scale bar detection.
    
    We calculate the sum of squared differences for:
    - x1, y1, x2, y2 (scale bar coordinates)
    """
    if not ground_truth_scale:
        return 0.0  # No ground truth scale, no penalty
    
    if not predicted_scale:
        # No predicted scale, heavy penalty
        return (ground_truth_scale.get('x1', 0) ** 2 + 
                ground_truth_scale.get('y1', 0) ** 2 + 
                ground_truth_scale.get('x2', 0) ** 2 + 
                ground_truth_scale.get('y2', 0) ** 2) * 10
    
    # Calculate squared differences for each coordinate
    x1_diff = (predicted_scale.get('x1', 0) - ground_truth_scale.get('x1', 0)) ** 2
    y1_diff = (predicted_scale.get('y1', 0) - ground_truth_scale.get('y1', 0)) ** 2
    x2_diff = (predicted_scale.get('x2', 0) - ground_truth_scale.get('x2', 0)) ** 2
    y2_diff = (predicted_scale.get('y2', 0) - ground_truth_scale.get('y2', 0)) ** 2
    
    return x1_diff + y1_diff + x2_diff + y2_diff

def evaluate_frame(image_path: str, ground_truth_path: str, method: str = "v1") -> Dict[str, Any]:
    """Evaluate a single frame"""
    print(f"🔍 Evaluating {os.path.basename(image_path)} with method {method}...")
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)
    if not ground_truth:
        return {"error": "Could not load ground truth"}
    
    # Run detection
    prediction = run_detection_on_frame(image_path, method)
    if not prediction:
        return {"error": "Could not run detection"}
    
    # Calculate losses
    droplet_loss = calculate_droplet_loss(
        prediction.get('droplets', []),
        ground_truth.get('droplets', [])
    )
    
    scale_loss = calculate_scale_loss(
        prediction.get('scale', {}),
        ground_truth.get('scale', {})
    )
    
    total_loss = droplet_loss + scale_loss
    
    # Calculate metrics
    gt_droplet_count = len(ground_truth.get('droplets', []))
    pred_droplet_count = len(prediction.get('droplets', []))
    
    return {
        "frame": os.path.basename(image_path),
        "droplet_loss": droplet_loss,
        "scale_loss": scale_loss,
        "total_loss": total_loss,
        "gt_droplet_count": gt_droplet_count,
        "pred_droplet_count": pred_droplet_count,
        "droplet_count_diff": abs(gt_droplet_count - pred_droplet_count),
        "ground_truth": ground_truth,
        "prediction": prediction
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate droplet detection algorithm')
    parser.add_argument('--eval-dir', default='eval/droplet_frames_want/raw', 
                       help='Directory containing evaluation frames')
    parser.add_argument('--output', default='eval_results.json',
                       help='Output file for evaluation results')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results for each frame')
    parser.add_argument('--method', default='v1', choices=['v1', 'v2', 'v3'],
                       help='Detection method: v1 (Hough circles), v2 (optimized template matching), or v3 (placeholder)')
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"❌ Error: Evaluation directory {eval_dir} does not exist")
        sys.exit(1)
    
    print("🎯 Droplet Detection Evaluation")
    print("=" * 50)
    print(f"📁 Evaluation directory: {eval_dir}")
    print(f"📊 Output file: {args.output}")
    print(f"🔧 Detection method: {args.method}")
    print()
    
    # Find all frame files
    frame_files = []
    for file_path in eval_dir.glob("frame_*.jpeg"):
        json_path = file_path.with_suffix('.json')
        if json_path.exists():
            frame_files.append((file_path, json_path))
    
    if not frame_files:
        print(f"❌ No frame files found in {eval_dir}")
        sys.exit(1)
    
    print(f"📸 Found {len(frame_files)} frames to evaluate")
    print()
    
    # Evaluate each frame
    results = []
    total_droplet_loss = 0.0
    total_scale_loss = 0.0
    total_loss = 0.0
    total_droplet_count_diff = 0
    
    for i, (image_path, json_path) in enumerate(frame_files):
        result = evaluate_frame(str(image_path), str(json_path), args.method)
        
        if "error" in result:
            print(f"❌ {result['frame']}: {result['error']}")
            continue
        
        results.append(result)
        total_droplet_loss += result['droplet_loss']
        total_scale_loss += result['scale_loss']
        total_loss += result['total_loss']
        total_droplet_count_diff += result['droplet_count_diff']
        
        if args.verbose:
            print(f"  📊 Droplet loss: {result['droplet_loss']:.2f}")
            print(f"  📏 Scale loss: {result['scale_loss']:.2f}")
            print(f"  🎯 Total loss: {result['total_loss']:.2f}")
            print(f"  🔢 Droplet count: {result['pred_droplet_count']}/{result['gt_droplet_count']}")
            print()
    
    # Calculate summary statistics
    num_frames = len(results)
    if num_frames == 0:
        print("❌ No frames were successfully evaluated")
        sys.exit(1)
    
    avg_droplet_loss = total_droplet_loss / num_frames
    avg_scale_loss = total_scale_loss / num_frames
    avg_total_loss = total_loss / num_frames
    avg_droplet_count_diff = total_droplet_count_diff / num_frames
    
    # Print summary
    print("📊 Evaluation Summary")
    print("=" * 30)
    print(f"🎬 Frames evaluated: {num_frames}")
    print(f"🎯 Average total loss: {avg_total_loss:.2f}")
    print(f"💧 Average droplet loss: {avg_droplet_loss:.2f}")
    print(f"📏 Average scale loss: {avg_scale_loss:.2f}")
    print(f"🔢 Average droplet count difference: {avg_droplet_count_diff:.2f}")
    print()
    
    # Save detailed results
    summary = {
        "evaluation_summary": {
            "num_frames": num_frames,
            "total_droplet_loss": total_droplet_loss,
            "total_scale_loss": total_scale_loss,
            "total_loss": total_loss,
            "avg_droplet_loss": avg_droplet_loss,
            "avg_scale_loss": avg_scale_loss,
            "avg_total_loss": avg_total_loss,
            "total_droplet_count_diff": total_droplet_count_diff,
            "avg_droplet_count_diff": avg_droplet_count_diff
        },
        "frame_results": results
    }
    
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Detailed results saved to {args.output}")
    
    # Print top 3 worst performing frames
    if len(results) >= 3:
        print("\n🔍 Top 3 worst performing frames:")
        worst_frames = sorted(results, key=lambda x: x['total_loss'], reverse=True)[:3]
        for i, frame in enumerate(worst_frames, 1):
            print(f"  {i}. {frame['frame']}: loss = {frame['total_loss']:.2f}")

if __name__ == "__main__":
    main()
