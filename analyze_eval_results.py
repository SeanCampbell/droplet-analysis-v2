#!/usr/bin/env python3
"""
Analysis script for droplet detection evaluation results

This script provides detailed analysis of the evaluation results,
including visualizations and performance breakdowns.
"""

import json
import argparse
import numpy as np
from pathlib import Path

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

def load_results(results_file: str) -> dict:
    """Load evaluation results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_droplet_performance(results: dict) -> dict:
    """Analyze droplet detection performance"""
    frame_results = results['frame_results']
    
    # Extract droplet metrics
    droplet_losses = [frame['droplet_loss'] for frame in frame_results]
    droplet_counts_gt = [frame['gt_droplet_count'] for frame in frame_results]
    droplet_counts_pred = [frame['pred_droplet_count'] for frame in frame_results]
    
    # Calculate per-droplet losses
    per_droplet_losses = []
    for frame in frame_results:
        if frame['gt_droplet_count'] > 0:
            per_droplet_loss = frame['droplet_loss'] / frame['gt_droplet_count']
            per_droplet_losses.append(per_droplet_loss)
    
    return {
        'total_droplet_loss': sum(droplet_losses),
        'avg_droplet_loss': np.mean(droplet_losses),
        'std_droplet_loss': np.std(droplet_losses),
        'min_droplet_loss': min(droplet_losses),
        'max_droplet_loss': max(droplet_losses),
        'avg_per_droplet_loss': np.mean(per_droplet_losses) if per_droplet_losses else 0,
        'std_per_droplet_loss': np.std(per_droplet_losses) if per_droplet_losses else 0,
        'droplet_count_accuracy': sum(1 for gt, pred in zip(droplet_counts_gt, droplet_counts_pred) if gt == pred) / len(frame_results),
        'avg_droplet_count_error': np.mean([abs(gt - pred) for gt, pred in zip(droplet_counts_gt, droplet_counts_pred)])
    }

def analyze_scale_performance(results: dict) -> dict:
    """Analyze scale detection performance"""
    frame_results = results['frame_results']
    
    scale_losses = [frame['scale_loss'] for frame in frame_results]
    
    return {
        'total_scale_loss': sum(scale_losses),
        'avg_scale_loss': np.mean(scale_losses),
        'std_scale_loss': np.std(scale_losses),
        'min_scale_loss': min(scale_losses),
        'max_scale_loss': max(scale_losses)
    }

def analyze_individual_droplets(results: dict) -> dict:
    """Analyze individual droplet detection accuracy"""
    frame_results = results['frame_results']
    
    all_cx_errors = []
    all_cy_errors = []
    all_r_errors = []
    
    for frame in frame_results:
        gt_droplets = frame['ground_truth']['droplets']
        pred_droplets = frame['prediction']['droplets']
        
        if not gt_droplets or not pred_droplets:
            continue
        
        # Simple matching: closest predicted droplet to each ground truth
        used_predictions = set()
        
        for gt_droplet in gt_droplets:
            best_distance = float('inf')
            best_pred = None
            
            for i, pred_droplet in enumerate(pred_droplets):
                if i in used_predictions:
                    continue
                
                # Calculate distance between centers
                distance = np.sqrt((pred_droplet['cx'] - gt_droplet['cx'])**2 + 
                                 (pred_droplet['cy'] - gt_droplet['cy'])**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_pred = pred_droplet
            
            if best_pred:
                all_cx_errors.append(abs(best_pred['cx'] - gt_droplet['cx']))
                all_cy_errors.append(abs(best_pred['cy'] - gt_droplet['cy']))
                all_r_errors.append(abs(best_pred['r'] - gt_droplet['r']))
    
    return {
        'avg_cx_error': np.mean(all_cx_errors) if all_cx_errors else 0,
        'std_cx_error': np.std(all_cx_errors) if all_cx_errors else 0,
        'avg_cy_error': np.mean(all_cy_errors) if all_cy_errors else 0,
        'std_cy_error': np.std(all_cy_errors) if all_cy_errors else 0,
        'avg_r_error': np.mean(all_r_errors) if all_r_errors else 0,
        'std_r_error': np.std(all_r_errors) if all_r_errors else 0,
        'num_matched_droplets': len(all_cx_errors)
    }

def print_detailed_analysis(results: dict):
    """Print detailed analysis of the results"""
    print("🔍 Detailed Performance Analysis")
    print("=" * 50)
    
    # Overall summary
    summary = results['evaluation_summary']
    print(f"📊 Overall Performance:")
    print(f"   🎬 Frames evaluated: {summary['num_frames']}")
    print(f"   🎯 Average total loss: {summary['avg_total_loss']:.2f}")
    print(f"   💧 Average droplet loss: {summary['avg_droplet_loss']:.2f}")
    print(f"   📏 Average scale loss: {summary['avg_scale_loss']:.2f}")
    print()
    
    # Droplet analysis
    droplet_analysis = analyze_droplet_performance(results)
    print(f"💧 Droplet Detection Analysis:")
    print(f"   📈 Average loss per frame: {droplet_analysis['avg_droplet_loss']:.2f} ± {droplet_analysis['std_droplet_loss']:.2f}")
    print(f"   🎯 Average loss per droplet: {droplet_analysis['avg_per_droplet_loss']:.2f} ± {droplet_analysis['std_per_droplet_loss']:.2f}")
    print(f"   🔢 Count accuracy: {droplet_analysis['droplet_count_accuracy']:.1%}")
    print(f"   📊 Average count error: {droplet_analysis['avg_droplet_count_error']:.2f}")
    print(f"   📉 Loss range: {droplet_analysis['min_droplet_loss']:.2f} - {droplet_analysis['max_droplet_loss']:.2f}")
    print()
    
    # Scale analysis
    scale_analysis = analyze_scale_performance(results)
    print(f"📏 Scale Detection Analysis:")
    print(f"   📈 Average loss per frame: {scale_analysis['avg_scale_loss']:.2f} ± {scale_analysis['std_scale_loss']:.2f}")
    print(f"   📉 Loss range: {scale_analysis['min_scale_loss']:.2f} - {scale_analysis['max_scale_loss']:.2f}")
    print()
    
    # Individual droplet analysis
    individual_analysis = analyze_individual_droplets(results)
    print(f"🎯 Individual Droplet Accuracy:")
    print(f"   📍 Center X error: {individual_analysis['avg_cx_error']:.2f} ± {individual_analysis['std_cx_error']:.2f} pixels")
    print(f"   📍 Center Y error: {individual_analysis['avg_cy_error']:.2f} ± {individual_analysis['std_cy_error']:.2f} pixels")
    print(f"   📏 Radius error: {individual_analysis['avg_r_error']:.2f} ± {individual_analysis['std_r_error']:.2f} pixels")
    print(f"   🔢 Matched droplets: {individual_analysis['num_matched_droplets']}")
    print()
    
    # Frame-by-frame breakdown
    print(f"📋 Frame-by-Frame Breakdown:")
    frame_results = results['frame_results']
    for frame in sorted(frame_results, key=lambda x: x['total_loss'], reverse=True):
        print(f"   {frame['frame']}:")
        print(f"      🎯 Total loss: {frame['total_loss']:.2f}")
        print(f"      💧 Droplet loss: {frame['droplet_loss']:.2f}")
        print(f"      📏 Scale loss: {frame['scale_loss']:.2f}")
        print(f"      🔢 Droplets: {frame['pred_droplet_count']}/{frame['gt_droplet_count']}")
        print()

def create_visualizations(results: dict, output_dir: str = "eval_plots"):
    """Create visualization plots of the results"""
    if not HAS_PLOTTING:
        print("⚠️  matplotlib/seaborn not available. Skipping visualizations.")
        print("💡 Install with: pip install matplotlib seaborn")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    frame_results = results['frame_results']
    
    # Extract data
    frame_names = [frame['frame'] for frame in frame_results]
    total_losses = [frame['total_loss'] for frame in frame_results]
    droplet_losses = [frame['droplet_loss'] for frame in frame_results]
    scale_losses = [frame['scale_loss'] for frame in frame_results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Droplet Detection Evaluation Results', fontsize=16)
    
    # Plot 1: Total losses by frame
    axes[0, 0].bar(range(len(frame_names)), total_losses, color='skyblue')
    axes[0, 0].set_title('Total Loss by Frame')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_xticks(range(len(frame_names)))
    axes[0, 0].set_xticklabels([name.replace('frame_', '').replace('.jpeg', '') for name in frame_names], rotation=45)
    
    # Plot 2: Droplet vs Scale losses
    axes[0, 1].scatter(droplet_losses, scale_losses, alpha=0.7, s=100)
    axes[0, 1].set_title('Droplet Loss vs Scale Loss')
    axes[0, 1].set_xlabel('Droplet Loss')
    axes[0, 1].set_ylabel('Scale Loss')
    
    # Add frame labels to points
    for i, frame in enumerate(frame_names):
        axes[0, 1].annotate(frame.replace('frame_', '').replace('.jpeg', ''), 
                           (droplet_losses[i], scale_losses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Loss distribution
    axes[1, 0].hist(total_losses, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Distribution of Total Losses')
    axes[1, 0].set_xlabel('Total Loss')
    axes[1, 0].set_ylabel('Frequency')
    
    # Plot 4: Droplet count comparison
    gt_counts = [frame['gt_droplet_count'] for frame in frame_results]
    pred_counts = [frame['pred_droplet_count'] for frame in frame_results]
    
    x = np.arange(len(frame_names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, gt_counts, width, label='Ground Truth', alpha=0.7)
    axes[1, 1].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
    axes[1, 1].set_title('Droplet Count Comparison')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Number of Droplets')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([name.replace('frame_', '').replace('.jpeg', '') for name in frame_names], rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to {output_dir}/evaluation_summary.png")

def main():
    parser = argparse.ArgumentParser(description='Analyze droplet detection evaluation results')
    parser.add_argument('--results', default='eval_results.json',
                       help='Path to evaluation results JSON file')
    parser.add_argument('--plots', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--output-dir', default='eval_plots',
                       help='Directory for output plots')
    
    args = parser.parse_args()
    
    if not Path(args.results).exists():
        print(f"❌ Error: Results file {args.results} does not exist")
        print("💡 Run eval_droplet_detection.py first to generate results")
        sys.exit(1)
    
    print("📊 Loading evaluation results...")
    results = load_results(args.results)
    
    print_detailed_analysis(results)
    
    if args.plots:
        create_visualizations(results, args.output_dir)

if __name__ == "__main__":
    main()
