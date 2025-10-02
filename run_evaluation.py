#!/usr/bin/env python3
"""
Complete evaluation pipeline for droplet detection

This script runs the full evaluation pipeline:
1. Runs droplet detection on evaluation frames
2. Analyzes the results
3. Optionally generates visualizations
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🚀 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run complete droplet detection evaluation')
    parser.add_argument('--eval-dir', default='eval/droplet_frames_want/raw',
                       help='Directory containing evaluation frames')
    parser.add_argument('--output', default='eval_results.json',
                       help='Output file for evaluation results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during evaluation')
    parser.add_argument('--plots', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation and only run analysis')
    
    args = parser.parse_args()
    
    print("🎯 Droplet Detection Evaluation Pipeline")
    print("=" * 50)
    
    # Check if evaluation directory exists
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"❌ Error: Evaluation directory {eval_dir} does not exist")
        sys.exit(1)
    
    # Step 1: Run evaluation (unless skipped)
    if not args.skip_eval:
        eval_cmd = f"python eval_droplet_detection.py --eval-dir {args.eval_dir} --output {args.output}"
        if args.verbose:
            eval_cmd += " --verbose"
        
        if not run_command(eval_cmd, "Running droplet detection evaluation"):
            sys.exit(1)
    
    # Check if results file exists
    if not Path(args.output).exists():
        print(f"❌ Error: Results file {args.output} does not exist")
        print("💡 Run evaluation first or check the output path")
        sys.exit(1)
    
    # Step 2: Run analysis
    analysis_cmd = f"python analyze_eval_results.py --results {args.output}"
    if args.plots:
        analysis_cmd += " --plots"
    
    if not run_command(analysis_cmd, "Analyzing evaluation results"):
        sys.exit(1)
    
    print("\n🎉 Evaluation pipeline completed successfully!")
    print(f"📊 Results saved to: {args.output}")
    if args.plots:
        print("📈 Visualizations saved to: eval_plots/")
    
    # Print summary
    print("\n📋 Quick Summary:")
    print("   - Run 'python eval_droplet_detection.py --help' for evaluation options")
    print("   - Run 'python analyze_eval_results.py --help' for analysis options")
    print("   - Use --plots flag to generate visualizations (requires matplotlib)")

if __name__ == "__main__":
    main()
