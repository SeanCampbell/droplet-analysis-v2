# Droplet Detection Evaluation System

This directory contains a comprehensive evaluation system for the droplet detection algorithm. The system runs the existing detection algorithm on ground truth data and provides detailed performance metrics.

## 📁 Directory Structure

```
eval/
├── droplet_frames_got/          # Previously processed frames
│   ├── processed/               # Processed frames with results
│   └── raw/                     # Raw frames
└── droplet_frames_want/         # Ground truth evaluation data
    ├── processed/               # Processed ground truth
    └── raw/                     # Raw ground truth frames (used for evaluation)
        ├── frame_0000.jpeg      # Frame images
        ├── frame_0000.json      # Ground truth annotations
        ├── frame_0001.jpeg
        ├── frame_0001.json
        └── ...
```

## 🎯 Evaluation Scripts

### 1. `eval_droplet_detection.py`
Main evaluation script that runs the droplet detection algorithm on evaluation frames.

**Usage:**
```bash
# Basic evaluation
python eval_droplet_detection.py

# Verbose output
python eval_droplet_detection.py --verbose

# Custom evaluation directory
python eval_droplet_detection.py --eval-dir eval/droplet_frames_want/raw

# Custom output file
python eval_droplet_detection.py --output my_results.json
```

**Output:**
- `eval_results.json`: Detailed results with per-frame metrics
- Console output with summary statistics

### 2. `analyze_eval_results.py`
Analysis script that provides detailed performance breakdowns and insights.

**Usage:**
```bash
# Basic analysis
python analyze_eval_results.py

# Generate visualizations (requires matplotlib)
python analyze_eval_results.py --plots

# Custom results file
python analyze_eval_results.py --results my_results.json
```

**Output:**
- Detailed performance analysis
- Optional visualization plots in `eval_plots/` directory

### 3. `run_evaluation.py`
Complete pipeline script that runs both evaluation and analysis.

**Usage:**
```bash
# Complete pipeline
python run_evaluation.py

# With visualizations
python run_evaluation.py --plots

# Verbose evaluation
python run_evaluation.py --verbose

# Skip evaluation, only analyze existing results
python run_evaluation.py --skip-eval
```

## 📊 Evaluation Metrics

### Loss Function
The evaluation uses a **sum of squared differences** approach:

**Droplet Loss:**
- For each ground truth droplet, find the closest predicted droplet
- Calculate squared differences for: `cx`, `cy`, `r` (center x, center y, radius)
- Penalize unmatched droplets heavily (10x radius squared)

**Scale Loss:**
- Calculate squared differences for scale bar coordinates: `x1`, `y1`, `x2`, `y2`
- Penalize missing scale detection heavily

**Total Loss:**
- Sum of droplet loss + scale loss

### Performance Metrics

1. **Overall Performance:**
   - Average total loss across all frames
   - Average droplet loss per frame
   - Average scale loss per frame

2. **Droplet Detection:**
   - Count accuracy (percentage of frames with correct droplet count)
   - Average loss per droplet
   - Individual coordinate errors (cx, cy, r)

3. **Scale Detection:**
   - Average scale loss per frame
   - Scale detection success rate

## 📈 Sample Results

```
📊 Overall Performance:
   🎬 Frames evaluated: 4
   🎯 Average total loss: 443430.61
   💧 Average droplet loss: 439037.11
   📏 Average scale loss: 4393.50

💧 Droplet Detection Analysis:
   📈 Average loss per frame: 439037.11 ± 294367.64
   🎯 Average loss per droplet: 219518.56 ± 147183.82
   🔢 Count accuracy: 100.0%
   📊 Average count error: 0.00

🎯 Individual Droplet Accuracy:
   📍 Center X error: 314.14 ± 238.48 pixels
   📍 Center Y error: 163.69 ± 78.34 pixels
   📏 Radius error: 160.05 ± 71.32 pixels
```

## 🔧 Ground Truth Format

The ground truth JSON files should follow this format:

```json
{
  "frame": 0,
  "timestamp": "4.480",
  "timestampFound": true,
  "scaleFound": true,
  "dropletsFound": true,
  "droplets": [
    {
      "cx": 865.42,
      "cy": 861.85,
      "id": 0,
      "r": 314.68
    }
  ],
  "scale": {
    "label": "100 µm",
    "length": 301,
    "x1": 1493,
    "x2": 1794,
    "y1": 1043,
    "y2": 1043
  }
}
```

## 🚀 Quick Start

1. **Prepare evaluation data:**
   - Place frame images (JPEG) in `eval/droplet_frames_want/raw/`
   - Place corresponding ground truth JSON files in the same directory

2. **Run evaluation:**
   ```bash
   python run_evaluation.py --verbose
   ```

3. **View results:**
   - Check console output for summary
   - Open `eval_results.json` for detailed results
   - Run with `--plots` flag for visualizations

## 📋 Troubleshooting

### Common Issues

1. **Import Error:**
   ```
   ❌ Error: Could not import detection functions from python-server/app.py
   ```
   **Solution:** Run the script from the project root directory

2. **No Frames Found:**
   ```
   ❌ No frame files found in eval/droplet_frames_want/raw
   ```
   **Solution:** Ensure JPEG and JSON files are in the correct directory

3. **Missing matplotlib:**
   ```
   ⚠️ matplotlib/seaborn not available. Skipping visualizations.
   ```
   **Solution:** Install with `pip install matplotlib seaborn`

### Performance Tips

- **Large datasets:** Use `--eval-dir` to specify a subset of frames
- **Debugging:** Use `--verbose` flag for detailed per-frame output
- **Visualization:** Install matplotlib for performance plots

## 🔬 Algorithm Details

The evaluation uses the existing `analyze_frame_comprehensive` function from `python-server/app.py`, which includes:

- **Hough Circle Detection:** Multiple parameter combinations for robust detection
- **Post-processing:** Radius refinement using edge detection and contour fitting
- **Scale Detection:** OCR-based scale bar detection
- **Timestamp Detection:** OCR-based timestamp extraction

The evaluation provides insights into how well these components perform on ground truth data, helping to identify areas for improvement.
