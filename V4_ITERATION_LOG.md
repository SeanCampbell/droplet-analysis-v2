# V4 Algorithm Iteration Log

## Overview
This document tracks the development and iteration of the V4 droplet detection algorithm, comparing performance against V2 and V3 with the new evaluation dataset.

## Baseline Performance (New Dataset - 11 frames)

### Current Performance Benchmarks:
- **V2 (Optimized Template Matching)**: 214,734.15 average total loss
- **V3 (Fast Hybrid Detection)**: 273,717.93 average total loss  
- **V4 (Placeholder)**: 497,789.74 average total loss

### Key Observations:
- V2 is currently the best performer with the new dataset
- V3 performs worse than V2 with the new data (unlike previous dataset where V3 was 50% better)
- V4 placeholder is significantly worse as expected
- New dataset has 11 frames (frames 0000-0003, 0007-0013) vs previous 7 frames

---

## Iteration 1: Enhanced Multi-Scale Hough with Advanced Preprocessing

### Changes Made:
- **Approach**: Multi-scale Hough circle detection with advanced preprocessing
- **Preprocessing**: Gaussian blur, CLAHE, bilateral filtering, morphological operations
- **Detection**: Multiple Hough parameter sets with different scales
- **Post-processing**: Confidence scoring, duplicate removal, radius refinement

### Implementation:
```python
def detect_circles_v4(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
    """
    V4 Detection Algorithm - Enhanced Multi-Scale Hough with Advanced Preprocessing
    
    This algorithm uses advanced preprocessing and multi-scale Hough detection
    to improve accuracy over previous versions.
    """
    height, width = image.shape[:2]
    
    logger.debug(f"V4 Detection: Starting enhanced multi-scale Hough on {width}x{height} image")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Advanced preprocessing pipeline
    preprocessed = create_advanced_preprocessing(gray)
    
    # Multi-scale Hough detection with different parameter sets
    all_circles = []
    
    # Scale 1: Standard parameters
    circles1 = cv2.HoughCircles(
        preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=80,
        param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
    )
    
    if circles1 is not None:
        all_circles.extend(circles1[0])
    
    # Scale 2: More sensitive parameters
    circles2 = cv2.HoughCircles(
        preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
        param1=40, param2=25, minRadius=min_radius, maxRadius=max_radius
    )
    
    if circles2 is not None:
        all_circles.extend(circles2[0])
    
    # Scale 3: High sensitivity for faint circles
    circles3 = cv2.HoughCircles(
        preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=30, param2=20, minRadius=min_radius, maxRadius=max_radius
    )
    
    if circles3 is not None:
        all_circles.extend(circles3[0])
    
    # Convert to droplets format and apply confidence scoring
    droplets = []
    for i, circle in enumerate(all_circles):
        x, y, r = circle
        confidence = calculate_circle_confidence(preprocessed, int(x), int(y), int(r))
        
        if confidence > 0.3:  # Confidence threshold
            droplets.append({
                'cx': int(x),
                'cy': int(y),
                'r': int(r),
                'id': i,
                'confidence': confidence
            })
    
    # Remove duplicates and select best circles
    droplets = remove_duplicate_circles(droplets, min_dist=80)
    droplets = select_best_circles_v4(droplets, max_circles=2)
    
    logger.debug(f"V4 Detection: Found {len(droplets)} droplets using enhanced multi-scale Hough")
    for i, droplet in enumerate(droplets):
        logger.debug(f"  Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}, confidence={droplet.get('confidence', 'N/A')}")
    
    return droplets

def create_advanced_preprocessing(gray):
    """
    Create advanced preprocessed image with multiple enhancement steps
    """
    # 1. Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # 3. Bilateral filtering to preserve edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 4. Morphological operations to enhance circular structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    
    # 5. Edge enhancement
    edges = cv2.Canny(morphed, 50, 150)
    enhanced_edges = cv2.addWeighted(morphed, 0.8, edges, 0.2, 0)
    
    return enhanced_edges

def calculate_circle_confidence(gray, x, y, r):
    """
    Calculate confidence score for a detected circle
    """
    height, width = gray.shape
    
    # Check if circle is within image bounds
    if x - r < 0 or x + r >= width or y - r < 0 or y + r >= height:
        return 0.0
    
    # Create mask for the circle
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Calculate edge strength along circle perimeter
    circle_edges = cv2.Canny(gray, 50, 150)
    edge_pixels = cv2.bitwise_and(circle_edges, mask)
    edge_density = np.sum(edge_pixels > 0) / (2 * np.pi * r)
    
    # Calculate intensity consistency within circle
    circle_region = cv2.bitwise_and(gray, mask)
    mean_intensity = np.mean(circle_region[circle_region > 0])
    intensity_std = np.std(circle_region[circle_region > 0])
    intensity_consistency = 1.0 / (1.0 + intensity_std / (mean_intensity + 1))
    
    # Combine metrics
    confidence = (edge_density * 0.6 + intensity_consistency * 0.4)
    return min(confidence, 1.0)

def remove_duplicate_circles(droplets, min_dist=80):
    """
    Remove duplicate circles that are too close to each other
    """
    if len(droplets) <= 1:
        return droplets
    
    # Sort by confidence (if available) or radius
    droplets.sort(key=lambda d: d.get('confidence', d['r']), reverse=True)
    
    filtered = []
    for droplet in droplets:
        is_duplicate = False
        for existing in filtered:
            dist = np.sqrt((droplet['cx'] - existing['cx'])**2 + (droplet['cy'] - existing['cy'])**2)
            if dist < min_dist:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(droplet)
    
    return filtered

def select_best_circles_v4(droplets, max_circles=2):
    """
    Select the best circles based on confidence and other criteria
    """
    if len(droplets) <= max_circles:
        return droplets
    
    # Sort by confidence and select top circles
    droplets.sort(key=lambda d: d.get('confidence', d['r']), reverse=True)
    return droplets[:max_circles]
```

### Performance Impact:
- **Expected**: Should perform better than V3 but may still be worse than V2 with new data
- **Target**: Aim for < 250,000 average total loss

### Analysis:
- **Strengths**: Advanced preprocessing should help with challenging images
- **Weaknesses**: May be slower due to multiple Hough passes and complex preprocessing
- **Next Steps**: Evaluate performance and potentially simplify if too slow

---

## Performance Summary

| Iteration | Average Total Loss | vs V2 | vs V3 | Notes |
|-----------|-------------------|-------|-------|-------|
| Baseline (Placeholder) | 497,789.74 | 232% worse | 82% worse | Random values |
| Iteration 1 | TBD | TBD | TBD | Enhanced Multi-Scale Hough |

## Key Learnings

### From V2 and V3 Development:
1. **Template matching** (V2) works well for consistent droplet shapes
2. **Hybrid approaches** (V3) can combine strengths of different methods
3. **Preprocessing** is crucial for Hough-based detection
4. **Speed vs accuracy** trade-offs are important for user experience
5. **New dataset** appears more challenging than previous dataset

### V4 Development Strategy:
1. Start with advanced preprocessing and multi-scale detection
2. Focus on edge detection and circle confidence scoring
3. Iterate based on performance results
4. Consider hybrid approaches if single method doesn't work
5. Maintain reasonable speed for user experience

---

## Next Steps

1. **Implement Iteration 1** and evaluate performance
2. **Analyze results** to identify strengths and weaknesses
3. **Iterate** based on performance data
4. **Continue** until V4 significantly outperforms V2 and V3
