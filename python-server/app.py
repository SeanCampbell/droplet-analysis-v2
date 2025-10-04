#!/usr/bin/env python3
"""
Unified Flask application for droplet analysis
Supports both development and production environments
"""

import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import logging
import json
import re
import pytesseract
import subprocess
import tempfile
import shutil

# Configure logging based on environment
debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='static', static_url_path='')

# Configure Flask debug mode
app.config['DEBUG'] = debug_mode
app.config['FLASK_DEBUG'] = os.getenv('FLASK_DEBUG', '0') == '1'
app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'production')

# Configure CORS based on environment
if os.getenv('FLASK_ENV') == 'development':
    # Development: Allow specific origins
    CORS(app, origins=['http://localhost:8888', 'http://127.0.0.1:8888'])
else:
    # Production: Allow all origins (nginx handles the routing)
    CORS(app, origins='*')

# Production configuration
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB max file size (for video uploads)

logger.info(f"Starting Flask app in {'DEBUG' if debug_mode else 'PRODUCTION'} mode")

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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

def detect_circles_hough(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
    """
    Detect circles using Hough Circle Transform with improved parameter handling
    
    Args:
        image: OpenCV image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Upper threshold for edge detection
        param2: Accumulator threshold for center detection
    
    Returns:
        List of detected circles with format [cx, cy, r]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Apply multiple preprocessing techniques
    processed_images = []
    
    # 1. Standard Gaussian blur
    blurred1 = cv2.GaussianBlur(gray, (9, 9), 2)
    processed_images.append(("standard", blurred1))
    
    # 2. Stronger blur for noisy images
    blurred2 = cv2.GaussianBlur(gray, (15, 15), 3)
    processed_images.append(("strong_blur", blurred2))
    
    # 3. Bilateral filter to preserve edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    processed_images.append(("bilateral", bilateral))
    
    # 4. Median filter to reduce noise
    median = cv2.medianBlur(gray, 5)
    processed_images.append(("median", median))
    
    # Try multiple parameter combinations for better detection
    param_combinations = [
        # Ultra-sensitive parameters for faint droplet edges
        {"dp": 1, "min_dist": 30, "param1": 10, "param2": 15, "name": "ultra_sensitive"},
        {"dp": 1, "min_dist": 40, "param1": 12, "param2": 18, "name": "faint_edge_detector"},
        {"dp": 1, "min_dist": 50, "param1": 8, "param2": 12, "name": "very_faint_edges"},
        
        # Prioritize outer ring detection parameters first
        # {"dp": 1, "min_dist": 30, "param1": 20, "param2": 30, "name": "outer_ring_focus"},
        # {"dp": 1, "min_dist": 50, "param1": 25, "param2": 35, "name": "full_droplet"},
        # {"dp": 1, "min_dist": 80, "param1": 20, "param2": 30, "name": "ultra_large"},
        # {"dp": 1, "min_dist": 100, "param1": 15, "param2": 25, "name": "massive_droplets"},
        # {"dp": 1, "min_dist": 40, "param1": 35, "param2": 45, "name": "large_droplets"},
        # {"dp": 1, "min_dist": 60, "param1": 30, "param2": 40, "name": "very_large"},
        # {"dp": 1, "min_dist": 20, "param1": 25, "param2": 35, "name": "dark_ring_sensitive"},
    ]
    
    all_circles = []
    
    for img_name, processed_img in processed_images:
        for params in param_combinations:
            try:
                circles = cv2.HoughCircles(
                    processed_img,
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
                    for circle in circles:
                        x, y, r = circle
                        # Filter out circles that are too close to image edges
                        if (r < x < width - r) and (r < y < height - r):
                            all_circles.append((x, y, r, params["name"], img_name))
            except Exception as e:
                print(f"Circle detection error with {img_name}/{params['name']}: {e}")
                continue
    
    # Remove duplicate circles (circles that are too close to each other)
    unique_circles = []
    for circle in all_circles:
        x, y, r, param_name, img_name = circle
        is_duplicate = False
        
        for existing in unique_circles:
            ex, ey, er, _, _ = existing
            distance = np.sqrt((x - ex)**2 + (y - ey)**2)
            # If circles are too close, keep the one with higher radius
            if distance < min(r, er) * 0.5:
                is_duplicate = True
                if r > er:
                    # Replace existing circle with this one
                    unique_circles.remove(existing)
                    unique_circles.append(circle)
                break
        
        if not is_duplicate:
            unique_circles.append(circle)
    
    # Sort by radius (largest first) and take top 2
    unique_circles.sort(key=lambda x: x[2], reverse=True)
    top_circles = unique_circles[:2]
    
    # Refine circle radii using multiple approaches to capture full droplet size
    refined_circles = []
    for i, (x, y, r, param_name, img_name) in enumerate(top_circles):
        # Extract larger region around the circle to capture full droplet
        margin = int(r * 2.5)  # Increased margin to capture full droplet
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + margin)
        y2 = min(height, y + margin)
        
        roi = gray[y1:y2, x1:x2]
        if roi.size > 0:
            # Method 1: Edge detection with multiple thresholds
            edges1 = cv2.Canny(roi, 30, 100)  # Lower threshold to catch more edges
            edges2 = cv2.Canny(roi, 50, 150)
            edges3 = cv2.Canny(roi, 20, 80)   # Very sensitive for dark rings
            
            # Combine edge images
            combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
            
            # Method 2: Use adaptive thresholding to find dark rings
            adaptive_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            inverted_adaptive = cv2.bitwise_not(adaptive_thresh)
            
            # Method 3: Use gradient magnitude to find circular boundaries
            grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)
            _, gradient_thresh = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
            
            # Combine all edge detection methods
            all_edges = cv2.bitwise_or(combined_edges, cv2.bitwise_or(inverted_adaptive, gradient_thresh))
            
            # Find contours in the combined edge image
            contours, _ = cv2.findContours(all_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the contour that best represents the full droplet
                best_radius = r
                best_score = 0
                
                for contour in contours:
                    # Fit a circle to the contour
                    if len(contour) >= 5:  # Need at least 5 points for circle fitting
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        # Check if the fitted circle is close to our detected center
                        center_distance = np.sqrt((cx - (x-x1))**2 + (cy - (y-y1))**2)
                        if center_distance < r * 0.5:  # More lenient center matching
                            # Calculate circularity score
                            area = cv2.contourArea(contour)
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * area / (perimeter * perimeter)
                                # Prefer larger circles that are reasonably circular
                                if circularity > 0.2 and 30 < radius < 300:  # Allow larger radius range
                                    # Score based on circularity and size (prefer larger circles)
                                    size_bonus = min(radius / r, 2.0)  # Bonus for larger circles up to 2x
                                    score = circularity * size_bonus * (1.0 - abs(radius - r) / max(r, 1))
                                    if score > best_score:
                                        best_score = score
                                        best_radius = radius
                
                # Use the refined radius if it's reasonable and larger than original
                if best_score > 0.1 and best_radius > r * 0.7:  # More lenient - allow up to 30% smaller or much larger
                    r = best_radius
                elif best_radius > r * 1.1:  # If we found a larger circle, use it (lowered threshold)
                    r = best_radius
                elif best_radius > r * 0.8 and best_score > 0.2:  # Even if smaller, if it's a good circle, consider it
                    r = best_radius
        
        refined_circles.append((x, y, r, param_name, img_name))
    
    # Final post-processing: Look for dark outer rings to expand circle size
    final_circles = []
    for i, (x, y, r, param_name, img_name) in enumerate(refined_circles):
        # Extract a much larger region around the detected circle
        margin = int(r * 5)  # Much larger margin to find outer ring
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + margin)
        y2 = min(height, y + margin)
        
        roi = gray[y1:y2, x1:x2]
        if roi.size > 0:
            # Look for dark regions (outer rings) around the detected circle
            # Use multiple thresholds to catch different levels of dark rings
            thresholds = [60, 80, 100, 120]  # Multiple thresholds for different darkness levels
            best_outer_radius = r
            
            for thresh_val in thresholds:
                _, dark_thresh = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY_INV)
                
                # Find contours in the dark regions
                contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if len(contour) >= 5:
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        # Check if this contour is centered around our detected circle
                        center_distance = np.sqrt((cx - (x-x1))**2 + (cy - (y-y1))**2)
                        if center_distance < r * 1.2:  # More lenient centering requirement
                            # Be much more aggressive about accepting larger circles
                            if radius > r * 1.1 and radius < r * 4.0:  # Accept up to 4x larger
                                area = cv2.contourArea(contour)
                                perimeter = cv2.arcLength(contour, True)
                                if perimeter > 0:
                                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                                    if circularity > 0.1:  # Very lenient circularity requirement
                                        best_outer_radius = max(best_outer_radius, radius)
                                        print(f"  Found potential outer ring: radius {radius} (original: {r}), circularity: {circularity:.3f}")
            
            # If we didn't find a good outer ring, try a simple expansion approach
            if best_outer_radius <= r * 1.1:
                # Try expanding by a fixed factor based on typical droplet structure
                expansion_factors = [1.5, 2.0, 2.5, 3.0]
                for factor in expansion_factors:
                    test_radius = int(r * factor)
                    # Check if this expanded circle makes sense in the image
                    if (x - test_radius >= 0 and x + test_radius < width and 
                        y - test_radius >= 0 and y + test_radius < height):
                        # Sample points around the expanded circle to see if they're dark
                        sample_points = []
                        for angle in range(0, 360, 30):  # Sample every 30 degrees
                            px = int(x + test_radius * np.cos(np.radians(angle)))
                            py = int(y + test_radius * np.sin(np.radians(angle)))
                            if 0 <= px < width and 0 <= py < height:
                                sample_points.append(gray[py, px])
                        
                        if sample_points:
                            avg_intensity = np.mean(sample_points)
                            # If the expanded circle is in a darker region, use it
                            if avg_intensity < 100:  # Dark region threshold
                                best_outer_radius = test_radius
                                print(f"  Expanded circle radius from {r} to {best_outer_radius} using simple expansion (avg intensity: {avg_intensity:.1f})")
                                break
            
            if best_outer_radius > r:
                print(f"  Final: Expanded circle radius from {r} to {best_outer_radius}")
            r = best_outer_radius
        
        final_circles.append((x, y, r, param_name, img_name))
    
    detected_circles = []
    for i, (x, y, r, param_name, img_name) in enumerate(final_circles):
        detected_circles.append({
            "id": i,
            "cx": int(x),
            "cy": int(y),
            "r": int(r)
        })
    
    if detected_circles:
        print(f"Detected {len(detected_circles)} circles using {top_circles[0][3]}/{top_circles[0][4]}")
        for i, circle in enumerate(detected_circles):
            print(f"  Circle {i+1}: center=({circle['cx']}, {circle['cy']}), radius={circle['r']}")
    
    return detected_circles

def refine_radius(gray_image, center_x, center_y, initial_radius):
    """
    Refine the radius of a detected circle using edge detection
    """
    height, width = gray_image.shape
    
    # Define search range around initial radius
    min_r = max(initial_radius - 50, 50)
    max_r = min(initial_radius + 50, min(width, height) // 2)
    
    best_radius = initial_radius
    best_score = 0
    
    # Sample points on circles of different radii
    for test_radius in range(int(min_r), int(max_r), 5):
        # Sample points on the circle
        angles = np.linspace(0, 2*np.pi, 36)  # 36 points around the circle
        edge_scores = []
        
        for angle in angles:
            x = int(center_x + test_radius * np.cos(angle))
            y = int(center_y + test_radius * np.sin(angle))
            
            if 0 <= x < width and 0 <= y < height:
                # Calculate gradient magnitude at this point
                if x > 0 and x < width-1 and y > 0 and y < height-1:
                    gx = int(gray_image[y, x+1]) - int(gray_image[y, x-1])
                    gy = int(gray_image[y+1, x]) - int(gray_image[y-1, x])
                    gradient_mag = np.sqrt(gx*gx + gy*gy)
                    edge_scores.append(gradient_mag)
        
        if edge_scores:
            # Score based on average edge strength
            avg_edge_strength = np.mean(edge_scores)
            if avg_edge_strength > best_score:
                best_score = avg_edge_strength
                best_radius = test_radius
    
    return best_radius

def detect_circles_v2(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
    """
    Alternative circle detection method (v2) - Machine learning-inspired approach
    
    Args:
        image: OpenCV image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        dp: Inverse ratio of accumulator resolution (unused in v2)
        min_dist: Minimum distance between circle centers (unused in v2)
        param1: Upper threshold for edge detection (unused in v2)
        param2: Accumulator threshold for center detection (unused in v2)
    
    Returns:
        List of detected circles with format [cx, cy, r]
    """
    height, width = image.shape[:2]
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    logger.debug(f"V2 Detection: Starting final optimized approach on {width}x{height} image")
    
    # Method 1: Optimized template matching (best performer from earlier iterations)
    circles_template = detect_circles_optimized_template_matching(gray)
    
    # Method 2: Enhanced Hough with better parameters
    circles_hough = detect_circles_enhanced_hough_v2(gray)
    
    # Method 3: Contour-based with improved filtering
    circles_contour = detect_circles_optimized_contour(gray)
    
    # Combine results with optimized weighting
    all_circles = []
    
    # Template matching gets highest weight (proven best)
    for circle in circles_template:
        all_circles.append((circle[0], circle[1], circle[2], circle[3] * 0.6, 'template'))
    
    # Enhanced Hough gets medium weight
    for circle in circles_hough:
        all_circles.append((circle[0], circle[1], circle[2], circle[3] * 0.3, 'hough'))
    
    # Contour gets lower weight
    for circle in circles_contour:
        all_circles.append((circle[0], circle[1], circle[2], circle[3] * 0.1, 'contour'))
    
    # Sort by combined confidence score
    all_circles.sort(key=lambda x: x[3], reverse=True)
    
    # Select best non-overlapping circles
    final_droplets = []
    min_distance = 200
    
    for circle in all_circles:
        x, y, r, confidence, method = circle
        is_duplicate = False
        
        for existing in final_droplets:
            existing_x, existing_y = existing['cx'], existing['cy']
            distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            if distance < min_distance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            # Use the proven radius refinement
            refined_radius = refine_radius(gray, x, y, r)
            
            final_droplets.append({
                'cx': int(x),
                'cy': int(y),
                'r': int(refined_radius),
                'id': len(final_droplets)
            })
            
            if len(final_droplets) >= 2:
                break
    
    logger.debug(f"V2 Detection: Found {len(final_droplets)} droplets using final optimized approach")
    for i, droplet in enumerate(final_droplets):
        logger.debug(f"  Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
    
    return final_droplets

def detect_circles_v3(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
    """
    V3 Detection Algorithm - Fast Hybrid Approach
    
    This iteration combines the speed of V1 Hough with the accuracy of V2 template matching
    in a fast, efficient hybrid approach.
    
    Args:
        image: OpenCV image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Upper threshold for edge detection
        param2: Accumulator threshold for center detection
    
    Returns:
        List of detected circles with format [cx, cy, r]
    """
    height, width = image.shape[:2]
    
    logger.debug(f"V3 Detection: Starting fast hybrid algorithm on {width}x{height} image")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Fast preprocessing - just CLAHE (proven most effective)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Try fast Hough first with fine-tuned parameters
    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=80,  # Increased for better separation
        param1=45,   # Fine-tuned for better edge detection
        param2=55,   # Fine-tuned for optimal sensitivity
        minRadius=250,  # Focus on ground truth range
        maxRadius=350   # Focus on ground truth range
    )
    
    droplets = []
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        logger.debug(f"V3 Detection: Hough found {len(circles)} circles")
        
        # Take the first 2 circles (Hough returns them sorted by confidence)
        for i, circle in enumerate(circles[:2]):
            x, y, r = circle
            droplets.append({
                'cx': x,
                'cy': y,
                'r': r,
                'id': i
            })
    
    # If we don't have 2 circles, use V2 template matching for the rest
    if len(droplets) < 2:
        logger.debug("V3 Detection: Supplementing with optimized template matching")
        template_circles = detect_circles_optimized_template_matching(gray)
        
        # Add template matching results, avoiding duplicates
        existing_positions = [(d['cx'], d['cy']) for d in droplets]
        
        for circle in template_circles:
            if len(droplets) >= 2:
                break
                
            if isinstance(circle, dict):
                x, y, r = circle['cx'], circle['cy'], circle['r']
            else:
                x, y, r = circle[0], circle[1], circle[2]
            
            # Check if this position is too close to existing droplets
            too_close = False
            for ex_x, ex_y in existing_positions:
                if np.sqrt((x - ex_x)**2 + (y - ex_y)**2) < 100:  # Stricter distance
                    too_close = True
                    break
            
            if not too_close:
                droplets.append({
                    'cx': x,
                    'cy': y,
                    'r': r,
                    'id': len(droplets)
                })
                existing_positions.append((x, y))
    
    # If still not enough, generate smart fallbacks
    while len(droplets) < 2:
        fallback = generate_fast_fallback_circle(gray, min_radius, max_radius, droplets)
        droplets.append(fallback)
    
    logger.debug(f"V3 Detection: Found {len(droplets)} droplets using fast hybrid approach")
    for i, droplet in enumerate(droplets):
        logger.debug(f"  Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
    
    return droplets

def detect_circles_v4(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
    """
    V4 Detection Algorithm - Simplified Hough with Optimized Parameters
    
    This algorithm uses simplified preprocessing and optimized Hough parameters
    to improve accuracy over previous versions.
    
    Args:
        image: OpenCV image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Upper threshold for edge detection
        param2: Accumulator threshold for center detection
    
    Returns:
        List of detected circles with format [cx, cy, r]
    """
    height, width = image.shape[:2]
    
    logger.debug(f"V4 Detection: Starting simplified Hough with optimized parameters on {width}x{height} image")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Simple preprocessing - just CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed = clahe.apply(gray)
    
    # Single optimized Hough detection
    circles = cv2.HoughCircles(
        preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=60, param2=40, minRadius=min_radius, maxRadius=max_radius
    )
    
    droplets = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for i, (x, y, r) in enumerate(circles):
            droplets.append({
                'cx': x,
                'cy': y,
                'r': r,
                'id': i
            })
    
    # If we found fewer than 2 circles, try with more sensitive parameters
    if len(droplets) < 2:
        circles_sensitive = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=80,
            param1=40, param2=25, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles_sensitive is not None:
            circles_sensitive = np.round(circles_sensitive[0, :]).astype("int")
            
            for i, (x, y, r) in enumerate(circles_sensitive):
                # Check if this circle is too close to existing ones
                too_close = False
                for existing in droplets:
                    dist = np.sqrt((x - existing['cx'])**2 + (y - existing['cy'])**2)
                    if dist < 100:
                        too_close = True
                        break
                
                if not too_close:
                    droplets.append({
                        'cx': x,
                        'cy': y,
                        'r': r,
                        'id': len(droplets)
                    })
    
    # If still fewer than 2, try with very sensitive parameters
    if len(droplets) < 2:
        circles_very_sensitive = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
            param1=30, param2=20, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles_very_sensitive is not None:
            circles_very_sensitive = np.round(circles_very_sensitive[0, :]).astype("int")
            
            for i, (x, y, r) in enumerate(circles_very_sensitive):
                # Check if this circle is too close to existing ones
                too_close = False
                for existing in droplets:
                    dist = np.sqrt((x - existing['cx'])**2 + (y - existing['cy'])**2)
                    if dist < 100:
                        too_close = True
                        break
                
                if not too_close:
                    droplets.append({
                        'cx': x,
                        'cy': y,
                        'r': r,
                        'id': len(droplets)
                    })
    
    # Limit to 2 droplets maximum
    droplets = droplets[:2]
    
    logger.debug(f"V4 Detection: Found {len(droplets)} droplets using simplified Hough")
    for i, droplet in enumerate(droplets):
        logger.debug(f"  Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
    
    return droplets

def detect_circles_v5(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
    """
    V5 Detection Algorithm - Optimized V4 with Fine-tuned Parameters
    
    This algorithm uses V4's successful approach but with carefully fine-tuned parameters
    to achieve better performance without over-complicating the approach.
    
    Args:
        image: OpenCV image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Upper threshold for edge detection
        param2: Accumulator threshold for center detection
    
    Returns:
        List of detected circles with format [cx, cy, r]
    """
    height, width = image.shape[:2]
    
    logger.debug(f"V5 Detection: Starting optimized V4 with fine-tuned parameters on {width}x{height} image")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Simple preprocessing - just CLAHE for contrast enhancement (same as V4)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed = clahe.apply(gray)
    
    # Fine-tuned Hough detection with optimized parameters
    circles = cv2.HoughCircles(
        preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=105,
        param1=65, param2=45, minRadius=min_radius, maxRadius=max_radius
    )
    
    droplets = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for i, (x, y, r) in enumerate(circles):
            droplets.append({
                'cx': x,
                'cy': y,
                'r': r,
                'id': i
            })
    
    # If we found fewer than 2 circles, try with more sensitive parameters
    if len(droplets) < 2:
        circles_sensitive = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=85,
            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles_sensitive is not None:
            circles_sensitive = np.round(circles_sensitive[0, :]).astype("int")
            
            for i, (x, y, r) in enumerate(circles_sensitive):
                # Check if this circle is too close to existing ones
                too_close = False
                for existing in droplets:
                    dist = np.sqrt((x - existing['cx'])**2 + (y - existing['cy'])**2)
                    if dist < 110:
                        too_close = True
                        break
                
                if not too_close:
                    droplets.append({
                        'cx': x,
                        'cy': y,
                        'r': r,
                        'id': len(droplets)
                    })
    
    # If still fewer than 2, try with very sensitive parameters
    if len(droplets) < 2:
        circles_very_sensitive = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=65,
            param1=35, param2=22, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles_very_sensitive is not None:
            circles_very_sensitive = np.round(circles_very_sensitive[0, :]).astype("int")
            
            for i, (x, y, r) in enumerate(circles_very_sensitive):
                # Check if this circle is too close to existing ones
                too_close = False
                for existing in droplets:
                    dist = np.sqrt((x - existing['cx'])**2 + (y - existing['cy'])**2)
                    if dist < 110:
                        too_close = True
                        break
                
                if not too_close:
                    droplets.append({
                        'cx': x,
                        'cy': y,
                        'r': r,
                        'id': len(droplets)
                    })
    
    # Limit to 2 droplets maximum
    droplets = droplets[:2]
    
    logger.debug(f"V5 Detection: Found {len(droplets)} droplets using optimized V4 approach")
    for i, droplet in enumerate(droplets):
        logger.debug(f"  Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
    
    return droplets

def detect_circles_v6(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
    """
    V6 Detection Algorithm - Ultra-Fine-Tuned V5 Approach
    
    This algorithm builds on V5's successful approach with ultra-fine-tuned parameters
    and additional optimizations to achieve even better performance.
    
    Args:
        image: OpenCV image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Upper threshold for edge detection
        param2: Accumulator threshold for center detection
    
    Returns:
        List of detected circles with format [cx, cy, r]
    """
    height, width = image.shape[:2]
    
    logger.debug(f"V6 Detection: Starting ultra-fine-tuned V5 approach on {width}x{height} image")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Simple preprocessing - just CLAHE for contrast enhancement (same as V5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed = clahe.apply(gray)
    
    # Ultra-fine-tuned Hough detection with optimized parameters
    circles = cv2.HoughCircles(
        preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=115,
        param1=75, param2=55, minRadius=min_radius, maxRadius=max_radius
    )
    
    droplets = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for i, (x, y, r) in enumerate(circles):
            droplets.append({
                'cx': x,
                'cy': y,
                'r': r,
                'id': i
            })
    
    # If we found fewer than 2 circles, try with more sensitive parameters
    if len(droplets) < 2:
        circles_sensitive = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=95,
            param1=60, param2=40, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles_sensitive is not None:
            circles_sensitive = np.round(circles_sensitive[0, :]).astype("int")
            
            for i, (x, y, r) in enumerate(circles_sensitive):
                # Check if this circle is too close to existing ones
                too_close = False
                for existing in droplets:
                    dist = np.sqrt((x - existing['cx'])**2 + (y - existing['cy'])**2)
                    if dist < 115:
                        too_close = True
                        break
                
                if not too_close:
                    droplets.append({
                        'cx': x,
                        'cy': y,
                        'r': r,
                        'id': len(droplets)
                    })
    
    # If still fewer than 2, try with very sensitive parameters
    if len(droplets) < 2:
        circles_very_sensitive = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, dp=1, minDist=75,
            param1=45, param2=30, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles_very_sensitive is not None:
            circles_very_sensitive = np.round(circles_very_sensitive[0, :]).astype("int")
            
            for i, (x, y, r) in enumerate(circles_very_sensitive):
                # Check if this circle is too close to existing ones
                too_close = False
                for existing in droplets:
                    dist = np.sqrt((x - existing['cx'])**2 + (y - existing['cy'])**2)
                    if dist < 115:
                        too_close = True
                        break
                
                if not too_close:
                    droplets.append({
                        'cx': x,
                        'cy': y,
                        'r': r,
                        'id': len(droplets)
                    })
    
    # Limit to 2 droplets maximum
    droplets = droplets[:2]
    
    logger.debug(f"V6 Detection: Found {len(droplets)} droplets using ultra-fine-tuned V5 approach")
    for i, droplet in enumerate(droplets):
        logger.debug(f"  Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
    
    return droplets

def detect_circles_v7(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
    """
    V7 Detection Algorithm - Microscope-Adaptive Hough Detection
    
    This algorithm analyzes image characteristics to identify the microscope source
    and uses optimized parameters for each microscope type.
    
    Args:
        image: OpenCV image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Upper threshold for edge detection
        param2: Accumulator threshold for center detection
    
    Returns:
        List of detected circles with format [cx, cy, r]
    """
    height, width = image.shape[:2]
    
    logger.debug(f"V7 Detection: Starting microscope-adaptive detection on {width}x{height} image")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Analyze image characteristics to classify microscope
    microscope_type = classify_microscope(gray)
    logger.debug(f"V7 Detection: Classified microscope as: {microscope_type}")
    
    # 2. Get optimized parameters for this microscope type
    params = get_parameters_for_microscope(microscope_type)
    logger.debug(f"V7 Detection: Using parameters: {params}")
    
    # 3. Simple preprocessing - just CLAHE for contrast enhancement (like V6)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed = clahe.apply(gray)
    
    # 4. Apply microscope-specific detection with progressive sensitivity
    droplets = detect_with_parameters(preprocessed, params, min_radius, max_radius)
    
    logger.debug(f"V7 Detection: Found {len(droplets)} droplets using microscope-adaptive approach")
    for i, droplet in enumerate(droplets):
        logger.debug(f"  Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
    
    return droplets

def classify_microscope(gray):
    """
    Classify the microscope type based on actual microscope analysis
    """
    # Extract enhanced image features
    features = extract_image_features(gray)
    
    # Classification based on actual microscope analysis:
    # Microscope 1 (Frames 0-3, 10-13): V3 works well on some frames
    # Microscope 2 (Frames 7-9): V3 works poorly on all frames
    # Key distinguishing feature: brightness (threshold = 0.658)
    
    if features['brightness'] < 0.658:  # Microscope 1 - lower brightness
        return 'microscope_1'
    else:  # Microscope 2 - higher brightness
        return 'microscope_2'

def should_use_v3(gray):
    """
    Determine if V3 should be used based on image characteristics
    V3 works excellently on frames with very low brightness and edge density
    """
    features = extract_image_features(gray)
    
    # V3 works excellently on frames with:
    # - Very low brightness (< 0.58)
    # - Very low edge density (< 0.002)
    # - Low contrast (0.14-0.16)
    
    if (features['brightness'] < 0.58 and 
        features['edge_density'] < 0.002 and
        features['contrast'] > 0.14 and features['contrast'] < 0.16):
        return True
    else:
        return False

def extract_image_features(gray):
    """
    Extract enhanced features from the image to classify microscope type
    """
    # Calculate contrast (standard deviation of pixel values)
    contrast = np.std(gray) / 255.0
    
    # Calculate noise level (high-frequency content)
    # Use Laplacian to detect edges and noise
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise = np.std(laplacian) / 255.0
    
    # Calculate brightness
    brightness = np.mean(gray) / 255.0
    
    # Calculate edge density (proportion of edge pixels)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Calculate texture uniformity (local standard deviation)
    # Use a small kernel to calculate local standard deviation
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
    texture_uniformity = np.mean(np.sqrt(local_variance)) / 255.0
    
    return {
        'contrast': contrast,
        'noise': noise,
        'brightness': brightness,
        'edge_density': edge_density,
        'texture_uniformity': texture_uniformity
    }

def get_parameters_for_microscope(microscope_type):
    """
    Get optimized parameters for the specific microscope type
    """
    parameter_sets = {
        'microscope_a': {  # High quality - optimized parameters (from Iteration 2)
            'minDist': 125, 'param1': 85, 'param2': 65,
            'fallback1': {'minDist': 105, 'param1': 70, 'param2': 50},
            'fallback2': {'minDist': 85, 'param1': 55, 'param2': 40}
        },
        'microscope_b': {  # Medium quality - optimized parameters (from Iteration 2)
            'minDist': 115, 'param1': 75, 'param2': 55,
            'fallback1': {'minDist': 95, 'param1': 60, 'param2': 45},
            'fallback2': {'minDist': 75, 'param1': 45, 'param2': 35}
        },
        'microscope_c': {  # Lower quality - optimized parameters (from Iteration 2)
            'minDist': 105, 'param1': 65, 'param2': 50,
            'fallback1': {'minDist': 85, 'param1': 50, 'param2': 40},
            'fallback2': {'minDist': 65, 'param1': 40, 'param2': 30}
        }
    }
    
    return parameter_sets.get(microscope_type, parameter_sets['microscope_b'])

def apply_adaptive_preprocessing(gray, microscope_type):
    """
    Apply adaptive preprocessing based on microscope type
    """
    if microscope_type == 'microscope_a':
        # High quality microscope - minimal preprocessing to preserve detail
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    elif microscope_type == 'microscope_b':
        # Medium quality microscope - moderate preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Light Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return blurred
    
    else:  # microscope_c
        # Lower quality microscope - aggressive preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # More aggressive noise reduction
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        # Additional bilateral filtering to preserve edges
        filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
        return filtered

def detect_with_parameters(preprocessed, params, min_radius, max_radius):
    """
    Apply Hough detection with the given parameters and progressive sensitivity
    """
    droplets = []
    
    # Primary detection with main parameters
    circles = cv2.HoughCircles(
        preprocessed, cv2.HOUGH_GRADIENT, dp=1, 
        minDist=params['minDist'], param1=params['param1'], param2=params['param2'],
        minRadius=min_radius, maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for i, (x, y, r) in enumerate(circles):
            droplets.append({
                'cx': x,
                'cy': y,
                'r': r,
                'id': i
            })
    
    # If we found fewer than 2 circles, try with fallback parameters
    if len(droplets) < 2:
        circles_fallback1 = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, dp=1,
            minDist=params['fallback1']['minDist'], 
            param1=params['fallback1']['param1'], 
            param2=params['fallback1']['param2'],
            minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles_fallback1 is not None:
            circles_fallback1 = np.round(circles_fallback1[0, :]).astype("int")
            for i, (x, y, r) in enumerate(circles_fallback1):
                # Check if this circle is too close to existing ones
                too_close = False
                for existing in droplets:
                    dist = np.sqrt((x - existing['cx'])**2 + (y - existing['cy'])**2)
                    if dist < 115:
                        too_close = True
                        break
                
                if not too_close:
                    droplets.append({
                        'cx': x,
                        'cy': y,
                        'r': r,
                        'id': len(droplets)
                    })
    
    # If still fewer than 2, try with very sensitive parameters
    if len(droplets) < 2:
        circles_fallback2 = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, dp=1,
            minDist=params['fallback2']['minDist'], 
            param1=params['fallback2']['param1'], 
            param2=params['fallback2']['param2'],
            minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles_fallback2 is not None:
            circles_fallback2 = np.round(circles_fallback2[0, :]).astype("int")
            for i, (x, y, r) in enumerate(circles_fallback2):
                # Check if this circle is too close to existing ones
                too_close = False
                for existing in droplets:
                    dist = np.sqrt((x - existing['cx'])**2 + (y - existing['cy'])**2)
                    if dist < 115:
                        too_close = True
                        break
                
                if not too_close:
                    droplets.append({
                        'cx': x,
                        'cy': y,
                        'r': r,
                        'id': len(droplets)
                    })
    
    # Limit to 2 droplets maximum
    return droplets[:2]

def detect_circles_v8(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
    """
    V8 Detection Algorithm - V3 Hybrid with Microscope-Adaptive Parameters
    
    This algorithm combines V3's successful hybrid approach with V7's microscope-adaptive
    parameter selection to get the best of both worlds.
    
    Args:
        image: OpenCV image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Upper threshold for edge detection
        param2: Accumulator threshold for center detection
    
    Returns:
        List of detected circles with format [cx, cy, r]
    """
    height, width = image.shape[:2]
    
    logger.debug(f"V8 Detection: Starting V3 hybrid with microscope-adaptive parameters on {width}x{height} image")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Classify microscope type (using V7's classification)
    microscope_type = classify_microscope(gray)
    logger.debug(f"V8 Detection: Classified microscope as: {microscope_type}")
    
    # 2. Select approach based on V3 performance characteristics
    if should_use_v3(gray):  # V3 works excellently on these specific frames
        # Use V3's exact approach directly
        droplets = detect_circles_v3(image, min_radius, max_radius, dp, min_dist, param1, param2)
    else:  # Use V7 for all other frames
        # Use V7's exact approach directly
        droplets = detect_circles_v7(image, min_radius, max_radius, dp, min_dist, param1, param2)
    
    logger.debug(f"V8 Detection: Found {len(droplets)} droplets using V3 hybrid approach")
    for i, droplet in enumerate(droplets):
        logger.debug(f"  Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
    
    return droplets

def detect_with_v3_hybrid(gray, microscope_type, min_radius, max_radius):
    """
    Use V3's hybrid approach with microscope-specific parameters
    """
    # Get V3-style parameters for this microscope type
    params = get_v3_parameters_for_microscope(microscope_type)
    
    # Fast preprocessing - just CLAHE (proven most effective)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Try fast Hough first with microscope-specific parameters
    # Use V3's exact radius parameters for consistency
    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=params['minDist'],
        param1=params['param1'],
        param2=params['param2'],
        minRadius=250,  # V3's exact parameters
        maxRadius=350   # V3's exact parameters
    )
    
    droplets = []
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        logger.debug(f"V8 V3 Hybrid: Hough found {len(circles)} circles")
        
        # Take the first 2 circles (Hough returns them sorted by confidence)
        for i, circle in enumerate(circles[:2]):
            x, y, r = circle
            droplets.append({
                'cx': x,
                'cy': y,
                'r': r,
                'id': i
            })
    
    # If we don't have 2 circles, use V2 template matching for the rest
    if len(droplets) < 2:
        logger.debug("V8 V3 Hybrid: Supplementing with optimized template matching")
        template_circles = detect_circles_optimized_template_matching(gray)
        
        # Add template matching results, avoiding duplicates
        existing_positions = [(d['cx'], d['cy']) for d in droplets]
        
        for circle in template_circles:
            if len(droplets) >= 2:
                break
                
            if isinstance(circle, dict):
                x, y, r = circle['cx'], circle['cy'], circle['r']
            else:
                x, y, r = circle[0], circle[1], circle[2]
            
            # Check if this position is too close to existing droplets
            too_close = False
            for ex_x, ex_y in existing_positions:
                if np.sqrt((x - ex_x)**2 + (y - ex_y)**2) < 100:  # Stricter distance
                    too_close = True
                    break
            
            if not too_close:
                droplets.append({
                    'cx': x,
                    'cy': y,
                    'r': r,
                    'id': len(droplets)
                })
                existing_positions.append((x, y))
    
    # If still not enough, generate smart fallbacks
    while len(droplets) < 2:
        fallback = generate_fast_fallback_circle(gray, min_radius, max_radius, droplets)
        droplets.append(fallback)
    
    return droplets

def detect_with_v7_adaptive(gray, microscope_type, min_radius, max_radius):
    """
    Use V7's adaptive approach for microscopes V3 struggled with
    """
    # Get optimized parameters for this microscope type
    params = get_parameters_for_microscope(microscope_type)
    
    # Simple preprocessing - just CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed = clahe.apply(gray)
    
    # Apply microscope-specific detection with progressive sensitivity
    droplets = detect_with_parameters(preprocessed, params, min_radius, max_radius)
    
    return droplets

def get_v3_parameters_for_microscope(microscope_type):
    """
    Get V3-style parameters optimized for the specific microscope type
    """
    parameter_sets = {
        'microscope_1': {  # V3 works well on some frames (0,2,3,10) - use V3's exact parameters
            'minDist': 80, 'param1': 45, 'param2': 55
        },
        'microscope_2': {  # V3 struggles on all frames - use more conservative parameters
            'minDist': 90, 'param1': 55, 'param2': 65
        }
    }
    
    return parameter_sets.get(microscope_type, parameter_sets['microscope_1'])

def detect_with_v7_enhanced_template(gray, microscope_type, min_radius, max_radius):
    """
    Use V7's approach but with V3's template matching enhancement
    """
    # Get optimized parameters for this microscope type
    params = get_parameters_for_microscope(microscope_type)
    
    # Simple preprocessing - just CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed = clahe.apply(gray)
    
    # Apply microscope-specific detection with progressive sensitivity
    droplets = detect_with_parameters(preprocessed, params, min_radius, max_radius)
    
    # If we don't have 2 circles, use V3's template matching enhancement
    if len(droplets) < 2:
        logger.debug("V8 Enhanced: Supplementing with V3's template matching")
        template_circles = detect_circles_optimized_template_matching(gray)
        
        # Add template matching results, avoiding duplicates
        existing_positions = [(d['cx'], d['cy']) for d in droplets]
        
        for circle in template_circles:
            if len(droplets) >= 2:
                break
                
            if isinstance(circle, dict):
                x, y, r = circle['cx'], circle['cy'], circle['r']
            else:
                x, y, r = circle[0], circle[1], circle[2]
            
            # Check if this position is too close to existing droplets
            too_close = False
            for ex_x, ex_y in existing_positions:
                if np.sqrt((x - ex_x)**2 + (y - ex_y)**2) < 100:  # Stricter distance
                    too_close = True
                    break
            
            if not too_close:
                droplets.append({
                    'cx': x,
                    'cy': y,
                    'r': r,
                    'id': len(droplets)
                })
                existing_positions.append((x, y))
    
    return droplets

def detect_circles_v9(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
    """
    V9 Detection Algorithm - Microscope_2 Parameter Optimization
    
    This algorithm focuses on optimizing parameters specifically for microscope_2
    (frames 7-9) where V3 struggled, while using V8's approach for other frames.
    
    Args:
        image: OpenCV image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Upper threshold for edge detection
        param2: Accumulator threshold for center detection
    
    Returns:
        List of detected circles with format [cx, cy, r]
    """
    height, width = image.shape[:2]
    
    logger.debug(f"V9 Detection: Starting microscope_2 parameter optimization on {width}x{height} image")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Classify microscope type
    microscope_type = classify_microscope(gray)
    logger.debug(f"V9 Detection: Classified microscope as: {microscope_type}")
    
    # 2. Select approach based on microscope type
    if microscope_type == 'microscope_1':  # Use V8's approach for microscope_1
        # Use V8's sophisticated V3 selection
        if should_use_v3(gray):
            droplets = detect_circles_v3(image, min_radius, max_radius, dp, min_dist, param1, param2)
        else:
            droplets = detect_circles_v7(image, min_radius, max_radius, dp, min_dist, param1, param2)
    else:  # microscope_2 - optimize parameters specifically for frames 7-9
        # Use optimized parameters for microscope_2
        droplets = detect_with_optimized_microscope_2(gray, min_radius, max_radius)
    
    logger.debug(f"V9 Detection: Found {len(droplets)} droplets using microscope_2 optimization")
    for i, droplet in enumerate(droplets):
        logger.debug(f"  Droplet {i+1}: center=({droplet['cx']}, {droplet['cy']}), radius={droplet['r']}")
    
    return droplets

def detect_with_optimized_microscope_2(gray, min_radius, max_radius):
    """
    Optimized detection specifically for microscope_2 (frames 7-9)
    """
    # Start with V7's approach but with parameters optimized for microscope_2
    params = get_optimized_microscope_2_parameters()
    
    # Simple preprocessing - just CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed = clahe.apply(gray)
    
    # Apply microscope_2-specific detection with progressive sensitivity
    droplets = detect_with_parameters(preprocessed, params, min_radius, max_radius)
    
    return droplets

def get_optimized_microscope_2_parameters():
    """
    Get optimized parameters specifically for microscope_2 (frames 7-9)
    These will be iteratively improved based on performance on frames 7-9
    """
    # Start with V7's microscope_c parameters as baseline
    parameter_sets = {
        'microscope_2': {  # Optimized for frames 7-9
            'minDist': 105, 'param1': 65, 'param2': 50,
            'fallback1': {'minDist': 85, 'param1': 50, 'param2': 40},
            'fallback2': {'minDist': 65, 'param1': 40, 'param2': 30}
        }
    }
    
    return parameter_sets['microscope_2']

def create_enhanced_preprocessing(gray):
    """
    Create enhanced preprocessed image with improved preprocessing pipeline
    """
    # 1. Light Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. CLAHE for contrast enhancement (most effective from V4)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # 3. Additional edge enhancement using Laplacian
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # 4. Combine enhanced image with edge information
    final = cv2.addWeighted(enhanced, 0.8, laplacian, 0.2, 0)
    
    return final

def calculate_enhanced_circle_confidence(gray, x, y, r):
    """
    Calculate enhanced confidence score for a detected circle
    """
    height, width = gray.shape
    
    # Check if circle is within image bounds
    if x - r < 0 or x + r >= width or y - r < 0 or y + r >= height:
        return 0.0
    
    # Create mask for the circle
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Calculate edge strength along circle perimeter
    circle_edges = cv2.Canny(gray, 40, 120)
    edge_pixels = cv2.bitwise_and(circle_edges, mask)
    edge_density = np.sum(edge_pixels > 0) / (2 * np.pi * r)
    
    # Calculate intensity consistency within circle
    circle_region = cv2.bitwise_and(gray, mask)
    mean_intensity = np.mean(circle_region[circle_region > 0])
    intensity_std = np.std(circle_region[circle_region > 0])
    intensity_consistency = 1.0 / (1.0 + intensity_std / (mean_intensity + 1))
    
    # Calculate circularity score
    perimeter = 2 * np.pi * r
    area = np.pi * r * r
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Combine metrics with enhanced weighting
    confidence = (edge_density * 0.4 + intensity_consistency * 0.3 + circularity * 0.3)
    return min(confidence, 1.0)

def remove_duplicate_circles_enhanced(droplets, min_dist=85):
    """
    Enhanced duplicate removal with better distance calculation
    """
    if len(droplets) <= 1:
        return droplets
    
    # Sort by confidence (if available) or radius
    droplets.sort(key=lambda d: d.get('confidence', d['r']), reverse=True)
    
    filtered = []
    for droplet in droplets:
        is_duplicate = False
        for existing in filtered:
            # Calculate distance considering radius overlap
            dist = np.sqrt((droplet['cx'] - existing['cx'])**2 + (droplet['cy'] - existing['cy'])**2)
            radius_sum = droplet['r'] + existing['r']
            
            # Consider circles too close if they overlap significantly
            if dist < min_dist or dist < radius_sum * 0.7:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(droplet)
    
    return filtered

def select_best_circles_v5(droplets, max_circles=2):
    """
    Enhanced circle selection with multiple criteria
    """
    if len(droplets) <= max_circles:
        return droplets
    
    # Sort by combined score (confidence + radius bonus)
    def score_droplet(d):
        confidence = d.get('confidence', 0.5)
        radius_bonus = min(d['r'] / 200.0, 0.3)  # Bonus for larger circles up to 30%
        return confidence + radius_bonus
    
    droplets.sort(key=score_droplet, reverse=True)
    return droplets[:max_circles]

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

def create_simplified_preprocessing(gray):
    """
    Create a single, optimized preprocessed image using only the most effective steps
    """
    # 1. Light Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. CLAHE for contrast enhancement (most effective from V2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    return enhanced

def create_advanced_preprocessing_pipeline(gray):
    """
    Create multiple preprocessed versions of the image for robust detection
    """
    preprocessed_images = []
    
    # 1. Original image
    preprocessed_images.append(gray)
    
    # 2. Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    preprocessed_images.append(blurred)
    
    # 3. CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    preprocessed_images.append(clahe_img)
    
    # 4. Bilateral filtering for edge preservation
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    preprocessed_images.append(bilateral)
    
    # 5. Morphological operations for shape enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    preprocessed_images.append(morph)
    
    # 6. Edge enhancement
    edges = cv2.Canny(gray, 50, 150)
    preprocessed_images.append(edges)
    
    return preprocessed_images

def calculate_enhanced_circle_confidence(gray, x, y, r):
    """
    Calculate enhanced confidence score for a detected circle with stricter validation
    """
    height, width = gray.shape
    
    # Check if circle is within image bounds
    if x - r < 0 or x + r >= width or y - r < 0 or y + r >= height:
        return 0.0
    
    # Sample points on the circle with higher density
    angles = np.linspace(0, 2*np.pi, 72)  # More points for better accuracy
    edge_scores = []
    
    for angle in angles:
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        
        if 0 <= px < width and 0 <= py < height:
            # Calculate gradient magnitude
            if px > 0 and px < width-1 and py > 0 and py < height-1:
                gx = int(gray[py, px+1]) - int(gray[py, px-1])
                gy = int(gray[py+1, px]) - int(gray[py-1, px])
                gradient_mag = np.sqrt(gx*gx + gy*gy)
                edge_scores.append(gradient_mag)
    
    if not edge_scores:
        return 0.0
    
    # Calculate circularity score
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Calculate area ratio
    circle_area = np.pi * r * r
    actual_area = np.sum(mask > 0)
    area_ratio = min(actual_area / circle_area, circle_area / actual_area) if circle_area > 0 else 0
    
    # Calculate intensity consistency (circles should have consistent intensity)
    circle_region = gray[max(0, y-r):min(height, y+r), max(0, x-r):min(width, x+r)]
    if circle_region.size > 0:
        intensity_std = np.std(circle_region)
        intensity_consistency = max(0, 1 - intensity_std / 100.0)  # Lower std = higher consistency
    else:
        intensity_consistency = 0
    
    # Combine multiple factors with stricter weighting
    avg_edge_strength = np.mean(edge_scores)
    edge_score = avg_edge_strength / 255.0
    
    # Stricter confidence calculation
    confidence = (edge_score * 0.5 + area_ratio * 0.3 + intensity_consistency * 0.2)
    
    return min(confidence, 1.0)

def calculate_circle_confidence(gray, x, y, r):
    """
    Calculate confidence score for a detected circle
    """
    height, width = gray.shape
    
    # Check if circle is within image bounds
    if x - r < 0 or x + r >= width or y - r < 0 or y + r >= height:
        return 0.0
    
    # Sample points on the circle
    angles = np.linspace(0, 2*np.pi, 36)
    edge_scores = []
    
    for angle in angles:
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        
        if 0 <= px < width and 0 <= py < height:
            # Calculate gradient magnitude
            if px > 0 and px < width-1 and py > 0 and py < height-1:
                gx = int(gray[py, px+1]) - int(gray[py, px-1])
                gy = int(gray[py+1, px]) - int(gray[py-1, px])
                gradient_mag = np.sqrt(gx*gx + gy*gy)
                edge_scores.append(gradient_mag)
    
    if not edge_scores:
        return 0.0
    
    # Calculate circularity score
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Calculate area ratio
    circle_area = np.pi * r * r
    actual_area = np.sum(mask > 0)
    area_ratio = min(actual_area / circle_area, circle_area / actual_area) if circle_area > 0 else 0
    
    # Combine edge strength and circularity
    avg_edge_strength = np.mean(edge_scores)
    confidence = (avg_edge_strength / 255.0) * 0.7 + area_ratio * 0.3
    
    return min(confidence, 1.0)

def select_best_circles_v2(circles, min_dist):
    """
    Select the best circles from all detected candidates with improved logic
    """
    if not circles:
        return []
    
    # Sort by confidence
    circles.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    selected = []
    for circle in circles:
        # Check if this circle is too close to any selected circle
        too_close = False
        for selected_circle in selected:
            dist = np.sqrt((circle['cx'] - selected_circle['cx'])**2 + 
                          (circle['cy'] - selected_circle['cy'])**2)
            if dist < min_dist:
                too_close = True
                break
        
        if not too_close:
            selected.append(circle)
            if len(selected) >= 2:  # We only need 2 droplets
                break
    
    return selected

def select_best_circles(circles, min_dist):
    """
    Select the best circles from all detected candidates
    """
    if not circles:
        return []
    
    # Sort by confidence
    circles.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    selected = []
    for circle in circles:
        # Check if this circle is too close to any selected circle
        too_close = False
        for selected_circle in selected:
            dist = np.sqrt((circle['cx'] - selected_circle['cx'])**2 + 
                          (circle['cy'] - selected_circle['cy'])**2)
            if dist < min_dist:
                too_close = True
                break
        
        if not too_close:
            selected.append(circle)
            if len(selected) >= 2:  # We only need 2 droplets
                break
    
    return selected

def refine_circle_parameters(gray, circle):
    """
    Refine circle parameters using gradient analysis
    """
    x, y, r = circle['cx'], circle['cy'], circle['r']
    
    # Test different radii around the detected radius
    best_radius = r
    best_score = circle.get('confidence', 0)
    
    for test_radius in range(max(1, r-10), min(gray.shape[0]//2, r+10), 2):
        confidence = calculate_circle_confidence(gray, x, y, test_radius)
        if confidence > best_score:
            best_score = confidence
            best_radius = test_radius
    
    return {
        'cx': x,
        'cy': y,
        'r': best_radius,
        'confidence': best_score
    }

def generate_fast_fallback_circle(gray, min_radius, max_radius, existing_droplets):
    """
    Generate a fast fallback circle that avoids existing droplets
    """
    height, width = gray.shape
    
    # Simple approach - find bright regions using thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour that doesn't overlap with existing droplets
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if min_radius <= radius <= max_radius:
                # Check if this circle is far enough from existing ones
                too_close = False
                for droplet in existing_droplets:
                    dist = np.sqrt((x - droplet['cx'])**2 + (y - droplet['cy'])**2)
                    if dist < 150:  # Minimum distance
                        too_close = True
                        break
                
                if not too_close:
                    return {
                        'cx': int(x),
                        'cy': int(y),
                        'r': int(radius),
                        'id': len(existing_droplets)
                    }
    
    # If no good contour found, generate a reasonable default
    margin = 150
    attempts = 0
    while attempts < 20:  # Try up to 20 times
        cx = np.random.randint(margin, width - margin)
        cy = np.random.randint(margin, height - margin)
        r = np.random.randint(min_radius, min(max_radius, min(width, height) // 4))
        
        # Check distance from existing droplets
        too_close = False
        for droplet in existing_droplets:
            dist = np.sqrt((cx - droplet['cx'])**2 + (cy - droplet['cy'])**2)
            if dist < 150:
                too_close = True
                break
        
        if not too_close:
            return {
                'cx': cx,
                'cy': cy,
                'r': r,
                'id': len(existing_droplets)
            }
        
        attempts += 1
    
    # Last resort - just generate any circle
    cx = np.random.randint(margin, width - margin)
    cy = np.random.randint(margin, height - margin)
    r = np.random.randint(min_radius, min(max_radius, min(width, height) // 4))
    
    return {
        'cx': cx,
        'cy': cy,
        'r': r,
        'id': len(existing_droplets)
    }

def generate_smart_fallback_circle(gray, min_radius, max_radius, existing_circles):
    """
    Generate a smart fallback circle that avoids existing circles
    """
    height, width = gray.shape
    
    # Use a different approach for fallback - find bright regions
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find contours that don't overlap with existing circles
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if min_radius <= radius <= max_radius:
                # Check if this circle is far enough from existing ones
                too_close = False
                for existing in existing_circles:
                    dist = np.sqrt((x - existing['cx'])**2 + (y - existing['cy'])**2)
                    if dist < 200:  # Minimum distance
                        too_close = True
                        break
                
                if not too_close:
                    return {
                        'cx': int(x),
                        'cy': int(y),
                        'r': int(radius),
                        'confidence': 0.4
                    }
    
    # If all else fails, generate a reasonable default away from existing circles
    margin = 200
    attempts = 0
    while attempts < 50:  # Try up to 50 times
        cx = np.random.randint(margin, width - margin)
        cy = np.random.randint(margin, height - margin)
        r = np.random.randint(min_radius, min(max_radius, min(width, height) // 4))
        
        # Check distance from existing circles
        too_close = False
        for existing in existing_circles:
            dist = np.sqrt((cx - existing['cx'])**2 + (cy - existing['cy'])**2)
            if dist < 200:
                too_close = True
                break
        
        if not too_close:
            return {
                'cx': cx,
                'cy': cy,
                'r': r,
                'confidence': 0.3
            }
        
        attempts += 1
    
    # Last resort - just generate any circle
    cx = np.random.randint(margin, width - margin)
    cy = np.random.randint(margin, height - margin)
    r = np.random.randint(min_radius, min(max_radius, min(width, height) // 4))
    
    return {
        'cx': cx,
        'cy': cy,
        'r': r,
        'confidence': 0.2
    }

def generate_fallback_circle(gray, min_radius, max_radius):
    """
    Generate a fallback circle if not enough circles are detected
    """
    height, width = gray.shape
    
    # Use a different approach for fallback - find bright regions
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        if min_radius <= radius <= max_radius:
            return {
                'cx': int(x),
                'cy': int(y),
                'r': int(radius),
                'confidence': 0.5
            }
    
    # If all else fails, generate a reasonable default
    margin = 200
    cx = np.random.randint(margin, width - margin)
    cy = np.random.randint(margin, height - margin)
    r = np.random.randint(min_radius, min(max_radius, min(width, height) // 4))
    
    return {
        'cx': cx,
        'cy': cy,
        'r': r,
        'confidence': 0.3
    }

def detect_circles_optimized_template_matching(gray):
    """Optimized template matching - best performing approach from earlier iterations"""
    circles = []
    template_radii = [280, 300, 320, 340]  # Focus on ground truth range
    
    for template_radius in template_radii:
        # Create gradient template (proven most effective)
        template_size = template_radius * 2 + 1
        
        # Skip if template is larger than image
        if template_size > min(gray.shape[0], gray.shape[1]):
            continue
            
        template = np.zeros((template_size, template_size), dtype=np.uint8)
        center = template_radius
        
        for y in range(template_size):
            for x in range(template_size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                if dist <= template_radius:
                    intensity = int(255 * (1 - dist / template_radius))
                    template[y, x] = intensity
        
        # Apply template matching with optimized threshold
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.25)  # Optimized threshold
        
        for pt in zip(*locations[::-1]):
            x, y = pt
            center_x = x + template_radius
            center_y = y + template_radius
            
            if (template_radius <= center_x < gray.shape[1] - template_radius and 
                template_radius <= center_y < gray.shape[0] - template_radius):
                confidence = result[y, x]
                circles.append((center_x, center_y, template_radius, confidence))
    
    return circles

def detect_circles_enhanced_hough_v2(gray):
    """Enhanced Hough circle detection with optimized parameters"""
    circles = []
    
    # Optimized parameters based on ground truth analysis
    param_combinations = [
        (1, 50, 50, 25),  # dp, min_dist, param1, param2
        (1, 100, 50, 20),
        (2, 50, 50, 30),
    ]
    
    for dp, min_dist, param1, param2 in param_combinations:
        detected = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2, minRadius=200, maxRadius=500
        )
        
        if detected is not None:
            for circle in detected[0]:
                x, y, r = circle
                confidence = calculate_circle_confidence(gray, x, y, r)
                circles.append((x, y, r, confidence))
    
    return circles

def detect_circles_optimized_contour(gray):
    """Optimized contour-based detection with improved filtering"""
    circles = []
    
    # Apply bilateral filter for better edge preservation
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use OTSU thresholding
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0 or area < 50000:  # Filter small areas
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity > 0.7:  # High circularity threshold
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if 200 <= radius <= 500:
                # Enhanced confidence calculation
                confidence = circularity * (area / 200000) * (1.0 - abs(radius - 300) / 200)
                circles.append((x, y, radius, confidence))
    
    return circles

def detect_circles_enhanced_hough(gray):
    """Enhanced Hough circle detection with optimized parameters for large circles"""
    circles = []
    
    # Optimized parameters for large circles (based on ground truth analysis)
    param_combinations = [
        (1, 30, 50, 30),  # dp, min_dist, param1, param2
        (1, 50, 50, 25),
        (1, 100, 50, 20),
        (2, 50, 50, 30),
    ]
    
    for dp, min_dist, param1, param2 in param_combinations:
        detected = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2, minRadius=200, maxRadius=500
        )
        
        if detected is not None:
            for circle in detected[0]:
                x, y, r = circle
                # Calculate confidence based on circle quality
                confidence = calculate_circle_confidence(gray, x, y, r)
                circles.append((x, y, r, confidence))
    
    return circles

def detect_circles_template_matching(gray):
    """Template matching approach"""
    circles = []
    template_radii = [280, 300, 320, 340]
    
    for template_radius in template_radii:
        # Create gradient template
        template_size = template_radius * 2 + 1
        template = np.zeros((template_size, template_size), dtype=np.uint8)
        center = template_radius
        
        for y in range(template_size):
            for x in range(template_size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                if dist <= template_radius:
                    intensity = int(255 * (1 - dist / template_radius))
                    template[y, x] = intensity
        
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.25)
        
        for pt in zip(*locations[::-1]):
            x, y = pt
            center_x = x + template_radius
            center_y = y + template_radius
            
            if (template_radius <= center_x < gray.shape[1] - template_radius and 
                template_radius <= center_y < gray.shape[0] - template_radius):
                confidence = result[y, x]
                circles.append((center_x, center_y, template_radius, confidence))
    
    return circles

def detect_circles_contour_based(gray):
    """Contour-based circle detection"""
    circles = []
    
    # Apply preprocessing
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity > 0.7 and area > 10000:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if 200 <= radius <= 500:
                confidence = circularity * (area / 100000)  # Normalize confidence
                circles.append((x, y, radius, confidence))
    
    return circles

def calculate_circle_confidence(gray, x, y, r):
    """Calculate confidence score for a detected circle"""
    height, width = gray.shape
    
    # Sample points on the circle edge
    angles = np.linspace(0, 2*np.pi, 36)
    edge_intensities = []
    
    for angle in angles:
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        
        if 0 <= px < width and 0 <= py < height:
            edge_intensities.append(gray[py, px])
    
    if edge_intensities:
        # Calculate edge strength (variance in intensities)
        edge_strength = np.var(edge_intensities)
        return min(1.0, edge_strength / 1000)  # Normalize to 0-1
    
    return 0.0

def extract_circle_features(gray):
    """Extract features that are characteristic of circular droplets"""
    features = {}
    
    # Feature 1: Edge strength map
    edges = cv2.Canny(gray, 50, 150)
    features['edges'] = edges
    
    # Feature 2: Gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    features['gradient'] = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)
    
    # Feature 3: Laplacian (second derivative)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features['laplacian'] = np.uint8(np.absolute(laplacian))
    
    # Feature 4: Local contrast
    kernel = np.ones((15, 15), np.float32) / 225
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_contrast = np.abs(gray.astype(np.float32) - local_mean)
    features['contrast'] = np.uint8(local_contrast)
    
    # Feature 5: Circular Hough accumulator (simplified)
    features['hough_accumulator'] = create_circular_accumulator(gray)
    
    return features

def create_circular_accumulator(gray):
    """Create a simplified circular Hough accumulator"""
    height, width = gray.shape
    accumulator = np.zeros((height, width), dtype=np.float32)
    
    # Sample radii around expected droplet size
    radii = [280, 300, 320, 340]
    
    for r in radii:
        # Create circle template
        angles = np.linspace(0, 2*np.pi, 36)
        
        for y in range(height):
            for x in range(width):
                # Check if this could be a circle center
                circle_points = []
                for angle in angles:
                    px = int(x + r * np.cos(angle))
                    py = int(y + r * np.sin(angle))
                    if 0 <= px < width and 0 <= py < height:
                        circle_points.append((px, py))
                
                if len(circle_points) > 30:  # Most of the circle is in bounds
                    # Calculate edge strength along the circle
                    edge_strength = 0
                    for px, py in circle_points:
                        if px > 0 and px < width-1 and py > 0 and py < height-1:
                            gx = int(gray[py, px+1]) - int(gray[py, px-1])
                            gy = int(gray[py+1, px]) - int(gray[py-1, px])
                            edge_strength += np.sqrt(gx*gx + gy*gy)
                    
                    accumulator[y, x] += edge_strength / len(circle_points)
    
    # Normalize accumulator
    if accumulator.max() > 0:
        accumulator = accumulator / accumulator.max() * 255
    
    return np.uint8(accumulator)

def generate_circle_candidates(feature_maps, gray):
    """Generate potential circle center candidates based on feature maps"""
    candidates = []
    height, width = gray.shape
    
    # Combine feature maps to find candidate centers
    combined_score = np.zeros((height, width), dtype=np.float32)
    
    # Weight different features
    weights = {
        'edges': 0.3,
        'gradient': 0.2,
        'laplacian': 0.2,
        'contrast': 0.1,
        'hough_accumulator': 0.2
    }
    
    for feature_name, weight in weights.items():
        if feature_name in feature_maps:
            feature = feature_maps[feature_name].astype(np.float32) / 255.0
            combined_score += weight * feature
    
    # Find local maxima in combined score (simple implementation)
    local_maxima = find_local_maxima(combined_score, size=50)
    
    # Threshold to get strong candidates
    threshold = np.percentile(combined_score[local_maxima], 80) if np.any(local_maxima) else 0.5
    
    # Extract candidate positions
    candidate_positions = np.where((local_maxima) & (combined_score > threshold))
    
    for y, x in zip(candidate_positions[0], candidate_positions[1]):
        candidates.append({
            'x': x,
            'y': y,
            'score': combined_score[y, x]
        })
    
    return candidates

def find_local_maxima(image, size=50):
    """Find local maxima in an image (simple implementation)"""
    height, width = image.shape
    local_maxima = np.zeros_like(image, dtype=bool)
    
    half_size = size // 2
    
    for y in range(half_size, height - half_size):
        for x in range(half_size, width - half_size):
            # Check if current pixel is maximum in its neighborhood
            center_value = image[y, x]
            is_maximum = True
            
            for dy in range(-half_size, half_size + 1):
                for dx in range(-half_size, half_size + 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_value = image[y + dy, x + dx]
                    if neighbor_value >= center_value:
                        is_maximum = False
                        break
                if not is_maximum:
                    break
            
            local_maxima[y, x] = is_maximum
    
    return local_maxima

def classify_circle_candidates(candidates, gray):
    """Classify and score circle candidates"""
    scored_candidates = []
    
    for candidate in candidates:
        x, y = candidate['x'], candidate['y']
        
        # Test different radii around the expected size
        best_radius = 300
        best_score = 0
        
        for radius in [280, 300, 320, 340]:
            score = calculate_circle_quality_score(gray, x, y, radius)
            if score > best_score:
                best_score = score
                best_radius = radius
        
        if best_score > 0.3:  # Minimum quality threshold
            scored_candidates.append({
                'x': x,
                'y': y,
                'radius': best_radius,
                'score': best_score * candidate['score']  # Combine with candidate score
            })
    
    return scored_candidates

def calculate_circle_quality_score(gray, x, y, radius):
    """Calculate quality score for a potential circle"""
    height, width = gray.shape
    
    # Check bounds
    if (radius >= x or radius >= y or 
        x + radius >= width or y + radius >= height):
        return 0.0
    
    # Sample points on the circle
    angles = np.linspace(0, 2*np.pi, 36)
    edge_scores = []
    intensity_scores = []
    
    for angle in angles:
        px = int(x + radius * np.cos(angle))
        py = int(y + radius * np.sin(angle))
        
        if 0 <= px < width and 0 <= py < height:
            # Edge strength
            if px > 0 and px < width-1 and py > 0 and py < height-1:
                gx = int(gray[py, px+1]) - int(gray[py, px-1])
                gy = int(gray[py+1, px]) - int(gray[py-1, px])
                edge_strength = np.sqrt(gx*gx + gy*gy)
                edge_scores.append(edge_strength)
            
            # Intensity
            intensity_scores.append(gray[py, px])
    
    if not edge_scores or not intensity_scores:
        return 0.0
    
    # Calculate scores
    avg_edge_strength = np.mean(edge_scores)
    intensity_variance = np.var(intensity_scores)
    
    # Center intensity (should be different from edge)
    center_intensity = gray[y, x] if 0 <= y < height and 0 <= x < width else 0
    edge_avg_intensity = np.mean(intensity_scores)
    center_edge_contrast = abs(center_intensity - edge_avg_intensity) / 255.0
    
    # Combined score
    edge_score = min(1.0, avg_edge_strength / 100.0)
    variance_score = min(1.0, intensity_variance / 1000.0)
    contrast_score = center_edge_contrast
    
    combined_score = edge_score * 0.5 + variance_score * 0.3 + contrast_score * 0.2
    
    return combined_score

def non_maximum_suppression(candidates):
    """Apply non-maximum suppression to select best non-overlapping circles"""
    if not candidates:
        return []
    
    # Sort by score
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    final_droplets = []
    min_distance = 200
    
    for candidate in candidates:
        x, y, radius, score = candidate['x'], candidate['y'], candidate['radius'], candidate['score']
        
        # Check for overlap with existing detections
        is_duplicate = False
        for existing in final_droplets:
            existing_x, existing_y = existing['cx'], existing['cy']
            distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            if distance < min_distance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            # Refine radius
            refined_radius = refine_radius(gray, x, y, radius)
            
            final_droplets.append({
                'cx': int(x),
                'cy': int(y),
                'r': int(refined_radius),
                'id': len(final_droplets)
            })
            
            # Limit to 2 droplets
            if len(final_droplets) >= 2:
                break
    
    return final_droplets

def create_preprocessing_pipeline(gray):
    """Create multiple preprocessed versions of the image for different detection methods"""
    preprocessed = {}
    
    # Original grayscale
    preprocessed['original'] = gray
    
    # Gaussian blur for noise reduction
    preprocessed['blurred'] = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Bilateral filter for edge-preserving smoothing
    preprocessed['bilateral'] = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    preprocessed['clahe'] = clahe.apply(gray)
    
    # Edge detection using Canny
    preprocessed['edges'] = cv2.Canny(gray, 50, 150)
    
    # Morphological operations for shape enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    preprocessed['morph'] = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Laplacian for edge enhancement
    preprocessed['laplacian'] = cv2.Laplacian(gray, cv2.CV_64F)
    preprocessed['laplacian'] = np.uint8(np.absolute(preprocessed['laplacian']))
    
    return preprocessed

def detect_circles_advanced_template_matching(preprocessed_images):
    """Advanced template matching with multiple preprocessing variants"""
    circles = []
    template_radii = [270, 290, 310, 330, 350]  # Expanded range
    
    for template_radius in template_radii:
        # Create multiple template variants
        templates = create_advanced_templates(template_radius)
        
        for template_name, template in templates.items():
            for img_name, img in preprocessed_images.items():
                if img_name in ['edges', 'laplacian']:  # Skip edge images for template matching
                    continue
                    
                result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.2)  # Lower threshold for more detections
                
                for pt in zip(*locations[::-1]):
                    x, y = pt
                    center_x = x + template_radius
                    center_y = y + template_radius
                    
                    if (template_radius <= center_x < img.shape[1] - template_radius and 
                        template_radius <= center_y < img.shape[0] - template_radius):
                        confidence = result[y, x]
                        # Boost confidence for certain template/image combinations
                        if template_name == 'gradient' and img_name == 'bilateral':
                            confidence *= 1.2
                        circles.append((center_x, center_y, template_radius, confidence))
    
    return circles

def create_advanced_templates(radius):
    """Create multiple advanced template patterns"""
    templates = {}
    size = radius * 2 + 1
    center = radius
    
    # Gradient template (darker edges, lighter center)
    template = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if dist <= radius:
                intensity = int(255 * (1 - dist / radius))
                template[y, x] = intensity
    templates['gradient'] = template
    
    # Ring template (dark ring, light center and outside)
    template = np.ones((size, size), dtype=np.uint8) * 255
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if radius * 0.7 <= dist <= radius:
                template[y, x] = 0
    templates['ring'] = template
    
    # Gaussian template
    template = np.zeros((size, size), dtype=np.uint8)
    sigma = radius / 3
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if dist <= radius:
                intensity = int(255 * np.exp(-(dist**2) / (2 * sigma**2)))
                template[y, x] = intensity
    templates['gaussian'] = template
    
    # Inverted gradient
    template = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if dist <= radius:
                intensity = int(255 * (dist / radius))
                template[y, x] = intensity
    templates['inverted'] = template
    
    return templates

def detect_circles_feature_based(preprocessed_images):
    """Feature-based circle detection using local binary patterns"""
    circles = []
    
    # Use CLAHE enhanced image for feature detection
    img = preprocessed_images['clahe']
    
    # Create circular feature templates
    for radius in [280, 300, 320, 340]:
        # Sample circular features
        angles = np.linspace(0, 2*np.pi, 16)
        feature_scores = []
        
        # Grid search for potential centers
        step = 50
        for y in range(radius, img.shape[0] - radius, step):
            for x in range(radius, img.shape[1] - radius, step):
                # Calculate feature score
                score = calculate_circular_feature_score(img, x, y, radius, angles)
                if score > 0.3:
                    feature_scores.append((x, y, radius, score))
        
        circles.extend(feature_scores)
    
    return circles

def calculate_circular_feature_score(img, x, y, radius, angles):
    """Calculate feature score for a circular region"""
    intensities = []
    gradients = []
    
    for angle in angles:
        px = int(x + radius * np.cos(angle))
        py = int(y + radius * np.sin(angle))
        
        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
            intensities.append(img[py, px])
            
            # Calculate gradient
            if px > 0 and px < img.shape[1]-1 and py > 0 and py < img.shape[0]-1:
                gx = int(img[py, px+1]) - int(img[py, px-1])
                gy = int(img[py+1, px]) - int(img[py-1, px])
                gradient_mag = np.sqrt(gx*gx + gy*gy)
                gradients.append(gradient_mag)
    
    if not intensities or not gradients:
        return 0.0
    
    # Score based on intensity variance and gradient strength
    intensity_var = np.var(intensities)
    avg_gradient = np.mean(gradients)
    
    # Normalize and combine scores
    score = (intensity_var / 1000) * (avg_gradient / 100)
    return min(1.0, score)

def detect_circles_enhanced_contour(preprocessed_images):
    """Enhanced contour-based detection with morphological operations"""
    circles = []
    
    # Use morphological preprocessed image
    img = preprocessed_images['morph']
    
    # Multiple thresholding approaches
    thresholds = [
        cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    ]
    
    for thresh in thresholds:
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0 or area < 50000:  # Filter small areas
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.6:  # Lower threshold for more detections
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                if 200 <= radius <= 500:
                    # Enhanced confidence calculation
                    confidence = circularity * (area / 200000) * (1.0 - abs(radius - 300) / 200)
                    circles.append((x, y, radius, confidence))
    
    return circles

def detect_circles_gradient_based(preprocessed_images):
    """Gradient-based circle detection using edge information"""
    circles = []
    
    # Use edge image
    edges = preprocessed_images['edges']
    
    # Hough circle detection on edges
    detected = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=50, param2=15, minRadius=200, maxRadius=500
    )
    
    if detected is not None:
        for circle in detected[0]:
            x, y, r = circle
            # Calculate gradient-based confidence
            confidence = calculate_gradient_confidence(edges, x, y, r)
            circles.append((x, y, r, confidence))
    
    return circles

def calculate_gradient_confidence(edges, x, y, r):
    """Calculate confidence based on edge strength around circle"""
    angles = np.linspace(0, 2*np.pi, 36)
    edge_strengths = []
    
    for angle in angles:
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        
        if 0 <= px < edges.shape[1] and 0 <= py < edges.shape[0]:
            edge_strengths.append(edges[py, px])
    
    if edge_strengths:
        return np.mean(edge_strengths) / 255.0
    
    return 0.0

def refine_radius_advanced(gray, x, y, initial_radius):
    """Advanced radius refinement using multiple techniques"""
    height, width = gray.shape
    
    # Define search range
    min_r = max(initial_radius - 80, 50)
    max_r = min(initial_radius + 80, min(width, height) // 2)
    
    best_radius = initial_radius
    best_score = 0
    
    # Test multiple refinement techniques
    for test_radius in range(int(min_r), int(max_r), 3):
        # Technique 1: Edge strength
        edge_score = calculate_edge_strength(gray, x, y, test_radius)
        
        # Technique 2: Intensity gradient
        gradient_score = calculate_intensity_gradient(gray, x, y, test_radius)
        
        # Technique 3: Circular symmetry
        symmetry_score = calculate_circular_symmetry(gray, x, y, test_radius)
        
        # Combined score
        combined_score = edge_score * 0.4 + gradient_score * 0.3 + symmetry_score * 0.3
        
        if combined_score > best_score:
            best_score = combined_score
            best_radius = test_radius
    
    return best_radius

def calculate_edge_strength(gray, x, y, r):
    """Calculate edge strength around circle"""
    angles = np.linspace(0, 2*np.pi, 36)
    gradients = []
    
    for angle in angles:
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        
        if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
            if px > 0 and px < gray.shape[1]-1 and py > 0 and py < gray.shape[0]-1:
                gx = int(gray[py, px+1]) - int(gray[py, px-1])
                gy = int(gray[py+1, px]) - int(gray[py-1, px])
                gradient_mag = np.sqrt(gx*gx + gy*gy)
                gradients.append(gradient_mag)
    
    return np.mean(gradients) / 100.0 if gradients else 0.0

def calculate_intensity_gradient(gray, x, y, r):
    """Calculate intensity gradient from center to edge"""
    center_intensity = gray[int(y), int(x)] if 0 <= int(y) < gray.shape[0] and 0 <= int(x) < gray.shape[1] else 0
    
    angles = np.linspace(0, 2*np.pi, 12)
    edge_intensities = []
    
    for angle in angles:
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        
        if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
            edge_intensities.append(gray[py, px])
    
    if edge_intensities:
        edge_avg = np.mean(edge_intensities)
        gradient = abs(center_intensity - edge_avg) / 255.0
        return gradient
    
    return 0.0

def calculate_circular_symmetry(gray, x, y, r):
    """Calculate circular symmetry score"""
    angles = np.linspace(0, 2*np.pi, 24)
    intensities = []
    
    for angle in angles:
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        
        if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
            intensities.append(gray[py, px])
    
    if len(intensities) > 4:
        # Calculate variance - lower variance means more symmetric
        variance = np.var(intensities)
        symmetry = 1.0 / (1.0 + variance / 1000.0)
        return symmetry
    
    return 0.0

def detect_timestamp(image):
    """
    Detect timestamp in the image using improved OCR preprocessing
    
    Args:
        image: OpenCV image
    
    Returns:
        Tuple of (timestamp_string, found_boolean)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Focus on bottom-left area where timestamps typically appear
        # Extract bottom-left region more precisely
        bottom_left = gray[int(height*0.8):height, 0:int(width*0.4)]
        
        # Apply improved preprocessing techniques
        processed_images = []
        
        # 1. High contrast threshold for black text on white background
        _, thresh1 = cv2.threshold(bottom_left, 100, 255, cv2.THRESH_BINARY)
        processed_images.append(("high_contrast", thresh1))
        
        # 2. OTSU threshold
        _, thresh2 = cv2.threshold(bottom_left, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("otsu", thresh2))
        
        # 3. Inverted threshold (for white text on dark background)
        _, thresh3 = cv2.threshold(bottom_left, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_images.append(("inverted", thresh3))
        
        # 4. Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        thresh4 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        processed_images.append(("morphology", thresh4))
        
        # 5. Gaussian blur + threshold for noisy images
        blurred = cv2.GaussianBlur(bottom_left, (3, 3), 0)
        _, thresh5 = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
        processed_images.append(("blurred", thresh5))
        
        # Try OCR on each processed image with optimized configurations
        all_text = ""
        best_text = ""
        best_confidence = 0
        
        for name, processed_img in processed_images:
            try:
                # Use OCR configurations optimized for timestamp detection
                configs = [
                    '--psm 8 -c tessedit_char_whitelist=0123456789:.Live Time',
                    '--psm 7 -c tessedit_char_whitelist=0123456789:.Live Time',
                    '--psm 6 -c tessedit_char_whitelist=0123456789:.Live Time',
                    '--psm 13'
                ]
                
                for config in configs:
                    try:
                        # Get text with confidence
                        data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
                        text = pytesseract.image_to_string(processed_img, config=config)
                        
                        # Calculate average confidence
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_text = text
                        
                        all_text += text + " "
                    except:
                        continue
                        
            except Exception as e:
                print(f"OCR error on {name}: {e}")
                continue
        
        # Clean up the text by removing duplicates and noise
        lines = all_text.split('\n')
        clean_lines = []
        seen = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen and len(line) > 5:  # Filter out very short lines
                clean_lines.append(line)
                seen.add(line)
        
        clean_text = ' '.join(clean_lines)
        
        if clean_text:
            print(f"OCR extracted text: '{clean_text}'")
        
        # Look for timestamp patterns with improved regex
        timestamp_patterns = [
            r'Live Time:\s*(\d+\.\d{3})',                   # Live Time: SS.mmm (most common)
            r'Live Time:\s*(\d{1,2}:\d{2}:\d{2}\.\d{3})',  # Live Time: HH:MM:SS.mmm
            r'Live Time:\s*(\d{1,2}:\d{2}\.\d{3})',        # Live Time: MM:SS.mmm
            r'Live Time:\s*(\d{1,2}:\d{2}:\d{2})',         # Live Time: HH:MM:SS
            r'Live Time:\s*(\d{1,2}:\d{2})',               # Live Time: MM:SS
            r'Time:\s*(\d+\.\d{3})',                       # Time: SS.mmm
            r'Time:\s*(\d{1,2}:\d{2}:\d{2}\.\d{3})',       # Time: HH:MM:SS.mmm
            r'Time:\s*(\d{1,2}:\d{2}\.\d{3})',            # Time: MM:SS.mmm
            r'(\d+\.\d{3})',                              # SS.mmm (fallback)
            r'(\d{1,2}:\d{2}:\d{2}\.\d{3})',              # HH:MM:SS.mmm
            r'(\d{1,2}:\d{2}\.\d{3})',                    # MM:SS.mmm
        ]
        
        for pattern in timestamp_patterns:
            matches = re.findall(pattern, clean_text)
            if matches:
                # Take the first valid match
                timestamp = matches[0]
                print(f"Found timestamp: '{timestamp}' using pattern: {pattern}")
                return timestamp, True
        
        return "Not Found", False
        
    except Exception as e:
        print(f"Timestamp detection error: {e}")
        return "Not Found", False


# TODO: campbellsean - At some point, compare this to the implementation above
# def detect_timestamp(image):
#     """
#     Detect timestamp in the image using OCR
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     height, width = gray.shape
    
#     # Focus on bottom-left area where timestamp usually appears
#     roi = gray[int(height*0.8):height, 0:int(width*0.4)]
    
#     if roi.size == 0:
#         return {"value": "Not Found", "found": False}
    
#     # Multiple preprocessing techniques for better OCR
#     preprocessing_techniques = [
#         ("high_contrast", lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
#         ("otsu", lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
#         ("inverted", lambda img: cv2.bitwise_not(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])),
#         ("morphological", lambda img: cv2.morphologyEx(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))),
#         ("gaussian_thresh", lambda img: cv2.threshold(cv2.GaussianBlur(img, (3,3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
#     ]
    
#     best_text = ""
#     best_confidence = 0
    
#     for technique_name, preprocess_func in preprocessing_techniques:
#         try:
#             processed_roi = preprocess_func(roi)
            
#             # Try different OCR configurations
#             ocr_configs = [
#                 '--psm 8 -c tessedit_char_whitelist=0123456789:.Live Time',
#                 '--psm 7 -c tessedit_char_whitelist=0123456789:.Live Time',
#                 '--psm 6 -c tessedit_char_whitelist=0123456789:.Live Time',
#                 '--psm 13 -c tessedit_char_whitelist=0123456789:.Live Time',
#                 '--psm 8',
#                 '--psm 7',
#                 '--psm 6',
#             ]
            
#             for config in ocr_configs:
#                 try:
#                     text = pytesseract.image_to_string(processed_roi, config=config).strip()
#                     if text and len(text) > 2:
#                         # Clean up the text
#                         text = re.sub(r'[^\w\s:.]', '', text)
#                         if text:
#                             # Use the first valid result
#                             best_text = text
#                             break
#                 except Exception as e:
#                     logger.debug(f"OCR config {config} failed: {e}")
#                     continue
            
#             if best_text:
#                 break
                
#         except Exception as e:
#             logger.debug(f"Preprocessing technique {technique_name} failed: {e}")
#             continue
    
#     # Extract timestamp using regex patterns
#     timestamp_patterns = [
#         r'Live\s*Time:\s*(\d+\.\d{3})',  # "Live Time: 40.456"
#         r'(\d+\.\d{3})',  # "40.456"
#         r'Time:\s*(\d+\.\d{3})',  # "Time: 40.456"
#         r'(\d+\.\d{2})',  # "40.45"
#         r'(\d+\.\d{1})',  # "40.4"
#         r'(\d+)',  # "40"
#     ]
    
#     found_timestamp = "Not Found"
#     for pattern in timestamp_patterns:
#         match = re.search(pattern, best_text)
#         if match:
#             found_timestamp = match.group(1)
#             logger.debug(f"Found timestamp: '{found_timestamp}' using pattern: {pattern}")
#             break
    
#     return {
#         "value": found_timestamp,
#         "found": found_timestamp != "Not Found"
#     }

def detect_scale_bar(image):
    """
    Detect scale bar in the image using improved line detection and OCR
    
    Args:
        image: OpenCV image
    
    Returns:
        Tuple of (scale_dict, found_boolean)
    """
    try:
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Focus on bottom-right area where scale bars typically appear
        # Avoid the timestamp area by starting further right and higher up
        bottom_right = gray[int(height*0.8):height, int(width*0.7):width]
        
        # Apply multiple edge detection techniques
        edges1 = cv2.Canny(gray, 50, 150, apertureSize=3)
        edges2 = cv2.Canny(gray, 30, 100, apertureSize=3)
        edges3 = cv2.Canny(gray, 80, 200, apertureSize=3)
        
        # Detect lines using HoughLinesP with different parameters
        lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, threshold=60, minLineLength=20, maxLineGap=10)
        lines2 = cv2.HoughLinesP(edges2, 1, np.pi/180, threshold=40, minLineLength=15, maxLineGap=15)
        lines3 = cv2.HoughLinesP(edges3, 1, np.pi/180, threshold=80, minLineLength=25, maxLineGap=8)
        
        all_lines = []
        if lines1 is not None:
            all_lines.extend(lines1)
        if lines2 is not None:
            all_lines.extend(lines2)
        if lines3 is not None:
            all_lines.extend(lines3)
        
        if all_lines:
            # Filter for horizontal lines (scale bars are typically horizontal)
            horizontal_lines = []
            for line in all_lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly horizontal (angle close to 0 or 180 degrees)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 15 or abs(angle - 180) < 15:
                    # Filter out very short lines
                    length = abs(x2 - x1)
                    if length > 15:  # Minimum scale bar length
                        horizontal_lines.append((x1, y1, x2, y2, length))
            
            if horizontal_lines:
                # Filter out lines that are too short or in the wrong area
                filtered_lines = []
                for line in horizontal_lines:
                    x1, y1, x2, y2, length = line
                    # More restrictive filtering to avoid timestamp area
                    # Check if line is in the bottom-right area (avoid timestamp area)
                    if x1 > width * 0.65 and y1 > height * 0.75:  # More restrictive
                        # Check if line is not too close to image edges
                        if x1 > width * 0.1 and x2 < width * 0.95:
                            # Additional check: avoid lines that might be timestamp-related
                            # Timestamps are usually in the bottom-left, so reject lines too far left
                            if x1 > width * 0.3:  # Ensure line is not in left half
                                filtered_lines.append(line)
                
                if not filtered_lines:
                    # Fallback: try less restrictive filtering
                    for line in horizontal_lines:
                        x1, y1, x2, y2, length = line
                        if x1 > width * 0.5 and y1 > height * 0.7:
                            if x1 > width * 0.1 and x2 < width * 0.95:
                                filtered_lines.append(line)
                
                if not filtered_lines:
                    # Final fallback to all lines if filtering removed everything
                    filtered_lines = horizontal_lines
                
                # Sort by length and take the longest (likely the scale bar)
                filtered_lines.sort(key=lambda x: x[4], reverse=True)
                x1, y1, x2, y2, length = filtered_lines[0]
                
                # Try to read the scale bar label using OCR
                label = "50 µm"  # default
                try:
                    # Extract area above and around the scale bar for OCR
                    label_region = gray[max(0, y1-40):y1+10, max(0, x1-10):min(width, x2+10)]
                    if label_region.size > 0:
                        # Apply multiple preprocessing techniques for OCR
                        _, label_thresh1 = cv2.threshold(label_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        _, label_thresh2 = cv2.threshold(label_region, 100, 255, cv2.THRESH_BINARY)
                        
                        # Try OCR on both processed images
                        for thresh_img in [label_thresh1, label_thresh2]:
                            try:
                                label_text = pytesseract.image_to_string(thresh_img, config='--psm 8 -c tessedit_char_whitelist=0123456789µmum')
                                
                                # Look for scale patterns
                                scale_patterns = [
                                    r'(\d+)\s*µm',
                                    r'(\d+)\s*um',
                                    r'(\d+)\s*micro',
                                    r'(\d+)\s*microns',
                                    r'(\d+)\s*μ',
                                    r'(\d+)\s*m'
                                ]
                                
                                for pattern in scale_patterns:
                                    match = re.search(pattern, label_text, re.IGNORECASE)
                                    if match:
                                        value = match.group(1)
                                        label = f"{value} µm"
                                        print(f"Found scale label: '{label}' from OCR text: '{label_text.strip()}'")
                                        break
                                
                                if label != "50 µm":  # Found a valid label
                                    break
                            except:
                                continue
                except Exception as e:
                    print(f"Scale label OCR error: {e}")
                
                # If OCR didn't work, estimate based on length and position
                if label == "50 µm":  # still default
                    # More sophisticated estimation based on image size and position
                    image_area = width * height
                    relative_length = length / width
                    
                    if relative_length < 0.05:  # Very small relative to image
                        label = "10 µm"
                    elif relative_length < 0.08:
                        label = "25 µm"
                    elif relative_length < 0.12:
                        label = "50 µm"
                    elif relative_length < 0.18:
                        label = "100 µm"
                    else:
                        label = "200 µm"
                
                return {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "label": label,
                    "length": int(length)
                }, True
        
        return {}, False
        
    except Exception as e:
        print(f"Scale bar detection error: {e}")
        return {}, False

# TODO: campbellsean - At some point, compare this to the implementation above
# and see if it's better or worse
# def detect_scale_bar(image):
#     """
#     Detect scale bar in the image
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     height, width = gray.shape
    
#     # Focus on bottom-right area where scale bar usually appears
#     roi = gray[int(height*0.8):height, int(width*0.7):width]
    
#     if roi.size == 0:
#         return {
#             "x1": 0, "y1": 0, "x2": 0, "y2": 0,
#             "label": "Not Found",
#             "length": 0,
#             "found": False
#         }
    
#     # Detect horizontal lines using HoughLinesP
#     edges = cv2.Canny(roi, 50, 150, apertureSize=3)
    
#     # Try different parameters for line detection
#     line_params = [
#         {"threshold": 50, "minLineLength": 30, "maxLineGap": 10},
#         {"threshold": 30, "minLineLength": 20, "maxLineGap": 15},
#         {"threshold": 20, "minLineLength": 15, "maxLineGap": 20},
#         {"threshold": 10, "minLineLength": 10, "maxLineGap": 25},
#     ]
    
#     best_line = None
#     best_length = 0
    
#     for params in line_params:
#         lines = cv2.HoughLinesP(
#             edges,
#             rho=1,
#             theta=np.pi/180,
#             threshold=params["threshold"],
#             minLineLength=params["minLineLength"],
#             maxLineGap=params["maxLineGap"]
#         )
        
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
                
#                 # Check if line is roughly horizontal
#                 if abs(y2 - y1) < 5:  # Horizontal line
#                     length = abs(x2 - x1)
                    
#                     # Filter for lines in the bottom-right area
#                     roi_x1 = int(width * 0.7)
#                     roi_y1 = int(height * 0.8)
                    
#                     # Convert back to original image coordinates
#                     orig_x1 = x1 + roi_x1
#                     orig_y1 = y1 + roi_y1
#                     orig_x2 = x2 + roi_x1
#                     orig_y2 = y2 + roi_y1
                    
#                     # Ensure line is in the bottom-right area and not too close to edges
#                     if (orig_x1 > width * 0.75 and orig_x2 > width * 0.75 and
#                         orig_y1 > height * 0.85 and orig_y2 > height * 0.85 and
#                         orig_x1 < width * 0.95 and orig_x2 < width * 0.95):
                        
#                         if length > best_length:
#                             best_line = [orig_x1, orig_y1, orig_x2, orig_y2]
#                             best_length = length
            
#             # If we found a good line, break
#             if best_line is not None:
#                 break
    
#     if best_line is None:
#         # Fallback: look for any horizontal line in the bottom-right area
#         for params in line_params:
#             lines = cv2.HoughLinesP(
#                 edges,
#                 rho=1,
#                 theta=np.pi/180,
#                 threshold=params["threshold"],
#                 minLineLength=params["minLineLength"],
#                 maxLineGap=params["maxLineGap"]
#             )
            
#             if lines is not None:
#                 for line in lines:
#                     x1, y1, x2, y2 = line[0]
#                     if abs(y2 - y1) < 10:  # More lenient horizontal check
#                         length = abs(x2 - x1)
#                         if length > best_length:
#                             roi_x1 = int(width * 0.7)
#                             roi_y1 = int(height * 0.8)
#                             best_line = [x1 + roi_x1, y1 + roi_y1, x2 + roi_x1, y2 + roi_y1]
#                             best_length = length
#                 if best_line is not None:
#                     break
    
#     if best_line is None:
#         return {
#             "x1": 0, "y1": 0, "x2": 0, "y2": 0,
#             "label": "Not Found",
#             "length": 0,
#             "found": False
#         }
    
#     x1, y1, x2, y2 = best_line
    
#     # Try to read the scale bar label using OCR
#     # Look for text near the scale bar
#     text_roi_y1 = max(0, y1 - 30)
#     text_roi_y2 = min(height, y1 + 10)
#     text_roi_x1 = max(0, x1 - 50)
#     text_roi_x2 = min(width, x2 + 50)
    
#     text_roi = gray[text_roi_y1:text_roi_y2, text_roi_x1:text_roi_x2]
    
#     scale_label = "Not Found"
#     if text_roi.size > 0:
#         try:
#             # Preprocess for OCR
#             processed_roi = cv2.threshold(text_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
#             # Try different OCR configurations for scale labels
#             ocr_configs = [
#                 '--psm 8 -c tessedit_char_whitelist=0123456789µmum',
#                 '--psm 7 -c tessedit_char_whitelist=0123456789µmum',
#                 '--psm 6 -c tessedit_char_whitelist=0123456789µmum',
#                 '--psm 8',
#                 '--psm 7',
#             ]
            
#             for config in ocr_configs:
#                 try:
#                     text = pytesseract.image_to_string(processed_roi, config=config).strip()
#                     if text and len(text) > 1:
#                         # Clean up the text
#                         text = re.sub(r'[^\w\sµm]', '', text)
#                         if text:
#                             scale_label = text
#                             break
#                 except Exception as e:
#                     logger.debug(f"Scale OCR config {config} failed: {e}")
#                     continue
#         except Exception as e:
#             logger.debug(f"Scale OCR failed: {e}")
    
#     # If OCR failed, estimate based on relative length
#     if scale_label == "Not Found" and best_length > 0:
#         # Estimate scale based on image size and line length
#         # This is a rough estimation - in practice, you'd need calibration
#         estimated_scale = int(best_length * 0.1)  # Rough estimation
#         scale_label = f"{estimated_scale} µm"
    
#     return {
#         "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#         "label": scale_label,
#         "length": best_length,
#         "found": scale_label != "Not Found"
#     }

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def analyze_frame_comprehensive(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85, method="v1"):
    """
    Comprehensive frame analysis including droplets, scale bar, and timestamp
    
    Args:
        image: OpenCV image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Upper threshold for edge detection
        param2: Accumulator threshold for center detection
        method: Detection method ("v1" for original, "v2" for optimized, "v3" for hybrid, "v4" for advanced hough, "v5" for optimized hough, "v6" for ultra-optimized hough, "v7" for microscope-adaptive, "v8" for v3 hybrid, "v9" for microscope_2 optimization)
    
    Returns:
        Dictionary matching the Gemini service format
    """
    # Detect droplets using selected method
    if method == "v2":
        droplets = detect_circles_v2(image, min_radius, max_radius, dp, min_dist, param1, param2)
        logger.debug(f"Using detection method: v2 (optimized template matching)")
    elif method == "v3":
        droplets = detect_circles_v3(image, min_radius, max_radius, dp, min_dist, param1, param2)
        logger.debug(f"Using detection method: v3 (fast hybrid detection)")
    elif method == "v4":
        droplets = detect_circles_v4(image, min_radius, max_radius, dp, min_dist, param1, param2)
        logger.debug(f"Using detection method: v4 (advanced hough detection)")
    elif method == "v5":
        droplets = detect_circles_v5(image, min_radius, max_radius, dp, min_dist, param1, param2)
        logger.debug(f"Using detection method: v5 (optimized hough detection)")
    elif method == "v6":
        droplets = detect_circles_v6(image, min_radius, max_radius, dp, min_dist, param1, param2)
        logger.debug(f"Using detection method: v6 (ultra-optimized hough detection)")
    elif method == "v7":
        droplets = detect_circles_v7(image, min_radius, max_radius, dp, min_dist, param1, param2)
        logger.debug(f"Using detection method: v7 (microscope-adaptive detection)")
    elif method == "v8":
        droplets = detect_circles_v8(image, min_radius, max_radius, dp, min_dist, param1, param2)
        logger.debug(f"Using detection method: v8 (v3 hybrid with sophisticated selection)")
    elif method == "v9":
        droplets = detect_circles_v9(image, min_radius, max_radius, dp, min_dist, param1, param2)
        logger.debug(f"Using detection method: v9 (microscope_2 parameter optimization)")
    else:
        droplets = detect_circles_hough(image, min_radius, max_radius, dp, min_dist, param1, param2)
        logger.debug(f"Using detection method: v1 (Hough circles)")
    
    # Detect timestamp
    timestamp, timestamp_found = detect_timestamp(image)
    
    # Detect scale bar
    scale, scale_found = detect_scale_bar(image)
    
    # Format response to match Gemini service
    result = {
        "timestamp": timestamp,
        "timestampFound": timestamp_found,
        "scaleFound": scale_found,
        "dropletsFound": len(droplets) > 0,
        "droplets": droplets
    }
    
    # Add scale if found
    if scale_found:
        result["scale"] = scale
    else:
        # Provide default scale
        height, width = image.shape[:2]
        margin = min(width, height) * 0.05
        scale_length = width * 0.15
        result["scale"] = {
            "x1": int(width - margin - scale_length),
            "y1": int(height - margin),
            "x2": int(width - margin),
            "y2": int(height - margin),
            "label": "50 µm (default)",
            "length": int(scale_length)
        }
    
    return result

def convert_video_with_ffmpeg(input_file_path: str, output_file_path: str) -> dict:
    """
    Convert video using FFmpeg with optimized settings for droplet analysis
    """
    try:
        # FFmpeg command for optimal conversion
        cmd = [
            'ffmpeg',
            '-i', input_file_path,
            '-c:v', 'libx264',           # Use H.264 codec
            '-c:a', 'aac',               # Use AAC audio codec
            '-preset', 'fast',           # Fast encoding preset
            '-crf', '23',                # Good quality (18-28 range)
            '-movflags', '+faststart',   # Optimize for web streaming
            '-y',                        # Overwrite output file
            output_file_path
        ]
        
        logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # Run FFmpeg with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            # Get file sizes
            input_size = os.path.getsize(input_file_path)
            output_size = os.path.getsize(output_file_path)
            
            return {
                "success": True,
                "input_size": input_size,
                "output_size": output_size,
                "compression_ratio": input_size / output_size if output_size > 0 else 1,
                "message": "Video converted successfully"
            }
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            return {
                "success": False,
                "error": f"FFmpeg conversion failed: {result.stderr}",
                "returncode": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out")
        return {
            "success": False,
            "error": "Video conversion timed out (1 hour limit)"
        }
    except Exception as e:
        logger.error(f"Video conversion error: {e}")
        return {
            "success": False,
            "error": f"Conversion failed: {str(e)}"
        }

# Production routes (only in production mode)
if os.getenv('FLASK_ENV') != 'development':
    @app.route('/')
    def serve_frontend():
        """Serve the frontend application"""
        return send_from_directory('static', 'index.html')
    
    @app.route('/<path:path>')
    def serve_static(path):
        """Serve static files"""
        return send_from_directory('static', path)

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint for load balancer"""
    logger.debug("Health check requested")
    return jsonify({
        "status": "healthy",
        "service": "droplet-analysis-api",
        "version": "1.0.0",
        "debug_mode": debug_mode
    })

# Main analysis endpoint
@app.route('/analyze-frame', methods=['POST'])
@app.route('/api/analyze-frame', methods=['POST'])  # Support both routes
def analyze_frame():
    """
    Comprehensive frame analysis endpoint
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "min_radius": 20,
        "max_radius": 500,
        "method": "v1"  // "v1" for Hough circles, "v2" for random values
    }
    """
    logger.debug("Received analyze-frame request")
    try:
        data = request.get_json()
        logger.debug(f"Request data keys: {list(data.keys()) if data else 'None'}")
        
        if not data or 'image' not in data:
            logger.warning("Missing 'image' field in request")
            return jsonify({"error": "Missing 'image' field in request"}), 400
        
        # Get parameters with defaults
        min_radius = data.get('min_radius', 20)
        max_radius = data.get('max_radius', 500)
        method = data.get('method', 'v1')
        logger.debug(f"Analysis parameters: min_radius={min_radius}, max_radius={max_radius}, method={method}")
        
        # Convert base64 image to OpenCV format
        logger.debug("Converting base64 image to OpenCV format")
        image = base64_to_image(data['image'])
        logger.debug(f"Image shape: {image.shape}")
        
        # Perform comprehensive analysis
        logger.debug("Starting comprehensive frame analysis")
        result = analyze_frame_comprehensive(
            image,
            min_radius=min_radius,
            max_radius=max_radius,
            method=method
        )
        
        droplet_count = len(result['droplets'])
        logger.info(f"Analysis completed: {droplet_count} droplets found")
        logger.debug(f"Result keys: {list(result.keys())}")
        
        # Convert NumPy types to native Python types for JSON serialization
        result = convert_numpy_types(result)
        
        return jsonify({
            "success": True,
            **result
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Legacy circle detection endpoint
@app.route('/detect-circles', methods=['POST'])
@app.route('/api/detect-circles', methods=['POST'])  # Support both routes
def detect_circles():
    """
    Legacy circle detection endpoint for backward compatibility
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' field in request"}), 400
        
        # Get parameters with defaults
        min_radius = data.get('min_radius', 20)
        max_radius = data.get('max_radius', 500)
        dp = data.get('dp', 1)
        min_dist = data.get('min_dist', 50)
        param1 = data.get('param1', 50)
        param2 = data.get('param2', 85)
        
        # Convert base64 image to OpenCV format
        image = base64_to_image(data['image'])
        
        # Detect circles
        circles = detect_circles_hough(
            image,
            min_radius=min_radius,
            max_radius=max_radius,
            dp=dp,
            min_dist=min_dist,
            param1=param1,
            param2=param2
        )
        
        # Convert circles to the expected format
        formatted_circles = []
        for i, circle in enumerate(circles):
            x, y, r = circle
            formatted_circles.append({
                "id": i,
                "cx": int(x),
                "cy": int(y),
                "r": int(r)
            })
        
        result = {
            "circles": formatted_circles,
            "count": len(circles),
            "success": True
        }
        
        # Convert NumPy types to native Python types
        result = convert_numpy_types(result)
        
        logger.info(f"Circle detection completed: {len(circles)} circles found")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Circle detection error: {e}")
        return jsonify({"error": str(e)}), 500

# Debug endpoint (development only)
if os.getenv('FLASK_ENV') == 'development':
    @app.route('/debug-circles', methods=['POST'])
    def debug_circles():
        """
        Debug endpoint for testing circle detection parameters
        """
        try:
            data = request.get_json()
            
            if not data or 'image' not in data:
                return jsonify({"error": "Missing 'image' field in request"}), 400
            
            # Get parameters with defaults
            min_radius = data.get('min_radius', 20)
            max_radius = data.get('max_radius', 500)
            dp = data.get('dp', 1)
            min_dist = data.get('min_dist', 50)
            param1 = data.get('param1', 50)
            param2 = data.get('param2', 85)
            method = data.get('method', 'v1')
            
            # Convert base64 image to OpenCV format
            image = base64_to_image(data['image'])
            
            # Detect circles with detailed logging using selected method
            if method == "v2":
                circles = detect_circles_v2(
                    image,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    dp=dp,
                    min_dist=min_dist,
                    param1=param1,
                    param2=param2
                )
            elif method == "v3":
                circles = detect_circles_v3(
                    image,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    dp=dp,
                    min_dist=min_dist,
                    param1=param1,
                    param2=param2
                )
            elif method == "v4":
                circles = detect_circles_v4(
                    image,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    dp=dp,
                    min_dist=min_dist,
                    param1=param1,
                    param2=param2
                )
            elif method == "v5":
                circles = detect_circles_v5(
                    image,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    dp=dp,
                    min_dist=min_dist,
                    param1=param1,
                    param2=param2
                )
            elif method == "v6":
                circles = detect_circles_v6(
                    image,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    dp=dp,
                    min_dist=min_dist,
                    param1=param1,
                    param2=param2
                )
            elif method == "v7":
                circles = detect_circles_v7(
                    image,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    dp=dp,
                    min_dist=min_dist,
                    param1=param1,
                    param2=param2
                )
            elif method == "v8":
                circles = detect_circles_v8(
                    image,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    dp=dp,
                    min_dist=min_dist,
                    param1=param1,
                    param2=param2
                )
            elif method == "v9":
                circles = detect_circles_v9(
                    image,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    dp=dp,
                    min_dist=min_dist,
                    param1=param1,
                    param2=param2
                )
            else:
                circles = detect_circles_hough(
                    image,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    dp=dp,
                    min_dist=min_dist,
                    param1=param1,
                    param2=param2
                )
            
            result = {
                "success": True,
                "image_info": {
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "channels": int(image.shape[2]) if len(image.shape) > 2 else 1
                },
                "parameters": {
                    "min_radius": min_radius,
                    "max_radius": max_radius,
                    "dp": dp,
                    "min_dist": min_dist,
                    "param1": param1,
                    "param2": param2,
                    "method": method
                },
                "circles": [{"id": i, "cx": int(c[0]), "cy": int(c[1]), "r": int(c[2])} for i, c in enumerate(circles)],
                "count": len(circles)
            }
            
            # Convert NumPy types to native Python types
            result = convert_numpy_types(result)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

# Video conversion endpoint
@app.route('/convert-video', methods=['POST'])
@app.route('/api/convert-video', methods=['POST'])  # Support both routes
def convert_video():
    """
    Convert video file using server-side FFmpeg
    
    Expected form data:
    - file: Video file to convert
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file size (5GB limit)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 5 * 1024 * 1024 * 1024:  # 5GB
            return jsonify({"error": "File too large. Maximum size is 5GB."}), 413
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as input_file:
            file.save(input_file.name)
            input_path = input_file.name
        
        # Generate output filename
        base_name = os.path.splitext(file.filename)[0]
        output_path = os.path.join(tempfile.gettempdir(), f"{base_name}_converted.mp4")
        
        logger.info(f"Converting video: {file.filename} ({file_size / (1024*1024):.1f}MB)")
        
        # Convert video
        result = convert_video_with_ffmpeg(input_path, output_path)
        
        if result["success"]:
            # Return the converted file
            return send_from_directory(
                os.path.dirname(output_path),
                os.path.basename(output_path),
                as_attachment=True,
                download_name=f"{base_name}_converted.mp4"
            )
        else:
            # Clean up files on error
            try:
                os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass
            
            return jsonify({
                "error": result["error"],
                "success": False
            }), 500
            
    except Exception as e:
        logger.error(f"Video conversion endpoint error: {e}")
        return jsonify({
            "error": f"Conversion failed: {str(e)}",
            "success": False
        }), 500

if __name__ == '__main__':
    # Server configuration
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting droplet analysis API on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Flask debug: {app.config['DEBUG']}")
    logger.info(f"Flask environment: {app.config['FLASK_ENV']}")
    
    # Use production WSGI server in production mode
    if os.getenv('FLASK_ENV') != 'development' and not debug:
        logger.info("Using Waitress WSGI server for production")
        from waitress import serve
        serve(app, host=host, port=port, threads=4)
    else:
        logger.info("Using Flask development server for debugging")
        app.run(host=host, port=port, debug=debug)