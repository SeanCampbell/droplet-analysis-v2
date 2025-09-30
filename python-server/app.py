from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import re
import pytesseract

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins=['http://localhost:8888', 'http://127.0.0.1:8888'])

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

def analyze_frame_comprehensive(image, min_radius=20, max_radius=500, dp=1, min_dist=50, param1=50, param2=85):
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
    
    Returns:
        Dictionary matching the Gemini service format
    """
    # Detect droplets
    droplets = detect_circles_hough(image, min_radius, max_radius, dp, min_dist, param1, param2)
    
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "hough-circle-detection"})

@app.route('/debug-circles', methods=['POST'])
def debug_circles():
    """
    Debug endpoint for circle detection with detailed information
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "min_radius": 20,
        "max_radius": 150
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' field in request"}), 400
        
        # Get parameters with defaults
        min_radius = data.get('min_radius', 20)
        max_radius = data.get('max_radius', 500)
        
        # Convert base64 image to OpenCV format
        image = base64_to_image(data['image'])
        
        # Get image info
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Try different parameter combinations
        param_combinations = [
            {"dp": 1, "min_dist": 50, "param1": 50, "param2": 85, "name": "default"},
            {"dp": 1, "min_dist": 25, "param1": 25, "param2": 42, "name": "relaxed"},
            {"dp": 1, "min_dist": 17, "param1": 17, "param2": 28, "name": "very_relaxed"},
            {"dp": 2, "min_dist": 50, "param1": 50, "param2": 85, "name": "dp2"},
            {"dp": 1, "min_dist": 10, "param1": 10, "param2": 20, "name": "minimal"},
        ]
        
        results = []
        best_result = None
        best_count = 0
        
        for params in param_combinations:
            circles = cv2.HoughCircles(
                blurred,
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
                detected_circles = []
                for i, (x, y, r) in enumerate(circles):
                    detected_circles.append({
                        "id": i,
                        "cx": int(x),
                        "cy": int(y),
                        "r": int(r)
                    })
                
                if len(circles) > best_count:
                    best_count = len(circles)
                    best_result = detected_circles
                
                results.append({
                    "params": params["name"],
                    "circles": detected_circles,
                    "count": len(circles)
                })
            else:
                results.append({
                    "params": params["name"],
                    "circles": [],
                    "count": 0
                })
        
        return jsonify({
            "success": True,
            "image_info": {
                "width": width,
                "height": height,
                "grayscale_range": [int(gray.min()), int(gray.max())],
                "blurred_range": [int(blurred.min()), int(blurred.max())]
            },
            "parameter_tests": results,
            "best_result": best_result,
            "best_count": best_count
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    """
    Comprehensive frame analysis matching Gemini service format
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "min_radius": 20,
        "max_radius": 150,
        "dp": 1,
        "min_dist": 50,
        "param1": 50,
        "param2": 85
    }
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
        
        # Perform comprehensive analysis
        result = analyze_frame_comprehensive(
            image,
            min_radius=min_radius,
            max_radius=max_radius,
            dp=dp,
            min_dist=min_dist,
            param1=param1,
            param2=param2
        )
        
        return jsonify({
            "success": True,
            **result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/detect-circles', methods=['POST'])
def detect_circles():
    """
    Detect circles in an image using Hough Circle Transform
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "min_radius": 20,
        "max_radius": 150,
        "dp": 1,
        "min_dist": 50,
        "param1": 50,
        "param2": 85
    }
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
        
        return jsonify({
            "success": True,
            "circles": circles,
            "count": len(circles)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/detect-circles-advanced', methods=['POST'])
def detect_circles_advanced():
    """
    Advanced circle detection with multiple preprocessing options
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "preprocessing": {
            "blur_kernel": 9,
            "blur_sigma": 2,
            "threshold_type": "adaptive",
            "threshold_value": 127
        },
        "hough_params": {
            "min_radius": 20,
            "max_radius": 150,
            "dp": 1,
            "min_dist": 50,
            "param1": 50,
            "param2": 85
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' field in request"}), 400
        
        # Get preprocessing parameters
        preprocessing = data.get('preprocessing', {})
        blur_kernel = preprocessing.get('blur_kernel', 9)
        blur_sigma = preprocessing.get('blur_sigma', 2)
        threshold_type = preprocessing.get('threshold_type', 'adaptive')
        threshold_value = preprocessing.get('threshold_value', 127)
        
        # Get Hough parameters
        hough_params = data.get('hough_params', {})
        min_radius = hough_params.get('min_radius', 20)
        max_radius = hough_params.get('max_radius', 500)
        dp = hough_params.get('dp', 1)
        min_dist = hough_params.get('min_dist', 50)
        param1 = hough_params.get('param1', 50)
        param2 = hough_params.get('param2', 85)
        
        # Convert base64 image to OpenCV format
        image = base64_to_image(data['image'])
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        if blur_kernel > 0:
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), blur_sigma)
        
        # Apply thresholding if specified
        if threshold_type == 'binary':
            _, gray = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_type == 'adaptive':
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        detected_circles = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Sort circles by radius (largest first) and take top 2
            circles = sorted(circles, key=lambda x: x[2], reverse=True)[:2]
            
            for i, (x, y, r) in enumerate(circles):
                detected_circles.append({
                    "id": i,
                    "cx": int(x),
                    "cy": int(y),
                    "r": int(r)
                })
        
        return jsonify({
            "success": True,
            "circles": detected_circles,
            "count": len(detected_circles),
            "parameters_used": {
                "preprocessing": preprocessing,
                "hough_params": hough_params
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

