# Hough Circle Detection Python Server

This Python server provides Hough circle detection capabilities for the droplet analysis application using OpenCV.

## Features

- **Basic Circle Detection**: Simple Hough circle detection with configurable parameters
- **Advanced Circle Detection**: Enhanced detection with preprocessing options
- **Base64 Image Support**: Accepts images in base64 format
- **RESTful API**: Clean HTTP endpoints for integration
- **Health Check**: Monitoring endpoint for service status
- **CORS Support**: Configured for cross-origin requests from the frontend

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Start the Server

```bash
python start_server.py
```

Or directly:
```bash
python app.py
```

The server will start on `http://localhost:5001`

### API Endpoints

#### Health Check
```
GET /health
```

#### Basic Circle Detection
```
POST /detect-circles
```

#### Debug Circle Detection
```
POST /debug-circles
```

#### Comprehensive Frame Analysis
```
POST /analyze-frame
```

**Request Body:**
```json
{
    "image": "base64_encoded_image",
    "min_radius": 20,
    "max_radius": 150,
    "dp": 1,
    "min_dist": 50,
    "param1": 50,
    "param2": 85
}
```

**Response:**
```json
{
    "success": true,
    "circles": [
        {
            "id": 0,
            "cx": 100,
            "cy": 150,
            "r": 45
        }
    ],
    "count": 1
}
```

#### Comprehensive Frame Analysis
```
POST /analyze-frame
```

**Request Body:**
```json
{
    "image": "base64_encoded_image",
    "min_radius": 20,
    "max_radius": 150,
    "dp": 1,
    "min_dist": 50,
    "param1": 50,
    "param2": 85
}
```

**Response:**
```json
{
    "success": true,
    "timestamp": "0:01:23.456",
    "timestampFound": true,
    "scaleFound": true,
    "dropletsFound": true,
    "droplets": [
        {
            "id": 0,
            "cx": 150,
            "cy": 150,
            "r": 50
        }
    ],
    "scale": {
        "x1": 450,
        "y1": 350,
        "x2": 550,
        "y2": 350,
        "label": "50 µm",
        "length": 100
    }
}
```

#### Advanced Circle Detection
```
POST /detect-circles-advanced
```

**Request Body:**
```json
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
```

## Parameters

### Hough Circle Parameters
- `min_radius`: Minimum circle radius to detect
- `max_radius`: Maximum circle radius to detect
- `dp`: Inverse ratio of accumulator resolution
- `min_dist`: Minimum distance between circle centers
- `param1`: Upper threshold for edge detection
- `param2`: Accumulator threshold for center detection

### Preprocessing Parameters
- `blur_kernel`: Gaussian blur kernel size (0 to disable)
- `blur_sigma`: Gaussian blur sigma value
- `threshold_type`: Threshold type ("binary", "adaptive", or "none")
- `threshold_value`: Threshold value for binary thresholding

## Integration

This server is designed to replace the Web Worker-based Hough circle detection in the droplet analysis application. The frontend will send base64-encoded images to this server and receive detected circles in the same format as the original implementation.

## Port Configuration

The Python server runs on port 5001 by default. If you need to change this port, update the following files:
- `app.py` - Change the port in the `app.run()` call
- `start_server.py` - Update the port and print statements
- `services/houghCircleService.ts` - Update the `PYTHON_API_BASE_URL` constant

## CORS Configuration

The server is configured to allow cross-origin requests from the frontend running on port 8888. The CORS configuration in `app.py` allows requests from:
- `http://localhost:8888`
- `http://127.0.0.1:8888`

If you change the frontend port, update the CORS origins in the `CORS()` configuration in `app.py`.

