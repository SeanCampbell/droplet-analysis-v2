<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Droplet Analysis Application

A sophisticated application for analyzing droplet images using both AI-powered detection (Gemini) and computer vision techniques (Hough Circle Transform). The application supports video frame extraction, real-time analysis, and CSV export functionality.

## Features

- **Dual Detection Methods**: 
  - AI-powered droplet detection using Google Gemini
  - Computer vision-based Hough circle detection via Python API
- **Video Processing**: Extract and analyze frames from video files with automatic format conversion
- **Interactive Canvas**: Manual adjustment of detected droplets and scale bars
- **Export Functionality**: Export analysis results to CSV format
- **Real-time Analysis**: Process multiple frames with progress tracking

## Architecture

The application consists of:
- **Frontend**: React-based web application (TypeScript)
- **Python API Server**: Flask-based server providing Hough circle detection
- **Fallback System**: Web Worker implementation for offline functionality

## Prerequisites

- **Node.js** (v16 or higher)
- **Python 3** (v3.8 or higher)
- **pip** (Python package manager)

## Quick Start

### Option 1: Automated Setup (Recommended)

**For macOS/Linux:**
```bash
./start-dev.sh
```

**For Windows:**
```cmd
start-dev.bat
```

This script will:
- Check prerequisites
- Install dependencies for both frontend and Python server
- Start both servers simultaneously
- Provide health check URLs

### Option 2: Manual Setup

#### 1. Install Frontend Dependencies
```bash
npm install
```

#### 2. Install Python Dependencies
```bash
cd python-server
pip install -r requirements.txt
cd ..
```

#### 3. Start Python API Server
```bash
cd python-server
python app.py
```

#### 4. Start Frontend Development Server
```bash
npm run dev
```

## Configuration

### Environment Variables

Create a `.env.local` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Python API Configuration

The Python server runs on `http://localhost:5001` by default and is configured with CORS support for the frontend. You can modify the URL in `services/houghCircleService.ts` if needed.

## API Endpoints

### Python API Server (Port 5001)

- `GET /health` - Health check endpoint
- `POST /detect-circles` - Basic Hough circle detection
- `POST /detect-circles-advanced` - Advanced detection with preprocessing options

### Frontend Application (Port 8888)

- Main application interface
- Video upload and processing
- Interactive analysis tools

## Usage

1. **Upload Video**: Select a video file containing droplet images (supports MP4, WebM, OGG directly; AVI, MOV, WMV, FLV, MKV, 3GP, M4V auto-converted)
2. **Automatic Conversion**: Unsupported formats are automatically converted to MP4 using client-side FFmpeg
3. **Choose Detection Method**: Select between "Gemini" (AI) or "Hough" (Computer Vision)
4. **Analyze**: Click "Analyze Video" to process all frames
5. **Review Results**: Use the interactive canvas to adjust detections
6. **Export**: Download results as CSV for further analysis

### Supported Video Formats

**Direct Support (No conversion needed):**
- MP4 (H.264)
- WebM
- OGG

**Automatic Conversion:**
- AVI (all variants - converted for maximum compatibility)
- MOV (QuickTime)
- WMV
- FLV
- MKV
- 3GP
- M4V

The application uses FFmpeg.js for client-side video conversion, ensuring compatibility with a wide range of video formats without requiring server-side processing.

## Development

### Project Structure

```
droplet-analysis-v2/
├── components/          # React components
├── services/           # API services and workers
├── python-server/      # Python Flask API server
├── types.ts           # TypeScript type definitions
└── README.md          # This file
```

### Adding New Features

1. **Frontend Changes**: Modify React components in `components/`
2. **API Changes**: Update Python server in `python-server/`
3. **Service Integration**: Modify services in `services/`

## Troubleshooting

### Common Issues

1. **Port Already in Use**: 
   - Stop existing services on ports 8888 and 5001
   - Use `lsof -i :8888` (macOS/Linux) or `netstat -an | find "8888"` (Windows)

2. **Python Dependencies**:
   - Ensure Python 3.8+ is installed
   - Use `pip3` instead of `pip` if needed
   - Check virtual environment activation

3. **Node.js Dependencies**:
   - Clear node_modules and reinstall: `rm -rf node_modules && npm install`
   - Check Node.js version compatibility

### Health Checks

- Frontend: http://localhost:8888
- Python API: http://localhost:5001/health

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both frontend and Python API
5. Submit a pull request

## License

This project is licensed under the MIT License.
