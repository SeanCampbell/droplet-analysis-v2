# Video Conversion Guide

## Video Conversion Methods

The application now supports **two conversion methods**:

### 🚀 Server-Side Conversion (Recommended)
- **Fast FFmpeg processing** on the server
- **Supports files up to 5GB**
- **Multi-threaded processing**
- **Automatic fallback** to client-side if server unavailable

### 🌐 Client-Side Conversion (Fallback)
- Uses FFmpeg.js (WebAssembly) in your browser
- **Slower processing** but works offline
- **Memory constraints**: Limited by browser memory allocation
- **Single-threaded**: Cannot utilize multiple CPU cores effectively

## Recommended Solutions

### 1. File Size Guidelines
- **Target size**: Under 100MB for best performance
- **Maximum size**: 5GB (conversion may take considerable time)
- **Tips**:
  - Trim your video to only the relevant sections
  - Reduce resolution (e.g., 720p instead of 1080p)
  - Lower frame rate if possible

### 2. Pre-convert to MP4 (Best Performance)
Convert your video to MP4 before uploading:

**Using FFmpeg (command line):**
```bash
ffmpeg -i input.avi -c:v libx264 -c:a aac -preset fast -crf 23 output.mp4
```

**Using online converters:**
- CloudConvert
- Online-Convert
- Convertio

**Using desktop software:**
- HandBrake (free)
- VLC Media Player
- Adobe Media Encoder

### 3. Supported Formats (No Conversion Needed)
These formats work directly without conversion:
- **MP4** (H.264 codec)
- **WebM**
- **OGG**

### 4. Formats That Need Conversion
These will be automatically converted (but slowly):
- AVI
- MOV (QuickTime)
- WMV
- FLV
- MKV
- 3GP
- M4V

## Performance Tips

### For Large Files:
1. **Pre-convert to MP4** using desktop software
2. **Reduce file size** by lowering quality/resolution
3. **Trim unnecessary parts** of the video
4. **Use MP4 format** whenever possible

### For Best Results:
- Use MP4 files under 50MB
- Ensure good lighting in your droplet videos
- Use consistent frame rates
- Avoid very long videos (extract relevant sections)

## Troubleshooting

### "413 Request Entity Too Large" Error
- **Server configuration issue**: The server needs to be restarted with updated settings
- **Solution**: Restart the development server using `./restart-dev-server.sh`
- **Alternative**: The app will automatically fall back to client-side conversion
- **For production**: Ensure nginx and Flask are configured for 5GB uploads

### "File too large" Error
- Reduce file size to under 5GB
- Convert to MP4 manually before uploading
- Trim video to relevant sections only

### "Conversion failed" Error
- Try converting the file manually to MP4
- Check if the file is corrupted
- Use a different video file

### Very Slow Conversion
- This is normal for large files
- Consider using a smaller file
- Pre-convert to MP4 for better performance

## Alternative Workflow

If conversion is too slow, try this workflow:

1. **Convert manually** using desktop software (HandBrake, VLC, etc.)
2. **Save as MP4** with H.264 codec
3. **Upload the MP4** directly to the application
4. **Skip conversion** entirely

This approach is much faster and more reliable than browser-based conversion.
