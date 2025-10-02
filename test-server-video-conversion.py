#!/usr/bin/env python3
"""
Test script for server-side video conversion API
"""

import requests
import os
import sys
import tempfile
from pathlib import Path

def test_video_conversion():
    """Test the video conversion endpoint"""
    
    # API endpoint
    api_url = "http://localhost:5001/api/convert-video"
    
    print("🧪 Testing Server-Side Video Conversion API")
    print("=" * 50)
    
    # Check if we have a test video file
    test_video_path = None
    possible_paths = [
        "test-video.avi",
        "test-video.mov", 
        "test-video.mkv",
        "sample.avi",
        "sample.mov"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            test_video_path = path
            break
    
    if not test_video_path:
        print("❌ No test video file found. Please place a test video file (AVI, MOV, MKV) in the current directory.")
        print("   Supported test files:", ", ".join(possible_paths))
        return False
    
    print(f"📹 Using test video: {test_video_path}")
    file_size = os.path.getsize(test_video_path)
    print(f"📊 File size: {file_size / (1024*1024):.1f} MB")
    
    # Test the conversion
    try:
        print(f"\n🚀 Sending conversion request to {api_url}")
        
        with open(test_video_path, 'rb') as f:
            files = {'file': (test_video_path, f, 'video/avi')}
            
            response = requests.post(api_url, files=files, timeout=300)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            # Save the converted file
            output_filename = f"converted_{Path(test_video_path).stem}.mp4"
            
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            
            output_size = os.path.getsize(output_filename)
            compression_ratio = file_size / output_size if output_size > 0 else 1
            
            print(f"✅ Conversion successful!")
            print(f"📁 Output file: {output_filename}")
            print(f"📊 Output size: {output_size / (1024*1024):.1f} MB")
            print(f"🗜️  Compression ratio: {compression_ratio:.1f}x")
            
            # Clean up
            try:
                os.remove(output_filename)
                print(f"🧹 Cleaned up output file")
            except:
                pass
                
            return True
            
        else:
            print(f"❌ Conversion failed!")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (5 minutes)")
        return False
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - is the server running?")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

def test_health_check():
    """Test the health check endpoint"""
    
    print("\n🏥 Testing Health Check")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ Health check passed")
            data = response.json()
            print(f"📊 Service: {data.get('service', 'Unknown')}")
            print(f"🔧 Debug mode: {data.get('debug_mode', 'Unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🎬 Server-Side Video Conversion Test")
    print("=" * 50)
    
    # Test health check first
    health_ok = test_health_check()
    
    if health_ok:
        # Test video conversion
        conversion_ok = test_video_conversion()
        
        if conversion_ok:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n💥 Video conversion test failed!")
            sys.exit(1)
    else:
        print("\n💥 Health check failed - server may not be running!")
        print("💡 Start the server with: python python-server/app.py")
        sys.exit(1)
