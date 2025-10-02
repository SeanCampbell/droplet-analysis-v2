import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile, toBlobURL } from '@ffmpeg/util';

export interface ConversionProgress {
  progress: number; // 0-100
  stage: 'loading' | 'converting' | 'complete' | 'error';
  message: string;
}

export interface VideoFormatInfo {
  isSupported: boolean;
  needsConversion: boolean;
  format: string;
  codec?: string;
}

class VideoConversionService {
  private ffmpeg: FFmpeg | null = null;
  private isInitialized = false;

  /**
   * Initialize FFmpeg instance
   */
  private async initializeFFmpeg(): Promise<void> {
    if (this.isInitialized && this.ffmpeg) {
      return;
    }

    this.ffmpeg = new FFmpeg();
    
    // Set up progress callback
    this.ffmpeg.on('progress', ({ progress }) => {
      // Progress is 0-1, convert to 0-100
      console.log(`FFmpeg progress: ${Math.round(progress * 100)}%`);
    });

    // Load FFmpeg from CDN
    const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd';
    await this.ffmpeg.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    });

    this.isInitialized = true;
  }

  /**
   * Try to use the original file directly (fallback for when conversion is too slow)
   */
  public async tryDirectPlayback(file: File): Promise<boolean> {
    return new Promise((resolve) => {
      const video = document.createElement('video');
      const timeout = setTimeout(() => {
        video.remove();
        resolve(false);
      }, 5000); // 5 second timeout

      video.onloadedmetadata = () => {
        clearTimeout(timeout);
        video.remove();
        resolve(true);
      };

      video.onerror = () => {
        clearTimeout(timeout);
        video.remove();
        resolve(false);
      };

      video.src = URL.createObjectURL(file);
      video.load();
    });
  }

  /**
   * Check if a video format is supported by the browser
   */
  public checkVideoFormat(file: File): VideoFormatInfo {
    const mimeType = file.type.toLowerCase();
    const fileName = file.name.toLowerCase();
    
    // Web-compatible formats that don't need conversion
    const supportedFormats = [
      'video/mp4',
      'video/webm',
      'video/ogg',
    ];

    // Formats that definitely need conversion (including AVI for maximum compatibility)
    const unsupportedFormats = [
      'video/quicktime', // .mov
      'video/avi', // .avi (all variants - convert for maximum compatibility)
      'video/x-msvideo', // .avi (older format)
      'video/x-ms-wmv', // .wmv
      'video/x-flv', // .flv
      'video/x-matroska', // .mkv
      'video/3gpp', // .3gp
      'video/x-m4v', // .m4v
    ];

    // Check by MIME type first
    if (supportedFormats.includes(mimeType)) {
      return {
        isSupported: true,
        needsConversion: false,
        format: mimeType,
      };
    }

    if (unsupportedFormats.includes(mimeType)) {
      return {
        isSupported: false,
        needsConversion: true,
        format: mimeType,
      };
    }

    // Check by file extension if MIME type is not recognized
    const extension = fileName.split('.').pop()?.toLowerCase();
    const supportedExtensions = ['mp4', 'webm', 'ogg'];
    const unsupportedExtensions = ['mov', 'avi', 'wmv', 'flv', 'mkv', '3gp', 'm4v'];

    if (extension && supportedExtensions.includes(extension)) {
      return {
        isSupported: true,
        needsConversion: false,
        format: extension,
      };
    }

    if (extension && unsupportedExtensions.includes(extension)) {
      return {
        isSupported: false,
        needsConversion: true,
        format: extension,
      };
    }

    // Default to unsupported if we can't determine
    return {
      isSupported: false,
      needsConversion: true,
      format: mimeType || 'unknown',
    };
  }

  /**
   * Convert video to web-compatible MP4 format with optimizations
   */
  public async convertVideo(
    file: File,
    onProgress?: (progress: ConversionProgress) => void
  ): Promise<File> {
    // Check file size and warn if too large
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > 5000) {
      onProgress?.({
        progress: 0,
        stage: 'error',
        message: `File too large (${fileSizeMB.toFixed(1)}MB). Please use a video under 5000MB (5GB) for conversion.`,
      });
      throw new Error(`File too large: ${fileSizeMB.toFixed(1)}MB. Maximum supported size is 5000MB (5GB).`);
    }

    if (!this.ffmpeg) {
      await this.initializeFFmpeg();
    }

    if (!this.ffmpeg) {
      throw new Error('Failed to initialize FFmpeg');
    }

    try {
      // Update progress
      onProgress?.({
        progress: 0,
        stage: 'loading',
        message: `Loading video file (${fileSizeMB.toFixed(1)}MB)...`,
      });

      // Write input file to FFmpeg
      const inputFileName = 'input' + this.getFileExtension(file.name);
      const outputFileName = 'output.mp4';

      await this.ffmpeg.writeFile(inputFileName, await fetchFile(file));

      onProgress?.({
        progress: 10,
        stage: 'converting',
        message: 'Converting video to MP4 (this may take a while for large files)...',
      });

      // Optimized conversion settings for faster processing
      await this.ffmpeg.exec([
        '-i', inputFileName,
        '-c:v', 'libx264', // H.264 video codec
        '-c:a', 'aac', // AAC audio codec
        '-preset', 'ultrafast', // Fastest encoding (was 'fast')
        '-crf', '28', // Lower quality but faster (was 23)
        '-vf', 'scale=640:480', // Scale down to reduce processing time
        '-r', '15', // Reduce frame rate to 15fps for faster processing
        '-movflags', '+faststart', // Optimize for web streaming
        '-y', // Overwrite output file
        outputFileName
      ]);

      onProgress?.({
        progress: 90,
        stage: 'converting',
        message: 'Finalizing conversion...',
      });

      // Read the converted file
      const data = await this.ffmpeg.readFile(outputFileName);
      const uint8Array = data as Uint8Array;
      const arrayBuffer = uint8Array.buffer instanceof ArrayBuffer 
        ? uint8Array.buffer.slice(uint8Array.byteOffset, uint8Array.byteOffset + uint8Array.byteLength)
        : new ArrayBuffer(uint8Array.length);
      const blob = new Blob([arrayBuffer], { type: 'video/mp4' });

      // Clean up files
      await this.ffmpeg.deleteFile(inputFileName);
      await this.ffmpeg.deleteFile(outputFileName);

      onProgress?.({
        progress: 100,
        stage: 'complete',
        message: 'Conversion complete!',
      });

      // Create a new File object with the converted data
      const convertedFile = new File([blob], this.getConvertedFileName(file.name), {
        type: 'video/mp4',
      });

      return convertedFile;

    } catch (error) {
      onProgress?.({
        progress: 0,
        stage: 'error',
        message: `Conversion failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
      throw error;
    }
  }

  /**
   * Get file extension from filename
   */
  private getFileExtension(filename: string): string {
    const extension = filename.split('.').pop()?.toLowerCase();
    return extension ? `.${extension}` : '';
  }

  /**
   * Generate converted filename
   */
  private getConvertedFileName(originalName: string): string {
    const nameWithoutExt = originalName.replace(/\.[^/.]+$/, '');
    return `${nameWithoutExt}_converted.mp4`;
  }

  /**
   * Check if FFmpeg is available and ready
   */
  public isReady(): boolean {
    return this.isInitialized && this.ffmpeg !== null;
  }

  /**
   * Get supported formats information
   */
  public getSupportedFormats(): { supported: string[]; unsupported: string[] } {
    return {
      supported: [
        'MP4 (H.264)',
        'WebM',
        'OGG',
      ],
      unsupported: [
        'AVI (all variants - auto-converted)',
        'MOV (QuickTime)',
        'WMV',
        'FLV',
        'MKV',
        '3GP',
        'M4V',
      ],
    };
  }
}

// Export singleton instance
export const videoConversionService = new VideoConversionService();
