export interface ServerConversionProgress {
  stage: 'uploading' | 'converting' | 'downloading' | 'complete' | 'error';
  progress: number; // 0-100
  message: string;
}

export interface ServerConversionResult {
  success: boolean;
  file?: File;
  error?: string;
  inputSize?: number;
  outputSize?: number;
  compressionRatio?: number;
}

class ServerVideoConversionService {
  private apiBaseUrl: string;

  constructor() {
    // Use relative URLs for production (Docker), absolute for development
    this.apiBaseUrl = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5001';
  }

  /**
   * Convert video using server-side FFmpeg
   */
  public async convertVideo(
    file: File,
    onProgress?: (progress: ServerConversionProgress) => void
  ): Promise<ServerConversionResult> {
    try {
      // Check file size (5GB limit)
      const fileSizeMB = file.size / (1024 * 1024);
      if (fileSizeMB > 5000) {
        const error = `File too large: ${fileSizeMB.toFixed(1)}MB. Maximum supported size is 5000MB (5GB).`;
        onProgress?.({
          stage: 'error',
          progress: 0,
          message: error
        });
        return { success: false, error };
      }

      // Upload file
      onProgress?.({
        stage: 'uploading',
        progress: 0,
        message: `Uploading ${file.name} (${fileSizeMB.toFixed(1)}MB)...`
      });

      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${this.apiBaseUrl}/api/convert-video`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `Server error: ${response.status} ${response.statusText}`;
        
        // Handle specific error codes
        if (response.status === 413) {
          errorMessage = `File too large for server processing (${fileSizeMB.toFixed(1)}MB). The server may need to be restarted with updated configuration. Falling back to client-side conversion.`;
        } else if (response.status === 504) {
          errorMessage = `Server timeout - file may be too large or server is overloaded. Falling back to client-side conversion.`;
        } else {
          try {
            const errorData = JSON.parse(errorText);
            errorMessage = errorData.error || errorMessage;
          } catch {
            // Use default error message if JSON parsing fails
          }
        }

        onProgress?.({
          stage: 'error',
          progress: 0,
          message: errorMessage
        });

        return { success: false, error: errorMessage };
      }

      // Download converted file
      onProgress?.({
        stage: 'downloading',
        progress: 90,
        message: 'Downloading converted video...'
      });

      const blob = await response.blob();
      const convertedFile = new File([blob], this.getConvertedFileName(file.name), {
        type: 'video/mp4'
      });

      onProgress?.({
        stage: 'complete',
        progress: 100,
        message: `Conversion complete! File size: ${(convertedFile.size / (1024 * 1024)).toFixed(1)}MB`
      });

      return {
        success: true,
        file: convertedFile,
        inputSize: file.size,
        outputSize: convertedFile.size,
        compressionRatio: file.size / convertedFile.size
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      
      onProgress?.({
        stage: 'error',
        progress: 0,
        message: `Conversion failed: ${errorMessage}`
      });

      return { success: false, error: errorMessage };
    }
  }

  /**
   * Check if server-side conversion is available
   */
  public async checkServerAvailability(): Promise<boolean> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/health`, {
        method: 'GET',
        timeout: 5000
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Get converted filename
   */
  private getConvertedFileName(originalName: string): string {
    const baseName = originalName.replace(/\.[^/.]+$/, '');
    return `${baseName}_converted.mp4`;
  }
}

export const serverVideoConversionService = new ServerVideoConversionService();
