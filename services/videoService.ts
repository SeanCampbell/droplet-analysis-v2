import { videoConversionService, ConversionProgress } from './videoConversionService';
import { serverVideoConversionService, ServerConversionProgress } from './serverVideoConversionService';

export interface VideoProcessingProgress {
  stage: 'checking' | 'converting' | 'extracting' | 'complete' | 'error';
  progress: number; // 0-100
  message: string;
  extractionProgress?: { current: number; total: number };
}

export const extractFramesFromVideo = async (
  videoFile: File,
  fps: number,
  onProgress: (progress: VideoProcessingProgress) => void
): Promise<string[]> => {
  try {
    // Check if video needs conversion
    onProgress({
      stage: 'checking',
      progress: 0,
      message: 'Checking video format...',
    });

    const formatInfo = videoConversionService.checkVideoFormat(videoFile);
    let processedFile = videoFile;

    if (formatInfo.needsConversion) {
      // First, try to use the original file directly
      onProgress({
        stage: 'converting',
        progress: 0,
        message: `Checking if ${formatInfo.format} can be used directly...`,
      });

      const canUseDirectly = await videoConversionService.tryDirectPlayback(videoFile);
      
      if (canUseDirectly) {
        onProgress({
          stage: 'converting',
          progress: 100,
          message: `Using ${formatInfo.format} directly (no conversion needed)!`,
        });
        processedFile = videoFile;
      } else {
        // File is too large or needs conversion
        const fileSizeMB = videoFile.size / (1024 * 1024);
        if (fileSizeMB > 5000) {
          onProgress({
            stage: 'error',
            progress: 0,
            message: `File too large (${fileSizeMB.toFixed(1)}MB). Please use a video under 5000MB (5GB) or convert it manually to MP4.`,
          });
          throw new Error(`File too large: ${fileSizeMB.toFixed(1)}MB. Please use a video under 5000MB (5GB) or convert it manually to MP4.`);
        }

        // Check if server-side conversion is available
        const serverAvailable = await serverVideoConversionService.checkServerAvailability();
        
        if (serverAvailable) {
          onProgress({
            stage: 'converting',
            progress: 0,
            message: `Converting ${formatInfo.format} to MP4 using server (faster)...`,
          });

          try {
            const result = await serverVideoConversionService.convertVideo(videoFile, (progress: ServerConversionProgress) => {
              onProgress({
                stage: 'converting',
                progress: progress.progress,
                message: progress.message,
              });
            });

            if (result.success && result.file) {
              processedFile = result.file;
              onProgress({
                stage: 'converting',
                progress: 100,
                message: `Server conversion complete! Compression ratio: ${result.compressionRatio?.toFixed(1)}x`,
              });
            } else {
              throw new Error(result.error || 'Server conversion failed');
            }
          } catch (serverError) {
            console.warn('Server conversion failed, falling back to client-side:', serverError);
            onProgress({
              stage: 'converting',
              progress: 0,
              message: `Server conversion failed, using client-side conversion...`,
            });
            
            // Fallback to client-side conversion
            try {
              processedFile = await videoConversionService.convertVideo(
                videoFile,
                (conversionProgress: ConversionProgress) => {
                  onProgress({
                    stage: 'converting',
                    progress: conversionProgress.progress,
                    message: conversionProgress.message,
                  });
                }
              );
            } catch (conversionError) {
              onProgress({
                stage: 'error',
                progress: 0,
                message: `Conversion failed: ${conversionError instanceof Error ? conversionError.message : 'Unknown error'}. Please try converting the video manually to MP4.`,
              });
              throw new Error(`Video conversion failed: ${conversionError instanceof Error ? conversionError.message : 'Unknown error'}. Please try converting the video manually to MP4.`);
            }
          }
        } else {
          onProgress({
            stage: 'converting',
            progress: 0,
            message: `Converting ${formatInfo.format} to MP4 using client-side (this may take a while)...`,
          });

          try {
            processedFile = await videoConversionService.convertVideo(
              videoFile,
              (conversionProgress: ConversionProgress) => {
                onProgress({
                  stage: 'converting',
                  progress: conversionProgress.progress,
                  message: conversionProgress.message,
                });
              }
            );
          } catch (conversionError) {
            onProgress({
              stage: 'error',
              progress: 0,
              message: `Conversion failed: ${conversionError instanceof Error ? conversionError.message : 'Unknown error'}. Please try converting the video manually to MP4.`,
            });
            throw new Error(`Video conversion failed: ${conversionError instanceof Error ? conversionError.message : 'Unknown error'}. Please try converting the video manually to MP4.`);
          }
        }
      }
    }

    // Extract frames from the processed video
    return await extractFramesFromProcessedVideo(processedFile, fps, onProgress);

  } catch (error) {
    onProgress({
      stage: 'error',
      progress: 0,
      message: error instanceof Error ? error.message : 'Unknown error occurred',
    });
    throw error;
  }
};

const extractFramesFromProcessedVideo = (
  videoFile: File,
  fps: number,
  onProgress: (progress: VideoProcessingProgress) => void
): Promise<string[]> => {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const frames: string[] = [];
    let frameCounter = 0;
    let totalFrames = 0;

    if (!context) {
      return reject(new Error('Could not get canvas context'));
    }

    const videoUrl = URL.createObjectURL(videoFile);

    const cleanup = () => {
      video.removeEventListener('seeked', seekAndCapture);
      video.removeEventListener('canplay', onCanPlay);
      video.removeEventListener('error', onError);
      URL.revokeObjectURL(videoUrl);
    };

    const seekAndCapture = () => {
        frameCounter++;
        onProgress({
          stage: 'extracting',
          progress: Math.round((frameCounter / totalFrames) * 100),
          message: `Extracting frame ${frameCounter} of ${totalFrames}...`,
          extractionProgress: { current: frameCounter, total: totalFrames },
        });
        
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        frames.push(dataUrl.split(',')[1]);

        const nextTime = video.currentTime + 1 / fps;
        if (nextTime <= video.duration) {
            video.currentTime = nextTime;
        } else {
            cleanup();
            onProgress({
              stage: 'complete',
              progress: 100,
              message: `Successfully extracted ${frames.length} frames`,
            });
            resolve(frames);
        }
    };

    const onCanPlay = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      totalFrames = Math.floor(video.duration * fps);
      
      onProgress({
        stage: 'extracting',
        progress: 0,
        message: `Starting frame extraction (${totalFrames} frames to extract)...`,
        extractionProgress: { current: 0, total: totalFrames },
      });
      
      video.addEventListener('seeked', seekAndCapture);
      video.currentTime = 0; // Start the process
    };

    const onError = () => {
      let errorMsg = 'Error loading video file.';
      if (video.error) {
        switch (video.error.code) {
          case video.error.MEDIA_ERR_ABORTED:
            errorMsg = 'The video loading was aborted.';
            break;
          case video.error.MEDIA_ERR_NETWORK:
            errorMsg = 'A network error caused the video download to fail.';
            break;
          case video.error.MEDIA_ERR_DECODE:
            errorMsg = 'The video could not be decoded. The file may be corrupt or use a format not supported by your browser.';
            break;
          case video.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
            errorMsg = `The video format (${videoFile.type || 'unknown'}) is not supported. Please try converting the video to a standard web format like H.264 MP4.`;
            break;
          default:
            errorMsg = 'An unknown error occurred while loading the video.';
            break;
        }
      }
      cleanup();
      onProgress({
        stage: 'error',
        progress: 0,
        message: errorMsg,
      });
      reject(new Error(errorMsg));
    };

    video.addEventListener('canplay', onCanPlay, { once: true });
    video.addEventListener('error', onError);

    video.src = videoUrl;
    video.muted = true;
    video.preload = 'auto';
  });
};