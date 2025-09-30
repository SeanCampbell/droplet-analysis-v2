export const extractFramesFromVideo = (
  videoFile: File,
  fps: number,
  onProgress: (progress: { current: number; total: number }) => void
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
        onProgress({ current: frameCounter, total: totalFrames });
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        frames.push(dataUrl.split(',')[1]);

        const nextTime = video.currentTime + 1 / fps;
        if (nextTime <= video.duration) {
            video.currentTime = nextTime;
        } else {
            cleanup();
            resolve(frames);
        }
    };

    const onCanPlay = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      totalFrames = Math.floor(video.duration * fps);
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
      reject(new Error(errorMsg));
    };

    video.addEventListener('canplay', onCanPlay, { once: true });
    video.addEventListener('error', onError);

    video.src = videoUrl;
    video.muted = true;
    video.preload = 'auto';
  });
};