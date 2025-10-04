import React, { useState, useCallback, useMemo } from 'react';
import JSZip from 'jszip';
import type { FrameAnalysis, Droplet } from './types';
import { extractFramesFromVideo, VideoProcessingProgress } from './services/videoService';
import { detectCirclesWithHough, analyzeFrameWithHough } from './services/houghCircleService';
import { FrameCanvas } from './components/FrameCanvas';

const App: React.FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [frames, setFrames] = useState<string[]>([]);
  const [analyses, setAnalyses] = useState<FrameAnalysis[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [status, setStatus] = useState<'idle' | 'extracting' | 'analyzing' | 'ready' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [extractionProgress, setExtractionProgress] = useState({ current: 0, total: 0 });
  const [videoProcessingStage, setVideoProcessingStage] = useState<string>('');
  const [videoProcessingMessage, setVideoProcessingMessage] = useState<string>('');
  const [imageDimensions, setImageDimensions] = useState({ width: 1280, height: 720 });
  const [frameInterval, setFrameInterval] = useState(120);
  const [detectionMethod, setDetectionMethod] = useState<'v1' | 'v2' | 'v3' | 'v4' | 'v5' | 'v6' | 'v7' | 'v8' | 'v9'>('v9');
  const [view, setView] = useState({ zoom: 1, pan: { x: 0, y: 0 } });
  const [showCSVModal, setShowCSVModal] = useState(false);
  const [csvViewMode, setCsvViewMode] = useState<'table' | 'raw'>('table');
  const [batchMode, setBatchMode] = useState(false);
  const [batchVideos, setBatchVideos] = useState<File[]>([]);
  const [batchAnalyses, setBatchAnalyses] = useState<{[key: string]: FrameAnalysis[]}>({});
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0, currentVideo: '' });
  const [selectedBatchVideo, setSelectedBatchVideo] = useState<string>('');
  const [batchFrames, setBatchFrames] = useState<{[key: string]: string[]}>({});

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const fileSizeMB = file.size / (1024 * 1024);
      
      setVideoFile(file);
      setFrames([]);
      setAnalyses([]);
      setCurrentFrame(0);
      setStatus('idle');
      setError(null);
      setView({ zoom: 1, pan: { x: 0, y: 0 } });
      
      // Show warning for large files
      if (fileSizeMB > 100) {
        setError(`Warning: File is ${fileSizeMB.toFixed(1)}MB. Conversion may be slow. Consider using a smaller file or converting to MP4 manually.`);
      }
    }
  };

  const handleBatchFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      setBatchVideos(files);
      setBatchAnalyses({});
      setBatchProgress({ current: 0, total: files.length, currentVideo: '' });
      setError(null);
    }
  };

  const handleBatchModeToggle = () => {
    setBatchMode(!batchMode);
    if (batchMode) {
      // Clear batch data when exiting batch mode
      setBatchVideos([]);
      setBatchAnalyses({});
      setBatchFrames({});
      setBatchProgress({ current: 0, total: 0, currentVideo: '' });
      setSelectedBatchVideo('');
    }
  };

  const handleBatchVideoSelect = (videoName: string) => {
    setSelectedBatchVideo(videoName);
    setFrames(batchFrames[videoName] || []);
    setAnalyses(batchAnalyses[videoName] || []);
    setCurrentFrame(0);
    setView({ zoom: 1, pan: { x: 0, y: 0 } });
  };

  const handleBatchAnalyze = async () => {
    if (batchVideos.length === 0) return;

    setStatus('analyzing');
    setError(null);
    setBatchProgress({ current: 0, total: batchVideos.length, currentVideo: '' });
    const newBatchAnalyses: {[key: string]: FrameAnalysis[]} = {};
    const newBatchFrames: {[key: string]: string[]} = {};

    try {
      for (let i = 0; i < batchVideos.length; i++) {
        const video = batchVideos[i];
        setBatchProgress({ current: i, total: batchVideos.length, currentVideo: video.name });
        
        // Extract frames from video
        const fps = 1 / frameInterval;
        const onProgressCallback = (progress: VideoProcessingProgress) => {
          // Update progress for current video
        };

        const extractedFrames = await extractFramesFromVideo(video, fps, onProgressCallback);
        newBatchFrames[video.name] = extractedFrames;
        
        if (extractedFrames.length === 0) {
          newBatchAnalyses[video.name] = [];
          continue;
        }

        // Get image dimensions from first frame
        const img = new Image();
        img.src = `data:image/jpeg;base64,${extractedFrames[0]}`;
        await new Promise(resolve => { img.onload = resolve; });
        const { width, height } = img;
        
        // Set image dimensions for the first video (they should be the same for all videos)
        if (i === 0) {
          setImageDimensions({ width, height });
        }

        // Analyze each frame
        const videoAnalyses: FrameAnalysis[] = [];
        for (let frameIndex = 0; frameIndex < extractedFrames.length; frameIndex++) {
          const frameData = extractedFrames[frameIndex];
          let analysisResult;

          // Convert base64 to ImageData for Hough
          const img = new Image();
          img.src = `data:image/jpeg;base64,${frameData}`;
          await new Promise(resolve => { img.onload = resolve; });
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          canvas.width = img.width;
          canvas.height = img.height;
          ctx?.drawImage(img, 0, 0);
          const imageData = ctx?.getImageData(0, 0, canvas.width, canvas.height);
          analysisResult = await analyzeFrameWithHough(imageData!, detectionMethod);

          // Create sensible defaults if AI detection fails
          if (!analysisResult.dropletsFound || analysisResult.droplets.length < 2) {
            const r = width * 0.1;
            analysisResult.droplets = [
              { id: 0, cx: width / 2 - r, cy: height / 2, r },
              { id: 1, cx: width / 2 + r, cy: height / 2, r },
            ];
            analysisResult.dropletsFound = false;
          }

          if (!analysisResult.scaleFound) {
            analysisResult.scale = {
              x1: width * 0.1,
              y1: height * 0.9,
              x2: width * 0.3,
              y2: height * 0.9,
              label: "50 µm",
              length: width * 0.2
            };
            analysisResult.scaleFound = false;
          }

          if (!analysisResult.timestampFound) {
            analysisResult.timestamp = `${(frameIndex * frameInterval / 1000).toFixed(1)}s`;
            analysisResult.timestampFound = false;
          }

          analysisResult.frame = frameIndex;
          videoAnalyses.push(analysisResult);
        }

        newBatchAnalyses[video.name] = videoAnalyses;
      }

      setBatchAnalyses(newBatchAnalyses);
      setBatchFrames(newBatchFrames);
      
      // Set the first video as selected
      if (batchVideos.length > 0) {
        setSelectedBatchVideo(batchVideos[0].name);
        setFrames(newBatchFrames[batchVideos[0].name] || []);
        setAnalyses(newBatchAnalyses[batchVideos[0].name] || []);
        setCurrentFrame(0);
      }
      
      setStatus('ready');
      setError(`✅ Successfully analyzed ${batchVideos.length} videos`);
      setTimeout(() => setError(null), 5000);

    } catch (error) {
      console.error('Batch analysis error:', error);
      setError('Batch analysis failed');
      setStatus('error');
    }
  };

  const handleAnalyze = async () => {
    if (!videoFile) return;

    setStatus('extracting');
    setError(null);
    setProgress(0);
    setExtractionProgress({ current: 0, total: 0 });
    setAnalyses([]);
    setView({ zoom: 1, pan: { x: 0, y: 0 } });
    setVideoProcessingStage('');
    setVideoProcessingMessage('');
    
    try {
      const fps = 1 / frameInterval;
      const onProgressCallback = (progress: VideoProcessingProgress) => {
        setVideoProcessingStage(progress.stage);
        setVideoProcessingMessage(progress.message);
        setProgress(progress.progress);
        
        if (progress.extractionProgress) {
          setExtractionProgress(progress.extractionProgress);
        }
      };

      const extractedFrames = await extractFramesFromVideo(videoFile, fps, onProgressCallback);
      setFrames(extractedFrames);
      
      if (extractedFrames.length === 0) {
        setStatus('ready');
        return;
      }

      const img = new Image();
      img.src = `data:image/jpeg;base64,${extractedFrames[0]}`;
      await new Promise(resolve => { img.onload = resolve; });
      const { width, height } = img;
      setImageDimensions({ width, height });

      setStatus('analyzing');

      for (let i = 0; i < extractedFrames.length; i++) {
        const frameData = extractedFrames[i];
        let analysisResult;

        // Hough Transform
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const image = new Image();
        image.src = `data:image/jpeg;base64,${frameData}`;
        await new Promise(r => image.onload = r);
        
        canvas.width = image.width;
        canvas.height = image.height;
        ctx?.drawImage(image, 0, 0);
        const imageData = ctx?.getImageData(0, 0, image.width, image.height);

        if (imageData) {
            // Use the new comprehensive analysis function
            analysisResult = await analyzeFrameWithHough(imageData, detectionMethod);
        } else {
            // Fallback if imageData is null
            analysisResult = {
                timestamp: "Not Found",
                timestampFound: false,
                scaleFound: false,
                dropletsFound: false,
                droplets: [],
                scale: {
                    x1: 0, y1: 0, x2: 0, y2: 0,
                    label: "Not Found",
                    length: 0
                }
            };
        }
        
        const newAnalysis: FrameAnalysis = { frame: i, ...analysisResult };

        setAnalyses(prev => [...prev, newAnalysis]);
        setCurrentFrame(i);
        setProgress(((i + 1) / extractedFrames.length) * 100);
      }

      setStatus('ready');
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
      setStatus('error');
    }
  };

  const handleAnalysisChange = (newAnalysis: FrameAnalysis) => {
    setAnalyses(prev => prev.map(a => a.frame === newAnalysis.frame ? newAnalysis : a));
  };

  const generateCSVContent = () => {
    if (analyses.length === 0) return '';

    let csvContent = "Frame,Timestamp,Scale_Pixels,Scale_Label,Droplet_1_cx,Droplet_1_cy,Droplet_1_radius,Droplet_2_cx,Droplet_2_cy,Droplet_2_radius\n";

    analyses.forEach(a => {
      const d1 = a.droplets[0] || { cx: '', cy: '', r: '' };
      const d2 = a.droplets[1] || { cx: '', cy: '', r: '' };
      const row = [a.frame, `"${a.timestamp}"`, a.scale.length, `"${a.scale.label}"`, d1.cx, d1.cy, d1.r, d2.cx, d2.cy, d2.r].join(",");
      csvContent += row + "\n";
    });

    return csvContent;
  };

  const renderCSVTable = () => {
    if (analyses.length === 0) return null;

    return (
      <div className="overflow-x-auto shadow-sm">
        <table className="min-w-full bg-white border border-gray-200 rounded-lg overflow-hidden">
          <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider border-b border-gray-200">Frame</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider border-b border-gray-200">Timestamp</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider border-b border-gray-200">Scale (px)</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider border-b border-gray-200">Scale Label</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider border-b border-gray-200">Droplet 1</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider border-b border-gray-200">Droplet 2</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {analyses.map((analysis, index) => {
              const d1 = analysis.droplets[0] || { cx: '', cy: '', r: '' };
              const d2 = analysis.droplets[1] || { cx: '', cy: '', r: '' };
              
              return (
                <tr key={index} className={`hover:bg-blue-50 transition-colors duration-150 ${index % 2 === 0 ? 'bg-white' : 'bg-gray-25'}`}>
                  <td className="px-4 py-3 text-sm font-medium text-gray-900 border-b border-gray-100">{analysis.frame}</td>
                  <td className="px-4 py-3 text-sm text-gray-700 border-b border-gray-100">{analysis.timestamp}</td>
                  <td className="px-4 py-3 text-sm text-gray-700 border-b border-gray-100">{analysis.scale.length}</td>
                  <td className="px-4 py-3 text-sm text-gray-700 border-b border-gray-100">{analysis.scale.label}</td>
                  <td className="px-4 py-3 text-sm text-gray-700 border-b border-gray-100">
                    {d1.cx ? (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        ({d1.cx}, {d1.cy}) r:{d1.r}
                      </span>
                    ) : (
                      <span className="text-gray-400 italic">N/A</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700 border-b border-gray-100">
                    {d2.cx ? (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        ({d2.cx}, {d2.cy}) r:{d2.r}
                      </span>
                    ) : (
                      <span className="text-gray-400 italic">N/A</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  };


  const handleViewCSV = () => {
    setShowCSVModal(true);
  };

  const handleExportAll = async () => {
    if (analyses.length === 0 || status !== 'ready') return;

    setStatus('analyzing'); // Show a status while creating export
    setProgress(0);

    try {
      const zip = new JSZip();
      
      // 1. Add CSV analysis data
      const csvContent = generateCSVContent();
      zip.file("analysis.csv", csvContent);

      // 2. Add JSON analysis data (for loading back into the app)
      const analysisData = {
        analyses,
        imageDimensions,
        frameInterval,
        detectionMethod,
        videoFileName: videoFile?.name || 'unknown',
        exportDate: new Date().toISOString()
      };
      zip.file("analysis_data.json", JSON.stringify(analysisData, null, 2));

      // 3. Add raw and processed frames
      const rawFolder = zip.folder('raw');
      const processedFolder = zip.folder('processed');
      const canvas = document.createElement('canvas');
      canvas.width = imageDimensions.width;
      canvas.height = imageDimensions.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        setError("Could not create canvas for export.");
        setStatus('error');
        return;
      }

      for (let i = 0; i < frames.length; i++) {
        const frameData = frames[i];
        const analysis = analyses[i];
        const baseName = `frame_${String(i).padStart(4, '0')}`;
        const imgName = `${baseName}.jpeg`;
        const jsonName = `${baseName}.json`;

        // Add raw image and JSON metadata
        rawFolder?.file(imgName, frameData, { base64: true });
        if(analysis) {
            rawFolder?.file(jsonName, JSON.stringify(analysis, null, 2));
            processedFolder?.file(jsonName, JSON.stringify(analysis, null, 2));
        }

        const img = new Image();
        img.src = `data:image/jpeg;base64,${frameData}`;
        await new Promise<void>(resolve => {
            img.onload = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);

                // Draw analysis
                if (analysis) {
                    analysis.droplets.forEach(d => {
                        ctx.beginPath();
                        ctx.strokeStyle = '#34D399';
                        ctx.lineWidth = 3;
                        ctx.arc(d.cx, d.cy, d.r, 0, 2 * Math.PI);
                        ctx.stroke();
                    });
                    const { scale } = analysis;
                    ctx.beginPath();
                    ctx.strokeStyle = '#FBBF24';
                    ctx.lineWidth = 3;
                    ctx.moveTo(scale.x1, scale.y1);
                    ctx.lineTo(scale.x2, scale.y2);
                    ctx.stroke();
                    ctx.fillStyle = '#F59E0B';
                    ctx.font = "16px Arial";
                    ctx.textAlign = 'center';
                    ctx.fillText(scale.label, (scale.x1 + scale.x2) / 2, Math.min(scale.y1, scale.y2) - 15);
                }

                canvas.toBlob(blob => {
                    if (blob) {
                        processedFolder?.file(imgName, blob);
                    }
                    resolve();
                }, 'image/jpeg');
            };
        });
        setProgress(((i + 1) / frames.length) * 100);
      }

      // 4. Add original video if available
      if (videoFile) {
        zip.file(`original_video/${videoFile.name}`, videoFile);
      }

      // 5. Add video info
      zip.file("video_info.txt", `Original video: ${videoFile?.name || 'unknown'}\nFormat: ${videoFile?.type || 'unknown'}\nFrames extracted: ${frames.length}\nFrame interval: ${frameInterval}ms\nDetection method: ${detectionMethod}`);

      // Generate and download the zip
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(zipBlob);
      link.download = `droplet_analysis_${new Date().toISOString().split('T')[0]}.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      setStatus('ready');
    } catch (error) {
      console.error('Export error:', error);
      setError('Failed to create export package');
      setStatus('error');
    }
  };

  const handleLoadAnalysis = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.zip')) {
      setError('Please select a valid analysis ZIP file');
      return;
    }

    setStatus('analyzing');
    setError(null);
    setProgress(0);

    try {
      const zip = new JSZip();
      const zipContent = await zip.loadAsync(file);
      
      // Check if this is a valid analysis file
      if (!zipContent.files['analysis_data.json']) {
        setError('Invalid analysis file format');
        setStatus('error');
        return;
      }

      // Load analysis data
      const analysisDataContent = await zipContent.files['analysis_data.json'].async('text');
      const analysisData = JSON.parse(analysisDataContent);
      
      // Validate analysis data
      if (!analysisData.analyses || !Array.isArray(analysisData.analyses)) {
        setError('Invalid analysis data format');
        setStatus('error');
        return;
      }
      
      // Restore application state
      setAnalyses(analysisData.analyses);
      setImageDimensions(analysisData.imageDimensions || { width: 1280, height: 720 });
      setFrameInterval(analysisData.frameInterval || 120);
      setDetectionMethod(analysisData.detectionMethod || 'v9');
      
      // Load frames from raw folder - use analysis data as source of truth for frame count
      const rawFolder = zipContent.folder('raw');
      if (!rawFolder) {
        setError('No raw frames found in analysis file');
        setStatus('error');
        return;
      }
      
      // Ensure we're only loading from raw folder, never processed
      console.log('Loading frames from RAW folder only (not processed)');
      
      // Verify raw folder path
      const rawFolderPath = 'raw/';
      console.log('Raw folder path:', rawFolderPath);

      // Use the number of analyses as the expected frame count
      const expectedFrameCount = analysisData.analyses.length;
      const loadedFrames: string[] = [];
      
      console.log(`Loading analysis with ${expectedFrameCount} expected frames`);
      
      // Debug: List all files in raw folder
      const allRawFiles = Object.keys(rawFolder.files);
      console.log('Available files in raw folder:', allRawFiles);
      
      // Filter to only files that are actually in the raw folder (not subfolders)
      // Files in raw folder should not have 'processed/' in their path
      const rawOnlyFiles = allRawFiles.filter(file => !file.includes('processed/'));
      console.log('Raw-only files (no processed):', rawOnlyFiles);
      
      // Also check if there are any .jpeg files at all (only from raw folder)
      const jpegFiles = rawOnlyFiles.filter(file => file.endsWith('.jpeg'));
      console.log('JPEG files found in raw folder:', jpegFiles);
      
      // Debug: Check if there's a processed folder and what's in it
      const processedFolder = zipContent.folder('processed');
      if (processedFolder) {
        const processedFiles = Object.keys(processedFolder.files);
        const processedJpegs = processedFiles.filter(file => file.endsWith('.jpeg'));
        console.log('Available files in processed folder:', processedFiles);
        console.log('JPEG files found in processed folder:', processedJpegs);
      } else {
        console.log('No processed folder found in ZIP file');
      }
      
      // Load frames in order based on analysis data
      for (let i = 0; i < expectedFrameCount; i++) {
        const frameFileName = `frame_${String(i).padStart(4, '0')}.jpeg`;
        const frameFile = rawFolder.files[frameFileName];
        
        console.log(`Looking for frame file: ${frameFileName}, found:`, !!frameFile);
        
        if (frameFile) {
          const frameData = await frameFile.async('base64');
          loadedFrames.push(frameData);
          console.log(`Successfully loaded frame ${i} from raw folder: ${frameFileName}`);
          console.log(`Full path: raw/${frameFileName}`);
        } else {
          console.warn(`Frame file ${frameFileName} not found in raw folder`);
          // Try alternative naming patterns
          const altFileName1 = `frame_${i}.jpeg`;
          const altFileName2 = `frame_${String(i).padStart(3, '0')}.jpeg`;
          const altFile1 = rawFolder.files[altFileName1];
          const altFile2 = rawFolder.files[altFileName2];
          
          if (altFile1) {
            console.log(`Found alternative file in raw folder: ${altFileName1}`);
            const frameData = await altFile1.async('base64');
            loadedFrames.push(frameData);
          } else if (altFile2) {
            console.log(`Found alternative file in raw folder: ${altFileName2}`);
            const frameData = await altFile2.async('base64');
            loadedFrames.push(frameData);
          } else {
            // Try to find any frame file that might match
            const framePattern = new RegExp(`frame_${i}(?:_\\d+)?\\.jpeg$`);
            const matchingFile = allRawFiles.find(file => framePattern.test(file));
            
            if (matchingFile) {
              console.log(`Found matching file in raw folder: ${matchingFile}`);
              const frameData = await rawFolder.files[matchingFile].async('base64');
              loadedFrames.push(frameData);
            } else {
              console.warn(`No frame file found for frame ${i} in raw folder`);
              loadedFrames.push('');
            }
          }
        }
        setProgress(((i + 1) / expectedFrameCount) * 100);
      }
      
      console.log(`Successfully loaded ${loadedFrames.length} frames`);
      
      // If we didn't load any frames successfully, try a different approach
      let alternativeFrames: string[] = [];
      if (loadedFrames.filter(frame => frame !== '').length === 0 && jpegFiles.length > 0) {
        console.log('No frames loaded with standard approach, trying alternative method...');
        
        // Try to load frames based on available files
        alternativeFrames = [];
        for (let i = 0; i < Math.min(expectedFrameCount, jpegFiles.length); i++) {
          const jpegFile = jpegFiles[i];
          if (jpegFile && !jpegFile.includes('processed/')) {
            const frameData = await rawFolder.files[jpegFile].async('base64');
            alternativeFrames.push(frameData);
            console.log(`Loaded alternative frame from raw folder: ${jpegFile}`);
          } else if (jpegFile && jpegFile.includes('processed/')) {
            console.warn(`Skipping processed file: ${jpegFile}`);
          }
        }
        
        if (alternativeFrames.length > 0) {
          console.log(`Loaded ${alternativeFrames.length} frames using alternative method`);
          setFrames(alternativeFrames);
        } else {
          setFrames(loadedFrames);
        }
      } else {
        setFrames(loadedFrames);
      }
      
      setCurrentFrame(0);
      setStatus('ready');
      
      // Show success message
      const finalFrames = loadedFrames.filter(frame => frame !== '').length > 0 ? loadedFrames : (alternativeFrames || []);
      const actualLoadedFrames = finalFrames.filter(frame => frame !== '').length;
      const missingFrames = expectedFrameCount - actualLoadedFrames;
      
      if (missingFrames > 0) {
        setError(`⚠️ Loaded analysis with ${actualLoadedFrames} frames (${missingFrames} missing) from ${analysisData.videoFileName}`);
      } else {
        setError(`✅ Successfully loaded analysis with ${actualLoadedFrames} frames from ${analysisData.videoFileName}`);
      }
      setTimeout(() => setError(null), 5000);
      
    } catch (error) {
      console.error('Load error:', error);
      setError('Failed to load analysis file');
      setStatus('error');
    }
    
    // Reset file input
    e.target.value = '';
  };

  const handleBatchExport = async () => {
    if (Object.keys(batchAnalyses).length === 0) return;

    setStatus('analyzing');
    setProgress(0);

    try {
      const zip = new JSZip();
      let processedVideos = 0;

      for (const [videoName, analyses] of Object.entries(batchAnalyses)) {
        if ((analyses as FrameAnalysis[]).length === 0) continue;

        const videoFolder = zip.folder(videoName.replace(/\.[^/.]+$/, "")); // Remove extension
        const rawFolder = videoFolder?.folder("raw");
        const processedFolder = videoFolder?.folder("processed");
        
        // Add raw frames
        const frames = batchFrames[videoName] || [];
        for (let i = 0; i < frames.length; i++) {
          const frameData = frames[i];
          const paddedIndex = i.toString().padStart(4, '0');
          rawFolder?.file(`frame_${paddedIndex}.jpeg`, frameData, { base64: true });
        }

        // Add processed frames (with overlays)
        for (let i = 0; i < (analyses as FrameAnalysis[]).length; i++) {
          const analysis = (analyses as FrameAnalysis[])[i];
          const frameData = frames[i];
          if (!frameData) continue;

          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          if (!ctx) continue;

          const img = new Image();
          img.src = `data:image/jpeg;base64,${frameData}`;
          await new Promise(resolve => { img.onload = resolve; });
          
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);

          // Draw droplets
          analysis.droplets.forEach((droplet, index) => {
            ctx.strokeStyle = index === 0 ? '#00ff00' : '#ff0000';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(droplet.cx, droplet.cy, droplet.r, 0, 2 * Math.PI);
            ctx.stroke();
            
            // Draw center dot
            ctx.fillStyle = index === 0 ? '#00ff00' : '#ff0000';
            ctx.beginPath();
            ctx.arc(droplet.cx, droplet.cy, 3, 0, 2 * Math.PI);
            ctx.fill();
          });

          // Draw scale bar
          if (analysis.scale) {
            ctx.strokeStyle = '#ffff00';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(analysis.scale.x1, analysis.scale.y1);
            ctx.lineTo(analysis.scale.x2, analysis.scale.y2);
            ctx.stroke();
            
            // Draw scale label
            ctx.fillStyle = '#ffff00';
            ctx.font = '16px Arial';
            ctx.fillText(analysis.scale.label, analysis.scale.x1, analysis.scale.y1 - 10);
          }

          // Draw timestamp
          if (analysis.timestamp) {
            ctx.fillStyle = '#ffffff';
            ctx.font = '20px Arial';
            ctx.fillText(analysis.timestamp, 20, 40);
          }

          const paddedIndex = i.toString().padStart(4, '0');
          const processedDataUrl = canvas.toDataURL('image/jpeg', 0.9);
          const base64Data = processedDataUrl.split(',')[1];
          processedFolder?.file(`frame_${paddedIndex}.jpeg`, base64Data, { base64: true });
        }

        // Add original video
        const videoFile = batchVideos.find(v => v.name === videoName);
        if (videoFile) {
          const videoData = await videoFile.arrayBuffer();
          videoFolder?.file(videoName, videoData);
        }
        
        // Add CSV analysis data
        let csvContent = "Frame,Timestamp,Scale_Pixels,Scale_Label,Droplet_1_cx,Droplet_1_cy,Droplet_1_radius,Droplet_2_cx,Droplet_2_cy,Droplet_2_radius\n";
        (analyses as FrameAnalysis[]).forEach(a => {
          const d1 = a.droplets[0] || { cx: '', cy: '', r: '' };
          const d2 = a.droplets[1] || { cx: '', cy: '', r: '' };
          const row = [a.frame, `"${a.timestamp}"`, a.scale.length, `"${a.scale.label}"`, d1.cx, d1.cy, d1.r, d2.cx, d2.cy, d2.r].join(",");
          csvContent += row + "\n";
        });
        videoFolder?.file("analysis.csv", csvContent);

        // Add JSON analysis data
        const analysisData = {
          analyses,
          imageDimensions,
          frameInterval,
          detectionMethod,
          videoFileName: videoName,
          exportDate: new Date().toISOString()
        };
        videoFolder?.file("analysis_data.json", JSON.stringify(analysisData, null, 2));

        // Add video info
        videoFolder?.file("video_info.txt", `Video: ${videoName}\nFrames analyzed: ${(analyses as FrameAnalysis[]).length}\nFrame interval: ${frameInterval}ms\nDetection method: ${detectionMethod}`);

        processedVideos++;
        setProgress((processedVideos / Object.keys(batchAnalyses).length) * 100);
      }

      // Generate and download the zip
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(zipBlob);
      link.download = `batch_analysis_${new Date().toISOString().split('T')[0]}.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      setStatus('ready');
    } catch (error) {
      console.error('Batch export error:', error);
      setError('Failed to create batch export');
      setStatus('error');
    }
  };

  const handleBatchImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.zip')) {
      setError('Please select a valid batch analysis ZIP file');
      return;
    }

    setStatus('analyzing');
    setError(null);
    setProgress(0);

    try {
      const zip = new JSZip();
      const zipContent = await zip.loadAsync(file);
      
      const importedAnalyses: {[key: string]: FrameAnalysis[]} = {};
      const importedFrames: {[key: string]: string[]} = {};
      let processedVideos = 0;
      const videoFolders = Object.keys(zipContent.files).filter(name => 
        zipContent.files[name].dir && !name.includes('/')
      );

      for (const folderName of videoFolders) {
        const folder = zipContent.folder(folderName);
        if (!folder) continue;

        // Check if this folder has analysis data
        if (!folder.files['analysis_data.json']) continue;

        // Load analysis data
        const analysisDataContent = await folder.files['analysis_data.json'].async('text');
        const analysisData = JSON.parse(analysisDataContent);
        
        if (analysisData.analyses && Array.isArray(analysisData.analyses)) {
          importedAnalyses[analysisData.videoFileName || folderName] = analysisData.analyses;
          
          // Set image dimensions from the analysis data if available
          if (analysisData.imageDimensions && processedVideos === 0) {
            setImageDimensions(analysisData.imageDimensions);
          }
        }

        // Load raw frames
        const rawFolder = folder.folder('raw');
        if (rawFolder) {
          const frameFiles = Object.keys(rawFolder.files).filter(name => 
            name.endsWith('.jpeg') && !rawFolder.files[name].dir
          ).sort();
          
          const frames: string[] = [];
          for (const frameFile of frameFiles) {
            const frameData = await rawFolder.files[frameFile].async('base64');
            frames.push(frameData);
          }
          importedFrames[analysisData.videoFileName || folderName] = frames;
        }

        processedVideos++;
        setProgress((processedVideos / videoFolders.length) * 100);
      }

      setBatchAnalyses(importedAnalyses);
      setBatchFrames(importedFrames);
      
      // Set the first video as selected
      const firstVideoName = Object.keys(importedAnalyses)[0];
      if (firstVideoName) {
        setSelectedBatchVideo(firstVideoName);
        setFrames(importedFrames[firstVideoName] || []);
        setAnalyses(importedAnalyses[firstVideoName] || []);
        setCurrentFrame(0);
      }
      
      setStatus('ready');
      setError(`✅ Successfully imported ${Object.keys(importedAnalyses).length} video analyses`);
      setTimeout(() => setError(null), 5000);
      
    } catch (error) {
      console.error('Batch import error:', error);
      setError('Failed to import batch analysis');
      setStatus('error');
    }
    
    // Reset file input
    e.target.value = '';
  };
  
  const currentAnalysis = useMemo(() => {
    return analyses.find(a => a.frame === currentFrame) || null;
  }, [analyses, currentFrame]);

  const renderStatus = () => {
    switch (status) {
      case 'idle':
        return <p className="text-gray-500">Upload a video to begin analysis.</p>;
      case 'extracting':
        const { current, total } = extractionProgress;
        return (
          <div className="w-full">
            <p className="text-blue-500 animate-pulse">
              {videoProcessingStage === 'checking' && 'Checking video format...'}
              {videoProcessingStage === 'converting' && `Converting video... (${Math.round(progress)}%)`}
              {videoProcessingStage === 'extracting' && `Extracting frame ${current} of ~${total} from video...`}
              {videoProcessingStage === 'complete' && 'Video processing complete!'}
              {videoProcessingStage === 'error' && 'Video processing error!'}
              {!videoProcessingStage && 'Processing video...'}
            </p>
            {videoProcessingMessage && (
              <p className="text-sm text-gray-600 mt-1">{videoProcessingMessage}</p>
            )}
            {(videoProcessingStage === 'converting' || videoProcessingStage === 'extracting') && (
              <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
              </div>
            )}
          </div>
        );
      case 'analyzing':
        return (
          <div className="w-full">
            {batchMode ? (
              <div>
                <p className="text-purple-500 animate-pulse">
                  Processing batch: {batchProgress.currentVideo}
                </p>
                <p className="text-sm text-gray-600">
                  Video {batchProgress.current + 1} of {batchProgress.total}
                </p>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                  <div className="bg-purple-600 h-2.5 rounded-full" style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }}></div>
                </div>
              </div>
            ) : (
              <div>
                <p className="text-purple-500 animate-pulse">Processing frames... ({Math.round(progress)}%)</p>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                  <div className="bg-purple-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
                </div>
              </div>
            )}
          </div>
        );
      case 'ready':
        return <p className="text-green-500">Analysis complete. Review and correct frames as needed.</p>;
      case 'error':
        return <p className={error?.startsWith('✅') ? "text-green-500 font-semibold" : "text-red-500 font-semibold"}>{error?.startsWith('✅') ? error : `Error: ${error}`}</p>;
    }
  };

  const renderContent = () => {
    if (frames.length > 0 && currentAnalysis) {
      return (
        <div className="w-full flex flex-col items-center">
            <div className="w-full max-w-4xl relative">
              {currentAnalysis && (
                <div className={`absolute top-2 left-2 z-10 bg-black bg-opacity-60 text-white p-2 rounded-md text-xs space-y-1 shadow-lg ${
                  (currentAnalysis.dropletsFound === false || 
                   currentAnalysis.scaleFound === false || 
                   currentAnalysis.timestampFound === false) 
                    ? 'block' : 'hidden'
                }`}>
                  {currentAnalysis.dropletsFound === false && <p className="text-amber-300">⚠️ Droplets defaulted</p>}
                  {currentAnalysis.scaleFound === false && <p className="text-amber-300">⚠️ Scale defaulted</p>}
                </div>
              )}
              <FrameCanvas 
                  frameData={frames[currentFrame]} 
                  analysis={currentAnalysis} 
                  onAnalysisChange={handleAnalysisChange}
                  imageDimensions={imageDimensions}
                  view={view}
                  onViewChange={setView}
              />
            </div>
            <div className="w-full max-w-4xl mt-4">
                <input
                    type="range"
                    min="0"
                    max={frames.length - 1}
                    value={currentFrame}
                    onChange={(e) => setCurrentFrame(Number(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-sm text-gray-600 mt-1">
                    <span>Frame: {currentFrame + 1} / {frames.length}</span>
                    <span>Time: {currentAnalysis.timestamp}</span>
                </div>
            </div>
        </div>
      );
    }
    return (
        <div className="flex flex-col items-center justify-center h-96 border-2 border-dashed border-gray-300 rounded-lg bg-gray-50 text-center p-8">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.55a2 2 0 01.45 2.12l-2.5 7A2 2 0 0115.5 21H8.5a2 2 0 01-2-2v-7a2 2 0 012-2h4M15 10V3a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2h2" />
            </svg>
            <h3 className="mt-4 text-xl font-medium text-gray-900">Analysis Preview</h3>
            <p className="mt-1 text-sm text-gray-500">Your video frames will appear here after analysis.</p>
        </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800 font-sans">
      <header className="bg-white shadow-md">
        <div className="container mx-auto px-6 py-4">
          <h1 className="text-3xl font-bold text-gray-900">Droplet Analysis Lab</h1>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-semibold mb-4 border-b pb-2">Controls</h2>
            <div className="space-y-6">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label htmlFor="video-upload" className="block text-sm font-medium text-gray-700">1. Upload Video</label>
                  <button
                    onClick={handleBatchModeToggle}
                    className={`px-3 py-1 text-xs rounded-full transition-colors ${
                      batchMode 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {batchMode ? 'Batch Mode' : 'Single Mode'}
                  </button>
                </div>
                
                {batchMode ? (
                  <div className="space-y-2">
                    <input 
                      id="video-upload"
                      type="file" 
                      accept="video/*" 
                      multiple
                      onChange={handleBatchFileChange}
                      className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                    />
                    {batchVideos.length > 0 && (
                      <div className="text-sm text-gray-600">
                        {batchVideos.length} video(s) selected
                      </div>
                    )}
                  </div>
                ) : (
                  <input 
                    id="video-upload"
                    type="file" 
                    accept="video/*" 
                    onChange={handleFileChange}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  />
                )}
                
                <p className="text-xs text-gray-500 mt-1">
                  Supports MP4, WebM, OGG directly. AVI, MOV, WMV, FLV, MKV, 3GP, M4V formats 
                  will be automatically converted to MP4. <strong>Supports videos up to 5GB.</strong>
                </p>
              </div>
              
              <div>
                <label htmlFor="frame-interval" className="block text-sm font-medium text-gray-700 mb-1">2. Set Frame Interval</label>
                <select
                  id="frame-interval"
                  value={frameInterval}
                  onChange={(e) => setFrameInterval(Number(e.target.value))}
                  disabled={status === 'extracting' || status === 'analyzing'}
                  className="w-full p-2 bg-white border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value={0.1}>0.1 seconds (10 FPS)</option>
                  <option value={0.2}>0.2 seconds (5 FPS)</option>
                  <option value={0.5}>0.5 seconds (2 FPS)</option>
                  <option value={1}>1 second (1 FPS)</option>
                  <option value={2}>2 seconds (0.5 FPS)</option>
                  <option value={5}>5 seconds (0.2 FPS)</option>
                  <option value={10}>10 seconds (0.1 FPS)</option>
                  <option value={30}>30 seconds (0.03 FPS)</option>
                  <option value={60}>60 seconds (0.017 FPS)</option>
                  <option value={120}>120 seconds (0.0083 FPS)</option>
                </select>
              </div>

               <div>
                <label htmlFor="detection-method" className="block text-sm font-medium text-gray-700 mb-1">3. Detection Version</label>
                <select
                  id="detection-method"
                  value={detectionMethod}
                  onChange={(e) => setDetectionMethod(e.target.value as 'v1' | 'v2' | 'v3' | 'v4' | 'v5' | 'v6' | 'v7' | 'v8' | 'v9')}
                  disabled={status === 'extracting' || status === 'analyzing'}
                  className="w-full p-2 bg-white border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="v1">V1 - Hough Circles (Computer Vision)</option>
                  <option value="v2">V2 - Optimized Template Matching</option>
                  <option value="v3">V3 - Fast Hybrid Detection (50% Better than V2)</option>
                  <option value="v4">V4 - Advanced Hough Detection (87% Better than V2)</option>
                  <option value="v5">V5 - Optimized Hough Detection (32% Better than V4)</option>
                  <option value="v6">V6 - Ultra-Optimized Hough Detection (4% Better than V5)</option>
                  <option value="v7">V7 - Microscope-Adaptive Detection (7% Better than V6)</option>
                  <option value="v8">V8 - V3 Hybrid with Sophisticated Selection (7.6% Better than V7)</option>
                  <option value="v9">V9 - Complete Microscope_2 Optimization (72.6% Droplet + 99.1% Scale)</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  {detectionMethod === 'v1' 
                    ? 'Uses computer vision algorithms to detect actual droplets' 
                    : detectionMethod === 'v2'
                    ? 'Advanced template matching with 98% better performance than V1'
                    : detectionMethod === 'v3'
                    ? 'Fast hybrid approach combining Hough circles with template matching fallback'
                    : detectionMethod === 'v4'
                    ? 'Advanced Hough detection with progressive sensitivity and optimized preprocessing'
                    : detectionMethod === 'v5'
                    ? 'Optimized Hough detection with fine-tuned parameters and progressive sensitivity'
                    : detectionMethod === 'v6'
                    ? 'Ultra-optimized Hough detection with aggressive parameter tuning'
                    : detectionMethod === 'v7'
                    ? 'Microscope-adaptive Hough detection with parameter optimization based on image characteristics'
                    : detectionMethod === 'v8'
                    ? 'V3 hybrid approach with sophisticated selection criteria for optimal performance'
                    : detectionMethod === 'v9'
                    ? 'Complete microscope_2 optimization with 72.6% better droplet detection and 99.1% better scale detection accuracy'
                    : 'Advanced Hough detection with progressive sensitivity and optimized preprocessing'
                  }
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">4. Start Analysis</label>
                {batchMode ? (
                  <button
                    onClick={handleBatchAnalyze}
                    disabled={batchVideos.length === 0 || status === 'extracting' || status === 'analyzing'}
                    className="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-300 flex items-center justify-center"
                  >
                    {(status === 'extracting' || status === 'analyzing') && <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
                    Analyze {batchVideos.length} Videos
                  </button>
                ) : (
                  <button
                    onClick={handleAnalyze}
                    disabled={!videoFile || status === 'extracting' || status === 'analyzing'}
                    className="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-300 flex items-center justify-center"
                  >
                    {(status === 'extracting' || status === 'analyzing') && <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
                    Analyze Video
                  </button>
                )}
              </div>

              <div>
                 <label className="block text-sm font-medium text-gray-700 mb-1">5. Export & Load</label>
                 {batchMode ? (
                   <div className="space-y-2">
                     <button
                       onClick={handleBatchExport}
                       disabled={Object.keys(batchAnalyses).length === 0}
                       className="w-full bg-green-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-300 text-center"
                     >
                       Export Batch Analysis
                     </button>
                     <label className="w-full bg-indigo-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-indigo-700 cursor-pointer transition duration-300 text-center">
                       Import Batch Analysis
                       <input
                         type="file"
                         accept=".zip"
                         onChange={handleBatchImport}
                         className="hidden"
                       />
                     </label>
                     {Object.keys(batchAnalyses).length > 0 && (
                       <div className="text-sm text-gray-600 text-center">
                         {Object.keys(batchAnalyses).length} video(s) analyzed
                       </div>
                     )}
                   </div>
                 ) : null}
                 
                   {batchMode && Object.keys(batchAnalyses).length > 0 && (
                     <div>
                       <label className="block text-sm font-medium text-gray-700 mb-1">6. Select Video to View</label>
                     <select
                       value={selectedBatchVideo}
                       onChange={(e) => handleBatchVideoSelect(e.target.value)}
                       className="w-full p-2 bg-white border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                     >
                       {Object.keys(batchAnalyses).map(videoName => (
                         <option key={videoName} value={videoName}>
                           {videoName} ({batchAnalyses[videoName].length} frames)
                         </option>
                       ))}
                     </select>
                   </div>
                 )}
                 
                 {!batchMode && (
                   <div className="space-y-2">
                     <div className="flex space-x-2 mb-2">
                        <button
                          onClick={handleExportAll}
                          disabled={status !== 'ready'}
                          className="w-full bg-green-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-300"
                        >
                          Export All (CSV + Frames + Video)
                        </button>
                        <label className="w-full bg-indigo-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-indigo-700 cursor-pointer transition duration-300 text-center">
                          Load Analysis
                          <input
                            type="file"
                            accept=".zip"
                            onChange={handleLoadAnalysis}
                            className="hidden"
                          />
                        </label>
                     </div>
                     <div className="flex space-x-2">
                        <button
                          onClick={handleViewCSV}
                          disabled={status !== 'ready' || analyses.length === 0}
                          className="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-300"
                        >
                          View CSV Data
                        </button>
                     </div>
                   </div>
                 )}
                 <p className="text-xs text-gray-500 mt-1">
                   Export includes CSV data, processed frames, original video, and analysis metadata. Load restores a previous analysis. View CSV shows the data in a readable format.
                 </p>
              </div>

              <div className="pt-4 border-t">
                  <h3 className="text-lg font-medium text-gray-800">Status</h3>
                  <div className="mt-2 text-sm">{renderStatus()}</div>
              </div>

            </div>
          </div>

          <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-lg">
             <h2 className="text-2xl font-semibold mb-4 border-b pb-2">Frame Viewer</h2>
             {renderContent()}
          </div>
        </div>
      </main>

      {/* CSV Modal */}
      {showCSVModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[80vh] flex flex-col">
            <div className="flex justify-between items-center p-4 border-b">
              <h3 className="text-lg font-semibold text-gray-800">CSV Data Preview</h3>
              <div className="flex items-center space-x-3">
                <div className="flex bg-gray-100 rounded-lg p-1">
                  <button
                    onClick={() => setCsvViewMode('table')}
                    className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                      csvViewMode === 'table' 
                        ? 'bg-white text-gray-900 shadow-sm' 
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    Table View
                  </button>
                  <button
                    onClick={() => setCsvViewMode('raw')}
                    className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                      csvViewMode === 'raw' 
                        ? 'bg-white text-gray-900 shadow-sm' 
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    Raw CSV
                  </button>
                </div>
                <div className="flex space-x-2">
                  <button
                  onClick={() => {
                    const csvContent = generateCSVContent();
                    navigator.clipboard.writeText(csvContent).then(() => {
                      // Show a brief success message
                      const button = event?.target as HTMLButtonElement;
                      const originalText = button.textContent;
                      button.textContent = 'Copied!';
                      button.className = 'bg-green-600 text-white px-3 py-1 rounded transition duration-200';
                      setTimeout(() => {
                        button.textContent = originalText;
                        button.className = 'bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 transition duration-200';
                      }, 2000);
                    }).catch(() => {
                      alert('Failed to copy to clipboard');
                    });
                  }}
                  className="bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 transition duration-200"
                >
                  Copy to Clipboard
                </button>
                <button
                  onClick={() => {
                    const csvContent = generateCSVContent();
                    const blob = new Blob([csvContent], { type: 'text/csv' });
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = 'droplet_analysis.csv';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(url);
                  }}
                  className="bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700 transition duration-200"
                >
                  Download CSV
                </button>
                <button
                  onClick={() => setShowCSVModal(false)}
                  className="bg-gray-500 text-white px-3 py-1 rounded hover:bg-gray-600 transition duration-200"
                >
                  Close
                </button>
                </div>
              </div>
            </div>
            <div className="flex-1 overflow-auto p-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="mb-3 text-sm text-gray-700">
                  <strong>Analysis Data Preview:</strong> {csvViewMode === 'table' ? 'Interactive table view of your droplet analysis results' : 'Raw CSV format for technical review'}
                </div>
                {csvViewMode === 'table' ? (
                  renderCSVTable()
                ) : (
                  <pre className="text-sm font-mono text-gray-800 whitespace-pre-wrap overflow-x-auto bg-white p-3 rounded border">
                    {generateCSVContent()}
                  </pre>
                )}
              </div>
            </div>
            <div className="p-4 border-t bg-gray-50 text-sm text-gray-600">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p><strong>Total frames analyzed:</strong> {analyses.length}</p>
                  <p><strong>Data format:</strong> CSV exportable</p>
                </div>
                <div>
                  {csvViewMode === 'table' ? (
                    <>
                      <p><strong>Droplet coordinates:</strong> (x, y) center, r=radius</p>
                      <p><strong>Scale units:</strong> Pixels and labeled units</p>
                    </>
                  ) : (
                    <>
                      <p><strong>CSV columns:</strong> Frame, Timestamp, Scale_Pixels, Scale_Label, Droplet_1_cx, Droplet_1_cy, Droplet_1_radius, Droplet_2_cx, Droplet_2_cy, Droplet_2_radius</p>
                      <p><strong>Format:</strong> Comma-separated values</p>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;