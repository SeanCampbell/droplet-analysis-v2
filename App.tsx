import React, { useState, useCallback, useMemo } from 'react';
import JSZip from 'jszip';
import type { FrameAnalysis, Droplet } from './types';
import { extractFramesFromVideo, VideoProcessingProgress } from './services/videoService';
import { analyzeFrameWithGemini } from './services/geminiService';
import { detectCirclesWithHough, analyzeFrameWithHough } from './services/houghCircleService';
import { FrameCanvas } from './components/FrameCanvas';

type DetectionAlgorithm = 'gemini' | 'hough';

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
  const [detectionAlgorithm, setDetectionAlgorithm] = useState<DetectionAlgorithm>('hough');
  const [detectionMethod, setDetectionMethod] = useState<'v1' | 'v2' | 'v3' | 'v4' | 'v5' | 'v6' | 'v7' | 'v8'>('v8');
  const [view, setView] = useState({ zoom: 1, pan: { x: 0, y: 0 } });

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

        if (detectionAlgorithm === 'gemini') {
            analysisResult = await analyzeFrameWithGemini(frameData);

            // Create sensible defaults if AI detection fails
            if (!analysisResult.dropletsFound || analysisResult.droplets.length < 2) {
                const r = width * 0.1;
                analysisResult.droplets = [
                    { id: 0, cx: width / 2 - r, cy: height / 2, r },
                    { id: 1, cx: width / 2 + r, cy: height / 2, r },
                ];
                analysisResult.dropletsFound = false;
            }
            if (!analysisResult.scaleFound || !analysisResult.scale.x1) {
                const margin = Math.min(width, height) * 0.05;
                const scaleLength = width * 0.15;
                analysisResult.scale = {
                    x1: width - margin - scaleLength,
                    y1: height - margin,
                    x2: width - margin,
                    y2: height - margin,
                    label: "50 µm (default)",
                    length: scaleLength
                };
                analysisResult.scaleFound = false;
            }
             if (!analysisResult.timestampFound) {
                analysisResult.timestamp = "Not Found";
            }

        } else { // Hough Transform
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

  const handleExportCSV = () => {
    if (analyses.length === 0) return;

    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Frame,Timestamp,Scale_Pixels,Scale_Label,Droplet_1_cx,Droplet_1_cy,Droplet_1_radius,Droplet_2_cx,Droplet_2_cy,Droplet_2_radius\n";

    analyses.forEach(a => {
      const d1 = a.droplets[0] || { cx: '', cy: '', r: '' };
      const d2 = a.droplets[1] || { cx: '', cy: '', r: '' };
      const row = [a.frame, `"${a.timestamp}"`, a.scale.length, `"${a.scale.label}"`, d1.cx, d1.cy, d1.r, d2.cx, d2.cy, d2.r].join(",");
      csvContent += row + "\n";
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "droplet_analysis.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadZip = async () => {
    if (status !== 'ready' || frames.length === 0) return;

    setStatus('analyzing'); // Show a status while zipping
    setProgress(0);

    const zip = new JSZip();
    const rawFolder = zip.folder('raw');
    const processedFolder = zip.folder('processed');

    const canvas = document.createElement('canvas');
    canvas.width = imageDimensions.width;
    canvas.height = imageDimensions.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        setError("Could not create canvas for ZIP export.");
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

                // Draw analysis (without handles)
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
    
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(zipBlob);
    link.download = 'droplet_frames.zip';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    setStatus('ready');
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
            <p className="text-purple-500 animate-pulse">Processing frames... ({Math.round(progress)}%)</p>
            <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
              <div className="bg-purple-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
            </div>
          </div>
        );
      case 'ready':
        return <p className="text-green-500">Analysis complete. Review and correct frames as needed.</p>;
      case 'error':
        return <p className="text-red-500 font-semibold">Error: {error}</p>;
    }
  };

  const renderContent = () => {
    if (frames.length > 0 && currentAnalysis) {
      return (
        <div className="w-full flex flex-col items-center">
            <div className="w-full max-w-4xl relative">
              <div className="absolute top-2 right-2 z-10 flex items-center space-x-2 bg-gray-800 bg-opacity-70 p-1 rounded-lg">
                  <span className="text-xs text-gray-300 italic hidden sm:block pr-2">Ctrl+Scroll to Zoom, Space+Drag to Pan</span>
                  <button onClick={() => setView(v => ({ ...v, zoom: Math.min(v.zoom * 1.2, 10) }))} className="px-2 py-1 text-white rounded hover:bg-gray-700 text-lg font-bold" title="Zoom In">+</button>
                  <button onClick={() => setView(v => ({ ...v, zoom: Math.max(v.zoom / 1.2, 1) }))} className="px-2 py-1 text-white rounded hover:bg-gray-700 text-lg font-bold" title="Zoom Out">-</button>
                  <button onClick={() => setView({ zoom: 1, pan: { x: 0, y: 0 } })} className="px-2 py-1 text-white rounded hover:bg-gray-700 text-xs" title="Reset Zoom">Reset</button>
                  <span className="text-xs text-white pr-2">{Math.round(view.zoom * 100)}%</span>
              </div>
              {currentAnalysis && (
                <div className={`absolute top-2 left-2 z-10 bg-black bg-opacity-60 text-white p-2 rounded-md text-xs space-y-1 shadow-lg ${
                  (currentAnalysis.dropletsFound === false || 
                   currentAnalysis.scaleFound === false || 
                   currentAnalysis.timestampFound === false) 
                    ? 'block' : 'hidden'
                }`}>
                  {currentAnalysis.dropletsFound === false && <p className="text-amber-300">⚠️ Droplets defaulted</p>}
                  {currentAnalysis.scaleFound === false && <p className="text-amber-300">⚠️ Scale defaulted</p>}
                  {currentAnalysis.timestampFound === false && <p className="text-amber-300">⚠️ Timestamp not found</p>}
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
                <label htmlFor="video-upload" className="block text-sm font-medium text-gray-700 mb-1">1. Upload Video</label>
                <input 
                  id="video-upload"
                  type="file" 
                  accept="video/*" 
                  onChange={handleFileChange}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
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
                <label htmlFor="detection-algorithm" className="block text-sm font-medium text-gray-700 mb-1">3. Droplet Detection Method</label>
                <select
                  id="detection-algorithm"
                  value={detectionAlgorithm}
                  onChange={(e) => setDetectionAlgorithm(e.target.value as DetectionAlgorithm)}
                  disabled={status === 'extracting' || status === 'analyzing'}
                  className="w-full p-2 bg-white border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="gemini">Gemini Vision</option>
                  <option value="hough">Hough Transform</option>
                </select>
              </div>

              {detectionAlgorithm === 'hough' && (
                <div>
                  <label htmlFor="detection-method" className="block text-sm font-medium text-gray-700 mb-1">4. Hough Detection Version</label>
                  <select
                    id="detection-method"
                    value={detectionMethod}
                    onChange={(e) => setDetectionMethod(e.target.value as 'v1' | 'v2' | 'v3' | 'v4' | 'v5' | 'v6' | 'v7' | 'v8')}
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
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    {detectionMethod === 'v1' 
                      ? 'Uses computer vision algorithms to detect actual droplets' 
                      : detectionMethod === 'v2'
                      ? 'Advanced template matching with 98% better performance than V1'
                      : detectionMethod === 'v3'
                      ? 'Fast hybrid approach combining Hough circles with template matching fallback'
                      : detectionMethod === 'v5'
                      ? 'Optimized Hough detection with fine-tuned parameters and progressive sensitivity'
                      : detectionMethod === 'v6'
                      ? 'Ultra-optimized Hough detection with aggressive parameter tuning'
                      : detectionMethod === 'v7'
                      ? 'Microscope-adaptive Hough detection with parameter optimization based on image characteristics'
                      : detectionMethod === 'v8'
                      ? 'V3 hybrid approach with sophisticated selection criteria for optimal performance'
                      : 'Advanced Hough detection with progressive sensitivity and optimized preprocessing'
                    }
                  </p>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">5. Start Analysis</label>
                <button
                  onClick={handleAnalyze}
                  disabled={!videoFile || status === 'extracting' || status === 'analyzing'}
                  className="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-300 flex items-center justify-center"
                >
                  {(status === 'extracting' || status === 'analyzing') && <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
                  Analyze Video
                </button>
              </div>

              <div>
                 <label className="block text-sm font-medium text-gray-700 mb-1">6. Export & Download</label>
                 <div className="flex space-x-2">
                    <button
                      onClick={handleExportCSV}
                      disabled={status !== 'ready'}
                      className="w-full bg-green-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-300"
                    >
                      Export CSV
                    </button>
                    <button
                      onClick={handleDownloadZip}
                      disabled={status !== 'ready'}
                      className="w-full bg-indigo-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-300"
                    >
                      Download Frames
                    </button>
                 </div>
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
    </div>
  );
};

export default App;