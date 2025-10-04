import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { FrameAnalysis, Droplet, Scale } from '../types';

interface FrameCanvasProps {
  frameData: string;
  analysis: FrameAnalysis | null;
  onAnalysisChange: (newAnalysis: FrameAnalysis) => void;
  imageDimensions: { width: number; height: number };
  view: { zoom: number; pan: { x: number; y: number; } };
  onViewChange: (newView: { zoom: number; pan: { x: number; y: number; } }) => void;
}

type DraggableObject = 
  | { type: 'droplet-center'; dropletId: number }
  | { type: 'droplet-radius'; dropletId: number }
  | { type: 'scale-p1'; }
  | { type: 'scale-p2'; };

const HANDLE_RADIUS = 8;
const LINE_WIDTH = 3;
const FONT = "16px Arial";

export const FrameCanvas: React.FC<FrameCanvasProps> = ({ frameData, analysis, onAnalysisChange, imageDimensions, view, onViewChange }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [draggingObject, setDraggingObject] = useState<DraggableObject | null>(null);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [isSpacebarDown, setIsSpacebarDown] = useState(false);
  const [editingTimestamp, setEditingTimestamp] = useState(false);
  const [editingScale, setEditingScale] = useState(false);
  const [tempTimestamp, setTempTimestamp] = useState('');
  const [tempScaleValue, setTempScaleValue] = useState('');
  const [tempScaleUnit, setTempScaleUnit] = useState('');

  const getMousePos = useCallback((e: React.MouseEvent | React.WheelEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { canvasX: 0, canvasY: 0, imageX: 0, imageY: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const canvasX = (e.clientX - rect.left) * scaleX;
    const canvasY = (e.clientY - rect.top) * scaleY;

    const imageX = (canvasX - view.pan.x) / view.zoom;
    const imageY = (canvasY - view.pan.y) / view.zoom;

    return { canvasX, canvasY, imageX, imageY };
  }, [view.pan, view.zoom]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!ctx || !canvas || !analysis) return;
    
    // Clear canvas first
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!frameData) {
      // Draw placeholder for missing frame data
      ctx.fillStyle = '#f3f4f6';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#6b7280';
      ctx.font = '24px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Frame data not available', canvas.width / 2, canvas.height / 2);
      ctx.font = '16px Arial';
      ctx.fillText('This frame could not be loaded from the analysis file', canvas.width / 2, canvas.height / 2 + 30);
      return;
    }
    
    const img = new Image();
    img.src = `data:image/jpeg;base64,${frameData}`;
    img.onload = () => {
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.translate(view.pan.x, view.pan.y);
      ctx.scale(view.zoom, view.zoom);
      
      ctx.drawImage(img, 0, 0, imageDimensions.width, imageDimensions.height);

      const scaledHandleRadius = HANDLE_RADIUS / view.zoom;
      const scaledLineWidth = LINE_WIDTH / view.zoom;

      // Draw droplets
      analysis.droplets.forEach(droplet => {
        ctx.beginPath();
        ctx.strokeStyle = '#34D399'; // Emerald-400
        ctx.lineWidth = scaledLineWidth;
        ctx.arc(droplet.cx, droplet.cy, droplet.r, 0, 2 * Math.PI);
        ctx.stroke();

        // Center handle
        ctx.beginPath();
        ctx.fillStyle = '#10B981'; // Emerald-500
        ctx.arc(droplet.cx, droplet.cy, scaledHandleRadius, 0, 2 * Math.PI);
        ctx.fill();

        // Radius handle
        ctx.beginPath();
        ctx.fillStyle = '#10B981';
        ctx.arc(droplet.cx + droplet.r, droplet.cy, scaledHandleRadius, 0, 2 * Math.PI);
        ctx.fill();
      });
      
      // Draw scale bar
      const { scale } = analysis;
      ctx.beginPath();
      ctx.strokeStyle = '#FBBF24'; // Amber-400
      ctx.lineWidth = scaledLineWidth;
      ctx.moveTo(scale.x1, scale.y1);
      ctx.lineTo(scale.x2, scale.y2);
      ctx.stroke();

      // Scale handles
      ctx.beginPath();
      ctx.fillStyle = '#F59E0B'; // Amber-500
      ctx.arc(scale.x1, scale.y1, scaledHandleRadius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(scale.x2, scale.y2, scaledHandleRadius, 0, 2 * Math.PI);
      ctx.fill();

      // Scale label
      ctx.fillStyle = '#F59E0B';
      ctx.font = `${16 / view.zoom}px Arial`;
      ctx.textAlign = 'center';
      ctx.fillText(scale.label, (scale.x1 + scale.x2) / 2, Math.min(scale.y1, scale.y2) - (15 / view.zoom));

      ctx.restore();
    };
  }, [frameData, analysis, imageDimensions, view]);
  
  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === ' ') {
        e.preventDefault();
        setIsSpacebarDown(true);
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === ' ') {
        setIsSpacebarDown(false);
        setIsPanning(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!analysis) return;
    
    if (isSpacebarDown && e.button === 0) {
        e.preventDefault();
        setIsPanning(true);
        const { canvasX, canvasY } = getMousePos(e);
        setPanStart({
            x: canvasX - view.pan.x,
            y: canvasY - view.pan.y,
        });
        return;
    }

    const { imageX, imageY } = getMousePos(e);
    const scaledHandleRadius = HANDLE_RADIUS / view.zoom;
    
    // Check for scale handles
    const { scale } = analysis;
    if (Math.hypot(imageX - scale.x1, imageY - scale.y1) < scaledHandleRadius) {
      setDraggingObject({ type: 'scale-p1' });
      return;
    }
    if (Math.hypot(imageX - scale.x2, imageY - scale.y2) < scaledHandleRadius) {
      setDraggingObject({ type: 'scale-p2' });
      return;
    }
    
    // Check for droplet handles
    for (const droplet of analysis.droplets) {
      if (Math.hypot(imageX - droplet.cx, imageY - droplet.cy) < scaledHandleRadius) {
        setDraggingObject({ type: 'droplet-center', dropletId: droplet.id });
        return;
      }
      if (Math.hypot(imageX - (droplet.cx + droplet.r), imageY - droplet.cy) < scaledHandleRadius) {
        setDraggingObject({ type: 'droplet-radius', dropletId: droplet.id });
        return;
      }
    }
  };
  
  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
        const { canvasX, canvasY } = getMousePos(e);
        onViewChange({
            zoom: view.zoom,
            pan: {
                x: canvasX - panStart.x,
                y: canvasY - panStart.y,
            }
        });
        return;
    }

    if (!draggingObject || !analysis) return;
    const { imageX, imageY } = getMousePos(e);
    
    const newAnalysis = JSON.parse(JSON.stringify(analysis)) as FrameAnalysis;

    switch (draggingObject.type) {
      case 'droplet-center': {
        const droplet = newAnalysis.droplets.find(d => d.id === draggingObject.dropletId);
        if (droplet) {
          droplet.cx = imageX;
          droplet.cy = imageY;
        }
        break;
      }
      case 'droplet-radius': {
        const droplet = newAnalysis.droplets.find(d => d.id === draggingObject.dropletId);
        if (droplet) {
          droplet.r = Math.max(HANDLE_RADIUS / view.zoom, Math.hypot(imageX - droplet.cx, imageY - droplet.cy));
        }
        break;
      }
      case 'scale-p1': {
        newAnalysis.scale.x1 = imageX;
        newAnalysis.scale.y1 = imageY;
        newAnalysis.scale.y2 = imageY; // Enforce horizontal
        break;
      }
      case 'scale-p2': {
        newAnalysis.scale.x2 = imageX;
        newAnalysis.scale.y2 = imageY;
        newAnalysis.scale.y1 = imageY; // Enforce horizontal
        break;
      }
    }
    if(draggingObject.type === 'scale-p1' || draggingObject.type === 'scale-p2') {
         newAnalysis.scale.length = Math.abs(newAnalysis.scale.x2 - newAnalysis.scale.x1);
    }
    
    onAnalysisChange(newAnalysis);
  };
  
  const handleMouseUp = () => {
    setDraggingObject(null);
    setIsPanning(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
      if (e.ctrlKey) {
        e.preventDefault();
        const newZoom = Math.max(1, Math.min(view.zoom * (1 - e.deltaY * 0.001), 10));

        const { canvasX, canvasY } = getMousePos(e);
        
        const imageX_before = (canvasX - view.pan.x) / view.zoom;
        const imageY_before = (canvasY - view.pan.y) / view.zoom;
        
        const newPanX = canvasX - imageX_before * newZoom;
        const newPanY = canvasY - imageY_before * newZoom;

        onViewChange({ zoom: newZoom, pan: { x: newPanX, y: newPanY } });
      }
  };

  const getCursor = () => {
      if (isSpacebarDown) return isPanning ? 'grabbing' : 'grab';
      if (draggingObject) return 'grabbing';
      return 'crosshair';
  }

  const handleTimestampEdit = () => {
    if (!analysis) return;
    setTempTimestamp(analysis.timestamp);
    setEditingTimestamp(true);
  };

  const handleTimestampSave = () => {
    if (!analysis) return;
    onAnalysisChange({
      ...analysis,
      timestamp: tempTimestamp
    });
    setEditingTimestamp(false);
  };

  const handleTimestampCancel = () => {
    setEditingTimestamp(false);
    setTempTimestamp('');
  };

  const handleScaleEdit = () => {
    if (!analysis) return;
    // Parse the scale label to extract value and unit
    const scaleLabel = analysis.scale.label;
    const match = scaleLabel.match(/^(\d+(?:\.\d+)?)\s*(.*)$/);
    if (match) {
      setTempScaleValue(match[1]);
      setTempScaleUnit(match[2] || 'µm');
    } else {
      setTempScaleValue('50');
      setTempScaleUnit('µm');
    }
    setEditingScale(true);
  };

  const handleScaleSave = () => {
    if (!analysis) return;
    const newLabel = `${tempScaleValue} ${tempScaleUnit}`;
    onAnalysisChange({
      ...analysis,
      scale: {
        ...analysis.scale,
        label: newLabel
      }
    });
    setEditingScale(false);
  };

  const handleScaleCancel = () => {
    setEditingScale(false);
    setTempScaleValue('');
    setTempScaleUnit('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      if (editingTimestamp) {
        handleTimestampSave();
      } else if (editingScale) {
        handleScaleSave();
      }
    } else if (e.key === 'Escape') {
      if (editingTimestamp) {
        handleTimestampCancel();
      } else if (editingScale) {
        handleScaleCancel();
      }
    }
  };

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={imageDimensions.width}
        height={imageDimensions.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        className="w-full h-auto rounded-lg shadow-md bg-white"
        style={{ cursor: getCursor() }}
      />
      
      {/* Editing Controls Overlay */}
      {analysis && (
        <div className="absolute top-4 left-4 bg-white bg-opacity-90 rounded-lg shadow-lg p-3 space-y-2">
          {/* Timestamp Editing */}
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Timestamp (seconds):</span>
            {editingTimestamp ? (
              <div className="flex items-center space-x-1">
                <input
                  type="text"
                  value={tempTimestamp}
                  onChange={(e) => setTempTimestamp(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="e.g., 0.5"
                  className="text-sm border border-gray-300 rounded px-2 py-1 w-32"
                  autoFocus
                />
                <button
                  onClick={handleTimestampSave}
                  className="text-green-600 hover:text-green-800 text-sm"
                >
                  ✓
                </button>
                <button
                  onClick={handleTimestampCancel}
                  className="text-red-600 hover:text-red-800 text-sm"
                >
                  ✗
                </button>
              </div>
            ) : (
              <div className="flex items-center space-x-1">
                <span className="text-sm text-gray-900">{analysis.timestamp}</span>
                <button
                  onClick={handleTimestampEdit}
                  className="text-blue-600 hover:text-blue-800 text-sm"
                >
                  ✏️
                </button>
              </div>
            )}
          </div>

          {/* Scale Editing */}
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Scale:</span>
            {editingScale ? (
              <div className="flex items-center space-x-1">
                <input
                  type="number"
                  value={tempScaleValue}
                  onChange={(e) => setTempScaleValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="text-sm border border-gray-300 rounded px-2 py-1 w-16"
                  step="0.1"
                  autoFocus
                />
                <select
                  value={tempScaleUnit}
                  onChange={(e) => setTempScaleUnit(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="text-sm border border-gray-300 rounded px-2 py-1"
                >
                  <option value="µm">µm</option>
                  <option value="mm">mm</option>
                  <option value="cm">cm</option>
                  <option value="m">m</option>
                  <option value="nm">nm</option>
                </select>
                <button
                  onClick={handleScaleSave}
                  className="text-green-600 hover:text-green-800 text-sm"
                >
                  ✓
                </button>
                <button
                  onClick={handleScaleCancel}
                  className="text-red-600 hover:text-red-800 text-sm"
                >
                  ✗
                </button>
              </div>
            ) : (
              <div className="flex items-center space-x-1">
                <span className="text-sm text-gray-900">{analysis.scale.label}</span>
                <button
                  onClick={handleScaleEdit}
                  className="text-blue-600 hover:text-blue-800 text-sm"
                >
                  ✏️
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};