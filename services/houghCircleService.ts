import type { Droplet, FrameAnalysis } from '../types';

// Configuration for the Python API server
// Use relative URLs for production (Docker), absolute for development
const PYTHON_API_BASE_URL = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5001';
const API_PREFIX = process.env.NODE_ENV === 'production' ? '/api' : '';

// Convert ImageData to base64 string
const imageDataToBase64 = (imageData: ImageData): string => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error('Could not get canvas context');
    }
    
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    ctx.putImageData(imageData, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.8).split(',')[1]; // Remove data URL prefix
};

// Check if Python API server is available
const checkPythonServerHealth = async (): Promise<boolean> => {
    const healthUrl = `${PYTHON_API_BASE_URL}/health`;
    console.log(`Checking Python API health at: ${healthUrl}`);
    
    try {
        const response = await fetch(healthUrl, {
            method: 'GET',
            timeout: 5000
        });
        console.log(`Health check response: ${response.status} ${response.statusText}`);
        return response.ok;
    } catch (error) {
        console.warn('Python API server not available, falling back to Web Worker:', error);
        return false;
    }
};

// Call Python API for comprehensive frame analysis
const analyzeFrameWithPythonAPI = async (imageData: ImageData): Promise<Omit<FrameAnalysis, 'frame'>> => {
    const base64Image = imageDataToBase64(imageData);
    
    const requestBody = {
        image: base64Image,
        min_radius: 20,
        max_radius: 150,
        dp: 1,
        min_dist: 50,
        param1: 50,
        param2: 85
    };
    
    const apiUrl = `${PYTHON_API_BASE_URL}${API_PREFIX}/analyze-frame`;
    console.log(`🔍 API Configuration:`);
    console.log(`  - NODE_ENV: ${process.env.NODE_ENV}`);
    console.log(`  - PYTHON_API_BASE_URL: "${PYTHON_API_BASE_URL}"`);
    console.log(`  - API_PREFIX: "${API_PREFIX}"`);
    console.log(`  - Final API URL: "${apiUrl}"`);
    console.log(`  - Request body size: ${JSON.stringify(requestBody).length} characters`);
    
    let result;
    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        console.log(`API response status: ${response.status} ${response.statusText}`);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error(`API error response: ${errorText}`);
            throw new Error(`Python API error: ${response.status} ${response.statusText} - ${errorText}`);
        }
        
        result = await response.json();
        console.log(`API response received:`, result);
        
        // Check if the response has an error field (indicates failure)
        if (result.error) {
            throw new Error(`Python API detection failed: ${result.error}`);
        }
        
        // Check if we have the expected data structure
        if (!result.droplets && !result.dropletsFound) {
            throw new Error(`Python API detection failed: Invalid response format`);
        }
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
    
    // Return the result in the same format as Gemini service
    return {
        timestamp: result.timestamp || "Not Found",
        timestampFound: result.timestampFound || false,
        scaleFound: result.scaleFound || false,
        dropletsFound: result.dropletsFound || false,
        droplets: result.droplets || [],
        scale: result.scale || {
            x1: 0, y1: 0, x2: 0, y2: 0,
            label: "Not Found",
            length: 0
        }
    };
};

// Legacy function for backward compatibility
const detectCirclesWithPythonAPI = async (imageData: ImageData): Promise<Droplet[]> => {
    const analysis = await analyzeFrameWithPythonAPI(imageData);
    return analysis.droplets;
};

// Fallback to Web Worker implementation
const createHoughWorker = async (): Promise<Worker> => {
    try {
        const response = await fetch('/services/hough.worker.ts');
        if (!response.ok) {
            throw new Error(`Failed to fetch worker script: ${response.statusText}`);
        }
        const workerScript = await response.text();
        const blob = new Blob([workerScript], { type: 'application/javascript' });
        const workerUrl = URL.createObjectURL(blob);
        return new Worker(workerUrl, { type: 'module' });
    } catch (error) {
        console.error("Error creating Hough worker:", error);
        throw error;
    }
};

const detectCirclesWithWebWorker = (imageData: ImageData): Promise<Droplet[]> => {
    return new Promise(async (resolve, reject) => {
        let worker: Worker | null = null;
        try {
            worker = await createHoughWorker();

            // Listen for messages from the worker
            worker.onmessage = (event: MessageEvent<Droplet[]>) => {
                resolve(event.data);
                worker?.terminate(); // Clean up the worker once done
            };

            // Handle errors from the worker
            worker.onerror = (error: ErrorEvent) => {
                const errorMessage = `Worker error: ${error.message} (at ${error.filename}:${error.lineno})`;
                console.error("Hough Worker Error:", errorMessage, error);
                reject(new Error(errorMessage));
                worker?.terminate(); // Clean up on error
            };

            // Send the image data to the worker to start the computation
            worker.postMessage({ imageData });
        } catch (error) {
            reject(error);
            if (worker) {
                worker.terminate();
            }
        }
    });
};

// Main function that tries Python API first, falls back to Web Worker
export const detectCirclesWithHough = async (imageData: ImageData): Promise<Droplet[]> => {
    try {
        // Check if Python server is available
        const isPythonServerAvailable = await checkPythonServerHealth();
        
        if (isPythonServerAvailable) {
            console.log('Using Python API for Hough circle detection');
            return await detectCirclesWithPythonAPI(imageData);
        } else {
            console.log('Python API not available, using Web Worker for Hough circle detection');
            return await detectCirclesWithWebWorker(imageData);
        }
    } catch (error) {
        console.error('Error in detectCirclesWithHough:', error);
        
        // If Python API fails, try Web Worker as fallback
        try {
            console.log('Python API failed, falling back to Web Worker');
            return await detectCirclesWithWebWorker(imageData);
        } catch (fallbackError) {
            console.error('Both Python API and Web Worker failed:', fallbackError);
            throw new Error(`Circle detection failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
};

// New function that provides comprehensive analysis matching Gemini service format
export const analyzeFrameWithHough = async (imageData: ImageData): Promise<Omit<FrameAnalysis, 'frame'>> => {
    try {
        // Check if Python server is available
        const isPythonServerAvailable = await checkPythonServerHealth();
        
        if (isPythonServerAvailable) {
            console.log('Using Python API for comprehensive frame analysis');
            return await analyzeFrameWithPythonAPI(imageData);
        } else {
            console.log('Python API not available, falling back to Web Worker for droplets only');
            // Fallback to Web Worker for droplets only, provide defaults for scale and timestamp
            const droplets = await detectCirclesWithWebWorker(imageData);
            const height = imageData.height;
            const width = imageData.width;
            
            return {
                timestamp: "Not Found",
                timestampFound: false,
                scaleFound: false,
                dropletsFound: droplets.length > 0,
                droplets: droplets,
                scale: {
                    x1: Math.floor(width - width * 0.05 - width * 0.15),
                    y1: Math.floor(height - height * 0.05),
                    x2: Math.floor(width - width * 0.05),
                    y2: Math.floor(height - height * 0.05),
                    label: "50 µm (default)",
                    length: Math.floor(width * 0.15)
                }
            };
        }
    } catch (error) {
        console.error('Error in analyzeFrameWithHough:', error);
        
        // If Python API fails, try Web Worker as fallback
        try {
            console.log('Python API failed, falling back to Web Worker');
            const droplets = await detectCirclesWithWebWorker(imageData);
            const height = imageData.height;
            const width = imageData.width;
            
            return {
                timestamp: "Not Found",
                timestampFound: false,
                scaleFound: false,
                dropletsFound: droplets.length > 0,
                droplets: droplets,
                scale: {
                    x1: Math.floor(width - width * 0.05 - width * 0.15),
                    y1: Math.floor(height - height * 0.05),
                    x2: Math.floor(width - width * 0.05),
                    y2: Math.floor(height - height * 0.05),
                    label: "50 µm (default)",
                    length: Math.floor(width * 0.15)
                }
            };
        } catch (fallbackError) {
            console.error('Both Python API and Web Worker failed:', fallbackError);
            throw new Error(`Frame analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
};