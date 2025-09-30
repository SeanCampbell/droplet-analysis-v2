import type { Droplet } from '../types';

// This function creates a worker from a Blob to bypass cross-origin restrictions.
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
        console.error("Failed to create worker from blob:", error);
        throw error;
    }
};


export const detectCirclesWithHough = (
    imageData: ImageData
): Promise<Droplet[]> => {
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