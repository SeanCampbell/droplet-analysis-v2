// This script runs in a separate, background thread (Web Worker)

const grayscale = (data, width, height) => {
  const gray = new Array(width * height);
  for (let i = 0; i < data.length; i += 4) {
    // Using a weighted average for better grayscale conversion (luminosity method)
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const avg = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[i / 4] = avg;
  }
  return gray;
};

const sobel = (grayData, width, height) => {
    const magnitudes = new Array(grayData.length).fill(0);
    const directions = new Array(grayData.length).fill(0);
    const kernelX = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ];
    const kernelY = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ];

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let gx = 0;
            let gy = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const pixel = grayData[(y + ky) * width + (x + kx)];
                    gx += pixel * kernelX[ky + 1][kx + 1];
                    gy += pixel * kernelY[ky + 1][kx + 1];
                }
            }
            const index = y * width + x;
            magnitudes[index] = Math.sqrt(gx * gx + gy * gy);
            directions[index] = Math.atan2(gy, gx);
        }
    }
    return { magnitudes, directions };
};

const runHoughDetection = (
    imageData,
    minRadius = 20,
    maxRadius = 150,
    edgeThreshold = 50,
    accumulatorThreshold = 85
) => {
    const { width, height, data } = imageData;
    const grayData = grayscale(data, width, height);
    const { magnitudes: edgeData } = sobel(grayData, width, height);
    
    // Fix: Explicitly type accumulator to prevent type errors on votes/values.
    const accumulator: { [key: string]: number } = {};
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (edgeData[y * width + x] > edgeThreshold) {
                for (let r = minRadius; r < maxRadius; r++) {
                    for (let t = 0; t < 360; t += 10) { // Increased step to improve performance
                        const a = Math.round(x - r * Math.cos(t * Math.PI / 180));
                        const b = Math.round(y - r * Math.sin(t * Math.PI / 180));
                         if (a >= 0 && a < width && b >= 0 && b < height) {
                            const key = `${a}|${b}|${r}`;
                            accumulator[key] = (accumulator[key] || 0) + 1;
                        }
                    }
                }
            }
        }
    }
    
    const sortedCircles = Object.entries(accumulator)
        .filter(([, votes]) => votes >= accumulatorThreshold)
        .sort(([, valA], [, valB]) => valB - valA)
        .map(([key]) => {
            const [cx, cy, r] = key.split('|').map(Number);
            return { cx, cy, r };
        });

    const distinctCircles = [];
    for(const circle of sortedCircles) {
        let isDistinct = true;
        // Check if this circle is too close to an already found, stronger circle
        for(const existing of distinctCircles) {
            const dist = Math.hypot(circle.cx - existing.cx, circle.cy - existing.cy);
            if (dist < existing.r) { // Don't add circles whose centers are inside a bigger one
                isDistinct = false;
                break;
            }
        }
        if(isDistinct) {
            distinctCircles.push({ id: distinctCircles.length, ...circle });
            if (distinctCircles.length >= 2) break; // We only need the two most prominent
        }
    }

    return distinctCircles;
};


// Listen for messages from the main thread
self.onmessage = (event) => {
    const { imageData } = event.data;
    try {
        const droplets = runHoughDetection(imageData);
        // Send the result back to the main thread
        self.postMessage(droplets);
    } catch (e) {
        console.error("Error in Hough Worker:", e);
        // In case of an error, you might want to post an error message back
        self.postMessage([]); 
    }
};
