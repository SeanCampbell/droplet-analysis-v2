import { GoogleGenAI, Type } from "@google/genai";
import type { FrameAnalysis } from '../types';

if (!process.env.API_KEY) {
  throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const responseSchema = {
  type: Type.OBJECT,
  properties: {
    timestamp: {
      type: Type.STRING,
      description: 'The timestamp text found in the image, e.g., "0:00:28.255".',
    },
    timestampFound: { type: Type.BOOLEAN, description: 'Set to true if a timestamp was found, otherwise false.' },
    scale: {
      type: Type.OBJECT,
      description: 'The scale bar found in the image. Only return if found.',
      properties: {
        x1: { type: Type.NUMBER }, y1: { type: Type.NUMBER },
        x2: { type: Type.NUMBER }, y2: { type: Type.NUMBER },
        label: { type: Type.STRING },
      },
    },
    scaleFound: { type: Type.BOOLEAN, description: 'Set to true if a scale bar was found, otherwise false.' },
    droplets: {
      type: Type.ARRAY,
      description: 'An array of the two most prominent droplets. Only return if found.',
      items: {
        type: Type.OBJECT,
        properties: {
          cx: { type: Type.NUMBER }, cy: { type: Type.NUMBER }, r: { type: Type.NUMBER },
        },
      },
    },
    dropletsFound: { type: Type.BOOLEAN, description: 'Set to true if at least one droplet was found, otherwise false.' },
  },
  required: ['timestamp', 'timestampFound', 'scaleFound', 'dropletsFound'],
};

export const analyzeFrameWithGemini = async (imageBase64: string): Promise<Omit<FrameAnalysis, 'frame'>> => {
    const imagePart = { inlineData: { mimeType: 'image/jpeg', data: imageBase64 } };
    const textPart = {
      text: `Analyze this microscope image. Identify the two main droplets, the scale bar, and the timestamp.
      1. Droplets: Find the two most prominent circular droplets. Provide the center coordinates (cx, cy) and the radius (r).
      2. Scale Bar: Locate the horizontal scale bar. The bar is always horizontal, so its start and end y-coordinates should be identical. Provide the start (x1, y1) and end (x2, y2) coordinates of the line and its text label.
      3. Timestamp: Read the text that looks like a timestamp (e.g., "Live Time: 0:00:28.255"). Extract only the time value.
      The origin (0,0) is the top-left corner. For each element, also return a boolean flag (e.g., dropletsFound) indicating if you successfully identified it. If an element is not found, omit its data field and set its 'Found' flag to false.`,
    };

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: { parts: [imagePart, textPart] },
      config: { responseMimeType: 'application/json', responseSchema: responseSchema },
    });

    const jsonString = response.text.trim();
    const result = JSON.parse(jsonString) as Partial<Omit<FrameAnalysis, 'frame'>>;
    
    // Ensure required structures exist to prevent downstream errors
    if (!result.droplets) result.droplets = [];
    if (!result.scale) result.scale = {} as any;
    if (!result.timestamp) result.timestamp = "N/A";

    if (result.droplets) {
      result.droplets.forEach((d: any, index: number) => d.id = index);
    }
    if (result.scale && result.scale.x1 !== undefined) {
       result.scale.y2 = result.scale.y1;
       result.scale.length = Math.abs(result.scale.x2 - result.scale.x1);
    }

    return result as Omit<FrameAnalysis, 'frame'>;
};
