export interface Droplet {
  id: number;
  cx: number;
  cy: number;
  r: number;
}

export interface Scale {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  length: number;
}

export interface FrameAnalysis {
  frame: number;
  timestamp: string;
  droplets: Droplet[];
  scale: Scale;
  timestampFound?: boolean;
  scaleFound?: boolean;
  dropletsFound?: boolean;
}
