import { build } from 'vite';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function buildElectron() {
  console.log('🔨 Building Electron application...');
  
  try {
    // Set Electron environment variable
    process.env.ELECTRON = 'true';
    
    // Build the React app with Vite
    console.log('📦 Building React app...');
    await build({
      configFile: path.resolve(__dirname, '../vite.config.ts'),
      mode: 'production'
    });
    
    // Copy Electron files to dist-electron
    console.log('📁 Copying Electron files...');
    const distElectronDir = path.resolve(__dirname, '../dist-electron');
    
    if (!fs.existsSync(distElectronDir)) {
      fs.mkdirSync(distElectronDir, { recursive: true });
    }
    
    // Copy main.cjs and preload.cjs
    fs.copyFileSync(
      path.resolve(__dirname, '../electron/main.cjs'),
      path.resolve(distElectronDir, 'main.cjs')
    );
    
    fs.copyFileSync(
      path.resolve(__dirname, '../electron/preload.cjs'),
      path.resolve(distElectronDir, 'preload.cjs')
    );
    
    console.log('✅ Electron build completed successfully!');
    console.log('📂 Output directory: dist-electron/');
    
  } catch (error) {
    console.error('❌ Build failed:', error);
    process.exit(1);
  }
}

buildElectron();
