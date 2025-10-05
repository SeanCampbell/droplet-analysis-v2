import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function startDevServer() {
  console.log('🚀 Starting Electron development environment...');
  
  // Start Vite dev server
  const viteProcess = spawn('npm', ['run', 'dev'], {
    stdio: 'inherit',
    shell: true,
    env: { ...process.env, ELECTRON: 'false' }
  });
  
  // Wait for Vite to start, then start Electron
  setTimeout(() => {
    console.log('⚡ Starting Electron...');
    
    const electronProcess = spawn('electron', ['.'], {
      stdio: 'inherit',
      shell: true,
      env: { ...process.env, NODE_ENV: 'development' }
    });
    
    electronProcess.on('close', (code) => {
      console.log(`Electron process exited with code ${code}`);
      viteProcess.kill();
      process.exit(code);
    });
    
    electronProcess.on('error', (error) => {
      console.error('Failed to start Electron:', error);
      viteProcess.kill();
      process.exit(1);
    });
    
  }, 3000); // Wait 3 seconds for Vite to start
  
  // Handle cleanup
  process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down development servers...');
    viteProcess.kill();
    process.exit(0);
  });
  
  process.on('SIGTERM', () => {
    viteProcess.kill();
    process.exit(0);
  });
}

startDevServer();
