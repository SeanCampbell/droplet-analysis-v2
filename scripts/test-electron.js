import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('🧪 Testing Electron setup...');

// Test 1: Check if build files exist
import fs from 'fs';

const distElectronDir = path.resolve(__dirname, '../dist-electron');
const requiredFiles = ['main.cjs', 'preload.cjs'];

console.log('📁 Checking build files...');
for (const file of requiredFiles) {
  const filePath = path.join(distElectronDir, file);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${file} exists`);
  } else {
    console.log(`❌ ${file} missing`);
    process.exit(1);
  }
}

// Test 2: Check if dist directory exists
const distDir = path.resolve(__dirname, '../dist');
if (fs.existsSync(distDir)) {
  console.log('✅ dist directory exists');
} else {
  console.log('❌ dist directory missing');
  process.exit(1);
}

// Test 3: Check Electron binary
console.log('⚡ Checking Electron installation...');
try {
  const { execSync } = await import('child_process');
  execSync('electron --version', { stdio: 'pipe' });
  console.log('✅ Electron is installed and accessible');
} catch (error) {
  console.log('❌ Electron not found or not accessible');
  process.exit(1);
}

console.log('✅ All Electron setup tests passed!');
console.log('');
console.log('📋 Next steps:');
console.log('  1. Run "npm run electron:dev" for development with hot reload');
console.log('  2. Run "npm run electron" to test the built application');
console.log('  3. Run "npm run electron:dist" to create distribution packages');
