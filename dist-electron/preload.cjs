const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // File operations
  showOpenDialog: () => ipcRenderer.invoke('show-open-dialog'),
  showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
  showMessageBox: (options) => ipcRenderer.invoke('show-message-box', options),
  
  // Menu event listeners
  onMenuOpenVideo: (callback) => {
    ipcRenderer.on('menu-open-video', callback);
  },
  onMenuExportAnalysis: (callback) => {
    ipcRenderer.on('menu-export-analysis', callback);
  },
  
  // Remove listeners
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  },
  
  // Platform info
  platform: process.platform,
  
  // App info
  versions: {
    node: process.versions.node,
    chrome: process.versions.chrome,
    electron: process.versions.electron
  }
});

// Security: Prevent the renderer from accessing Node.js APIs
window.addEventListener('DOMContentLoaded', () => {
  // Remove any existing Node.js globals that might have leaked
  delete window.require;
  delete window.exports;
  delete window.module;
  
  // Log that preload script has loaded
  console.log('Electron preload script loaded successfully');
});
