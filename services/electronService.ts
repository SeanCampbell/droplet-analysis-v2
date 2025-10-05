// Service for handling Electron-specific functionality

export const isElectron = (): boolean => {
  return typeof window !== 'undefined' && window.electronAPI !== undefined;
};

export const getElectronAPI = () => {
  if (!isElectron()) {
    throw new Error('Electron API not available');
  }
  return window.electronAPI;
};

export const showOpenDialog = async () => {
  if (!isElectron()) {
    // Fallback for web version - create a file input
    return new Promise<{ canceled: boolean; filePaths: string[] }>((resolve) => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'video/*';
      input.multiple = false;
      
      input.onchange = (e) => {
        const target = e.target as HTMLInputElement;
        const files = target.files;
        if (files && files.length > 0) {
          resolve({ canceled: false, filePaths: [files[0].name] });
        } else {
          resolve({ canceled: true, filePaths: [] });
        }
      };
      
      input.oncancel = () => {
        resolve({ canceled: true, filePaths: [] });
      };
      
      input.click();
    });
  }
  
  return getElectronAPI().showOpenDialog();
};

export const showSaveDialog = async (options: {
  title?: string;
  defaultPath?: string;
  filters?: Array<{ name: string; extensions: string[] }>;
}) => {
  if (!isElectron()) {
    // Fallback for web version - use browser download
    return { canceled: false, filePath: options.defaultPath || 'download' };
  }
  
  return getElectronAPI().showSaveDialog(options);
};

export const showMessageBox = async (options: {
  type?: 'info' | 'warning' | 'error' | 'question';
  title?: string;
  message?: string;
  detail?: string;
  buttons?: string[];
  defaultId?: number;
  cancelId?: number;
}) => {
  if (!isElectron()) {
    // Fallback for web version - use browser alert/confirm
    if (options.type === 'question' && options.buttons) {
      const result = confirm(`${options.message}\n\n${options.detail || ''}`);
      return { response: result ? 0 : 1 };
    } else {
      alert(`${options.message}\n\n${options.detail || ''}`);
      return { response: 0 };
    }
  }
  
  return getElectronAPI().showMessageBox(options);
};

export const setupElectronMenuHandlers = (
  onOpenVideo: () => void,
  onExportAnalysis: () => void
) => {
  if (!isElectron()) return;
  
  const api = getElectronAPI();
  
  api.onMenuOpenVideo(onOpenVideo);
  api.onMenuExportAnalysis(onExportAnalysis);
  
  // Cleanup function
  return () => {
    api.removeAllListeners('menu-open-video');
    api.removeAllListeners('menu-export-analysis');
  };
};

export const getPlatformInfo = () => {
  if (!isElectron()) {
    return {
      platform: 'web',
      versions: {
        node: 'N/A',
        chrome: navigator.userAgent.includes('Chrome') ? 'Web' : 'N/A',
        electron: 'N/A'
      }
    };
  }
  
  const api = getElectronAPI();
  return {
    platform: api.platform,
    versions: api.versions
  };
};
