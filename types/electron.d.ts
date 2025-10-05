// Type definitions for Electron API exposed through preload script

export interface ElectronAPI {
  // File operations
  showOpenDialog: () => Promise<{
    canceled: boolean;
    filePaths: string[];
  }>;
  
  showSaveDialog: (options: {
    title?: string;
    defaultPath?: string;
    filters?: Array<{
      name: string;
      extensions: string[];
    }>;
  }) => Promise<{
    canceled: boolean;
    filePath?: string;
  }>;
  
  showMessageBox: (options: {
    type?: 'info' | 'warning' | 'error' | 'question';
    title?: string;
    message?: string;
    detail?: string;
    buttons?: string[];
    defaultId?: number;
    cancelId?: number;
  }) => Promise<{
    response: number;
    checkboxChecked?: boolean;
  }>;
  
  // Menu event listeners
  onMenuOpenVideo: (callback: () => void) => void;
  onMenuExportAnalysis: (callback: () => void) => void;
  
  // Remove listeners
  removeAllListeners: (channel: string) => void;
  
  // Platform info
  platform: string;
  
  // App info
  versions: {
    node: string;
    chrome: string;
    electron: string;
  };
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
