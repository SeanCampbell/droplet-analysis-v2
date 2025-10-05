# Electron Setup for Droplet Analysis Lab

This document explains how to set up and use the Electron desktop application version of the Droplet Analysis Lab.

## Prerequisites

- Node.js (version 16 or higher)
- npm or yarn package manager

## Installation

1. Install dependencies:
```bash
npm install
```

2. Install Electron dependencies:
```bash
npm run postinstall
```

3. Test the setup:
```bash
npm run electron:test
```

## Development

### Running in Development Mode

To run the application in development mode with hot reload:

```bash
npm run electron:dev
```

This will:
- Start the Vite development server
- Launch Electron with the development build
- Enable hot reload for both React and Electron

### Building for Production

To build the application for production:

```bash
npm run electron:build
```

This creates a production build in the `dist-electron` directory.

## Packaging and Distribution

### Create Installer Packages

To create platform-specific installer packages:

```bash
# For all platforms (based on your OS)
npm run electron:dist

# For specific platforms (requires cross-compilation setup)
npm run electron:pack
```

### Platform-Specific Builds

The application supports building for:
- **macOS**: Creates a `.dmg` file
- **Windows**: Creates a `.exe` installer
- **Linux**: Creates an `AppImage` file

## File Structure

```
├── electron/
│   ├── main.cjs         # Main Electron process (CommonJS)
│   └── preload.cjs      # Preload script for secure IPC (CommonJS)
├── scripts/
│   ├── build-electron.js # Build script
│   ├── dev-electron.js   # Development script
│   └── test-electron.js  # Test script
├── build/
│   └── entitlements.mac.plist # macOS entitlements
├── types/
│   └── electron.d.ts    # TypeScript definitions
├── services/
│   └── electronService.ts # Electron service utilities
└── dist-electron/       # Built Electron files
    ├── main.cjs
    └── preload.cjs
```

## Features

### Desktop Integration

- **Native file dialogs**: Use system file picker for video uploads
- **Menu bar integration**: Native application menu with keyboard shortcuts
- **Window management**: Proper window controls and behavior
- **Security**: Context isolation and secure IPC communication

### Keyboard Shortcuts

- `Cmd/Ctrl + O`: Open video file
- `Cmd/Ctrl + E`: Export analysis
- `Cmd/Ctrl + Q`: Quit application
- `F11`: Toggle fullscreen
- `Cmd/Ctrl + R`: Reload application

### Platform-Specific Features

#### macOS
- Native menu bar integration
- Proper window controls
- Code signing support (when configured)

#### Windows
- Native installer with NSIS
- Desktop and Start Menu shortcuts
- Proper file associations

#### Linux
- AppImage format for easy distribution
- Desktop integration

## Configuration

### Environment Variables

- `ELECTRON=true`: Indicates running in Electron mode
- `NODE_ENV=development|production`: Environment mode

### Build Configuration

The build configuration is in `electron-builder.json` and includes:
- App metadata and icons
- Platform-specific settings
- File inclusion/exclusion rules
- Code signing configuration

## Security

The application implements several security measures:

1. **Context Isolation**: Renderer process cannot access Node.js APIs directly
2. **Preload Script**: Secure bridge between main and renderer processes
3. **Content Security Policy**: Prevents execution of unsafe scripts
4. **External Link Handling**: Opens external links in default browser

## Troubleshooting

### Common Issues

1. **Build fails**: Ensure all dependencies are installed with `npm install`
2. **Electron won't start**: Check that the main process file exists in `dist-electron/`
3. **File dialogs don't work**: Verify the preload script is properly configured
4. **Menu shortcuts don't work**: Check that menu handlers are properly set up

### Development Tips

1. Use `npm run electron:dev` for development with hot reload
2. Check the Electron DevTools for debugging renderer process issues
3. Use `console.log` in the main process for debugging main process issues
4. The application logs are available in the Electron DevTools console

## Distribution

### Code Signing (Optional)

For production distribution, you may want to configure code signing:

1. **macOS**: Add your Apple Developer certificate
2. **Windows**: Add your code signing certificate
3. **Linux**: No code signing required for AppImage

### Auto-Updater (Future Enhancement)

The current setup doesn't include auto-updater functionality, but it can be added using `electron-updater`.

## Performance Considerations

- The application bundles all dependencies for offline use
- FFmpeg.wasm is included for video processing
- Large video files are processed in chunks to prevent memory issues
- The build process optimizes chunks for better loading performance

## Support

For issues related to the Electron setup, check:
1. The main application documentation
2. Electron documentation: https://www.electronjs.org/docs
3. Electron Builder documentation: https://www.electron.build/
