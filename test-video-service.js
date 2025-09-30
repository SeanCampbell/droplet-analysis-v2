// Simple test for video conversion service
// This can be run in the browser console

// Test format detection
function testFormatDetection() {
    console.log('Testing video format detection...');
    
    // Mock file objects for testing
    const testFiles = [
        { name: 'test.mp4', type: 'video/mp4' },
        { name: 'test.mov', type: 'video/quicktime' },
        { name: 'test.wmv', type: 'video/x-ms-wmv' },
        { name: 'test.webm', type: 'video/webm' },
        { name: 'test.avi', type: 'video/x-msvideo' },
        { name: 'test.mkv', type: 'video/x-matroska' },
        { name: 'test.3gp', type: 'video/3gpp' },
        { name: 'test.m4v', type: 'video/x-m4v' },
    ];

    testFiles.forEach(file => {
        const needsConversion = checkIfConversionNeeded(file);
        console.log(`${file.name} (${file.type}): ${needsConversion ? 'NEEDS CONVERSION' : 'SUPPORTED'}`);
    });
}

function checkIfConversionNeeded(file) {
    const mimeType = file.type.toLowerCase();
    const fileName = file.name.toLowerCase();
    
    const supportedFormats = [
        'video/mp4',
        'video/webm',
        'video/ogg',
        'video/avi',
    ];

    const unsupportedFormats = [
        'video/quicktime',
        'video/x-msvideo',
        'video/x-ms-wmv',
        'video/x-flv',
        'video/x-matroska',
        'video/3gpp',
        'video/x-m4v',
    ];

    if (supportedFormats.includes(mimeType)) {
        return false;
    }

    if (unsupportedFormats.includes(mimeType)) {
        return true;
    }

    // Check by extension
    const extension = fileName.split('.').pop()?.toLowerCase();
    const supportedExtensions = ['mp4', 'webm', 'ogg', 'avi'];
    const unsupportedExtensions = ['mov', 'wmv', 'flv', 'mkv', '3gp', 'm4v'];

    if (extension && supportedExtensions.includes(extension)) {
        return false;
    }

    if (extension && unsupportedExtensions.includes(extension)) {
        return true;
    }

    return true; // Default to needing conversion
}

// Run the test
testFormatDetection();
