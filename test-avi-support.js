// Test AVI file support
// This can be run in the browser console

function testAVISupport() {
    console.log('Testing AVI file support...');
    
    // Test different AVI file scenarios
    const aviTestFiles = [
        { name: 'test.avi', type: 'video/avi' },
        { name: 'test.avi', type: 'video/x-msvideo' },
        { name: 'test.avi', type: '' }, // No MIME type
        { name: 'test.AVI', type: 'video/avi' }, // Uppercase extension
        { name: 'test.avi', type: 'application/octet-stream' }, // Generic MIME type
    ];

    aviTestFiles.forEach(file => {
        const needsConversion = checkIfConversionNeeded(file);
        console.log(`${file.name} (${file.type || 'no MIME type'}): ${needsConversion ? 'WILL BE CONVERTED' : 'DIRECT SUPPORT'}`);
    });
}

function checkIfConversionNeeded(file) {
    const mimeType = file.type.toLowerCase();
    const fileName = file.name.toLowerCase();
    
    // Web-compatible formats that don't need conversion
    const supportedFormats = [
        'video/mp4',
        'video/webm',
        'video/ogg',
    ];

    // Formats that definitely need conversion (including AVI for maximum compatibility)
    const unsupportedFormats = [
        'video/quicktime', // .mov
        'video/avi', // .avi (all variants - convert for maximum compatibility)
        'video/x-msvideo', // .avi (older format)
        'video/x-ms-wmv', // .wmv
        'video/x-flv', // .flv
        'video/x-matroska', // .mkv
        'video/3gpp', // .3gp
        'video/x-m4v', // .m4v
    ];

    // Check by MIME type first
    if (supportedFormats.includes(mimeType)) {
        return false;
    }

    if (unsupportedFormats.includes(mimeType)) {
        return true;
    }

    // Check by file extension if MIME type is not recognized
    const extension = fileName.split('.').pop()?.toLowerCase();
    const supportedExtensions = ['mp4', 'webm', 'ogg'];
    const unsupportedExtensions = ['mov', 'avi', 'wmv', 'flv', 'mkv', '3gp', 'm4v'];

    if (extension && supportedExtensions.includes(extension)) {
        return false;
    }

    if (extension && unsupportedExtensions.includes(extension)) {
        return true;
    }

    return true; // Default to needing conversion
}

// Run the test
testAVISupport();

console.log('\n✅ AVI Support Summary:');
console.log('- All AVI files will be automatically converted to MP4');
console.log('- This ensures maximum compatibility across different AVI variants');
console.log('- Conversion happens client-side using FFmpeg.js');
console.log('- No server upload required for conversion');
