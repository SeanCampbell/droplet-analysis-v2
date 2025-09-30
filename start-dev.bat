@echo off
REM Start development environment for droplet analysis application
REM This script starts both the Python API server and the frontend development server

echo 🚀 Starting Droplet Analysis Development Environment
echo ==================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python to continue.
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH. Please install Node.js to continue.
    pause
    exit /b 1
)

echo 🔍 Checking port availability...

REM Check if port 5001 is in use
netstat -an | find "5001" | find "LISTENING" >nul
if not errorlevel 1 (
    echo ⚠️  Port 5001 (Python API) is already in use. Please stop the service using this port.
    pause
    exit /b 1
)

REM Check if port 8888 is in use
netstat -an | find "8888" | find "LISTENING" >nul
if not errorlevel 1 (
    echo ⚠️  Port 8888 (Frontend) is already in use. Please stop the service using this port.
    pause
    exit /b 1
)

REM Install Python dependencies if requirements.txt exists
if exist "python-server\requirements.txt" (
    echo 📦 Installing Python dependencies...
    cd python-server
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install Python dependencies
        pause
        exit /b 1
    )
    cd ..
    echo ✅ Python dependencies installed
)

REM Install Node.js dependencies if package.json exists
if exist "package.json" (
    echo 📦 Installing Node.js dependencies...
    npm install
    if errorlevel 1 (
        echo ❌ Failed to install Node.js dependencies
        pause
        exit /b 1
    )
    echo ✅ Node.js dependencies installed
)

echo 🐍 Starting Python API server on port 5001...
cd python-server
start "Python API Server" cmd /k "python app.py"
cd ..

REM Wait a moment for Python server to start
timeout /t 3 /nobreak >nul

echo ⚛️  Starting frontend development server on port 8888...
start "Frontend Server" cmd /k "npm run dev"

REM Wait a moment for frontend server to start
timeout /t 3 /nobreak >nul

echo.
echo 🎉 Development environment is ready!
echo ==================================
echo Frontend: http://localhost:8888
echo Python API: http://localhost:5001
echo API Health Check: http://localhost:5001/health
echo.
echo Both servers are running in separate windows.
echo Close the command windows to stop the services.
echo.
pause

