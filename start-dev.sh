#!/bin/bash

# Start development environment for droplet analysis application
# This script starts both the Python API server and the frontend development server

echo "🚀 Starting Droplet Analysis Development Environment"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 to continue."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js to continue."
    exit 1
fi

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Check if ports are available
echo "🔍 Checking port availability..."
if ! check_port 5001; then
    echo "❌ Port 5001 (Python API) is already in use. Please stop the service using this port."
    exit 1
fi

if ! check_port 8888; then
    echo "❌ Port 8888 (Frontend) is already in use. Please stop the service using this port."
    exit 1
fi

# Install Python dependencies if requirements.txt exists
if [ -f "python-server/requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    cd python-server
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Python dependencies"
        exit 1
    fi
    cd ..
    echo "✅ Python dependencies installed"
fi

# Install Node.js dependencies if package.json exists
if [ -f "package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Node.js dependencies"
        exit 1
    fi
    echo "✅ Node.js dependencies installed"
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null
        echo "✅ Python API server stopped"
    fi
    if [ ! -z "$NODE_PID" ]; then
        kill $NODE_PID 2>/dev/null
        echo "✅ Frontend server stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start Python API server
echo "🐍 Starting Python API server on port 5001..."
cd python-server
python3 app.py &
PYTHON_PID=$!
cd ..

# Wait a moment for Python server to start
sleep 3

# Check if Python server started successfully
if ! kill -0 $PYTHON_PID 2>/dev/null; then
    echo "❌ Failed to start Python API server"
    exit 1
fi

echo "✅ Python API server started (PID: $PYTHON_PID)"

# Start frontend development server
echo "⚛️  Starting frontend development server on port 8888..."
npm run dev &
NODE_PID=$!

# Wait a moment for frontend server to start
sleep 3

# Check if frontend server started successfully
if ! kill -0 $NODE_PID 2>/dev/null; then
    echo "❌ Failed to start frontend server"
    cleanup
    exit 1
fi

echo "✅ Frontend server started (PID: $NODE_PID)"
echo ""
echo "🎉 Development environment is ready!"
echo "=================================="
echo "Frontend: http://localhost:8888"
echo "Python API: http://localhost:5001"
echo "API Health Check: http://localhost:5001/health"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
wait

