#!/bin/bash

# Test script for droplet analysis deployment
# This script tests the Docker deployment locally

set -e

echo "🧪 Testing Droplet Analysis Deployment"
echo "======================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it first."
    exit 1
fi

echo "✅ Docker and docker-compose are available"

# Build the application
echo "🔨 Building Docker image..."
docker-compose build

# Start the application
echo "🚀 Starting application..."
docker-compose up -d

# Wait for application to start
echo "⏳ Waiting for application to start..."
sleep 30

# Test health endpoint
echo "🔍 Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/health)

if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed (HTTP $HEALTH_RESPONSE)"
    echo "📋 Application logs:"
    docker-compose logs --tail=20
    exit 1
fi

# Test frontend
echo "🌐 Testing frontend..."
FRONTEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/)

if [ "$FRONTEND_RESPONSE" = "200" ]; then
    echo "✅ Frontend is accessible"
else
    echo "❌ Frontend test failed (HTTP $FRONTEND_RESPONSE)"
fi

# Test API endpoint with a simple request
echo "🔬 Testing API endpoint..."
API_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:5001/api/analyze-frame \
    -H "Content-Type: application/json" \
    -d '{"image": "test"}')

if [ "$API_RESPONSE" = "400" ]; then
    echo "✅ API endpoint is responding (400 expected for invalid request)"
else
    echo "❌ API endpoint test failed (HTTP $API_RESPONSE)"
fi

# Show container status
echo "📊 Container status:"
docker-compose ps

echo ""
echo "✅ Deployment test completed!"
echo "🌐 Application URL: http://localhost:5001"
echo "🔍 Health Check: http://localhost:5001/health"
echo ""
echo "📋 Useful commands:"
echo "  View logs: docker-compose logs -f"
echo "  Stop app: docker-compose down"
echo "  Restart app: docker-compose restart"
echo ""
echo "🧹 To clean up:"
echo "  docker-compose down"
echo "  docker-compose down --volumes --rmi all"
