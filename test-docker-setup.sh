#!/bin/bash

echo "🐳 Testing Multi-Container Docker Setup"
echo "======================================"

# Build and start the services
echo "📦 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Test backend health
echo "🔍 Testing backend health..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Backend health check passed"
else
    echo "❌ Backend health check failed"
fi

# Test frontend
echo "🔍 Testing frontend..."
if curl -f http://localhost/ > /dev/null 2>&1; then
    echo "✅ Frontend is accessible"
else
    echo "❌ Frontend is not accessible"
fi

# Test API endpoint
echo "🔍 Testing API endpoint..."
if curl -f http://localhost/api/health > /dev/null 2>&1; then
    echo "✅ API endpoint is accessible"
else
    echo "❌ API endpoint is not accessible"
fi

echo ""
echo "📋 Service Status:"
docker-compose ps

echo ""
echo "🌐 Access URLs:"
echo "  Frontend: http://localhost/"
echo "  API Health: http://localhost/health"
echo "  API Endpoint: http://localhost/api/analyze-frame"
echo ""
echo "📝 To stop services: docker-compose down"
echo "📝 To view logs: docker-compose logs -f"
