#!/bin/bash

echo "🔍 Testing Production Connectivity"
echo "================================="

# Test 1: Check if containers are running
echo "1. Checking container status..."
docker-compose ps

echo ""
echo "2. Testing backend health directly..."
docker-compose exec backend curl -f http://localhost:5001/health

echo ""
echo "3. Testing frontend to backend connectivity..."
docker-compose exec frontend wget -qO- http://backend:5001/health

echo ""
echo "4. Testing nginx configuration..."
docker-compose exec frontend nginx -t

echo ""
echo "5. Testing external health endpoint..."
curl -f http://localhost/health

echo ""
echo "6. Testing external API endpoint..."
curl -X POST http://localhost/api/analyze-frame \
  -H "Content-Type: application/json" \
  -d '{"image":"test","min_radius":20,"max_radius":100}'

echo ""
echo "7. Checking nginx logs..."
echo "Access logs (last 5 lines):"
docker-compose exec frontend tail -5 /var/log/nginx/access.log 2>/dev/null || echo "No access logs"

echo ""
echo "Error logs (last 5 lines):"
docker-compose exec frontend tail -5 /var/log/nginx/error.log 2>/dev/null || echo "No error logs"

echo ""
echo "8. Checking backend logs..."
docker-compose logs backend --tail=10

echo ""
echo "✅ Connectivity test complete!"
