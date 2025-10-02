#!/bin/bash

echo "🔍 Production Debugging Script"
echo "=============================="

# Get VM details from user
read -p "Enter your GCE VM name: " VM_NAME
read -p "Enter your GCE zone (e.g., us-central1-a): " ZONE

echo ""
echo "🔍 Step 1: Checking container status..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    echo 'Container Status:'
    docker-compose ps
    echo ''
    echo 'Container Logs (last 20 lines):'
    docker-compose logs --tail=20
"

echo ""
echo "🔍 Step 2: Testing internal connectivity..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    echo 'Testing frontend to backend connectivity:'
    docker-compose exec frontend wget -qO- http://backend:5001/health || echo '❌ Frontend cannot reach backend'
    echo ''
    echo 'Testing backend health directly:'
    docker-compose exec backend curl -f http://localhost:5001/health || echo '❌ Backend health check failed'
"

echo ""
echo "🔍 Step 3: Testing nginx configuration..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    echo 'Nginx configuration test:'
    docker-compose exec frontend nginx -t
    echo ''
    echo 'Nginx access logs (last 10 lines):'
    docker-compose exec frontend tail -10 /var/log/nginx/access.log 2>/dev/null || echo 'No access logs found'
    echo ''
    echo 'Nginx error logs (last 10 lines):'
    docker-compose exec frontend tail -10 /var/log/nginx/error.log 2>/dev/null || echo 'No error logs found'
"

echo ""
echo "🔍 Step 4: Testing external access..."
echo "Testing health endpoint:"
curl -f http://$(gcloud compute instances describe $VM_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')/health || echo "❌ Health endpoint not accessible"

echo ""
echo "Testing API endpoint:"
curl -X POST http://$(gcloud compute instances describe $VM_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')/api/analyze-frame \
  -H "Content-Type: application/json" \
  -d '{"image":"test","min_radius":20,"max_radius":100}' || echo "❌ API endpoint not accessible"

echo ""
echo "🔍 Step 5: Network inspection..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    echo 'Docker networks:'
    docker network ls
    echo ''
    echo 'Network details:'
    docker network inspect droplet-analysis-v2_droplet-analysis-network 2>/dev/null || echo 'Network not found'
    echo ''
    echo 'Container IPs:'
    docker-compose exec frontend ip addr show | grep inet
    docker-compose exec backend ip addr show | grep inet
"

echo ""
echo "🔍 Step 6: Port binding check..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    echo 'Port 80 binding:'
    netstat -tlnp | grep :80 || echo 'Port 80 not bound'
    echo ''
    echo 'Docker port mappings:'
    docker port \$(docker-compose ps -q frontend) 2>/dev/null || echo 'No port mappings found'
"

echo ""
echo "✅ Debugging complete!"
echo ""
echo "📋 Common fixes based on results:"
echo "1. If containers aren't running: docker-compose up -d"
echo "2. If nginx config is invalid: check nginx.frontend.conf"
echo "3. If network connectivity fails: restart containers"
echo "4. If external access fails: check firewall rules"
