# Production Debug Guide

## 🐛 Troubleshooting Frontend-Backend Communication

### **Common Issues and Solutions**

#### **1. Frontend Can't Reach Backend API**

**Symptoms:**
- Frontend loads but analysis never completes
- Console shows network errors
- API requests timeout or fail

**Debug Steps:**

1. **Check if containers are running:**
   ```bash
   docker-compose ps
   ```

2. **Check container logs:**
   ```bash
   # Frontend logs
   docker-compose logs frontend
   
   # Backend logs
   docker-compose logs backend
   ```

3. **Test health endpoint directly:**
   ```bash
   curl http://localhost/health
   ```

4. **Test API endpoint directly:**
   ```bash
   curl -X POST http://localhost/api/analyze-frame \
     -H "Content-Type: application/json" \
     -d '{"image":"test","min_radius":20,"max_radius":100}'
   ```

5. **Check nginx configuration:**
   ```bash
   docker-compose exec frontend nginx -t
   ```

#### **2. CORS Issues**

**Symptoms:**
- Browser console shows CORS errors
- Requests fail with "Access-Control-Allow-Origin" errors

**Solution:**
- Backend CORS is configured to allow all origins (`*`)
- If still having issues, check that nginx is properly proxying requests

#### **3. Network Connectivity Issues**

**Symptoms:**
- Containers can't communicate with each other
- Backend not reachable from frontend

**Debug Steps:**

1. **Check Docker network:**
   ```bash
   docker network ls
   docker network inspect droplet-analysis-v2_droplet-analysis-network
   ```

2. **Test internal connectivity:**
   ```bash
   # From frontend container
   docker-compose exec frontend wget -qO- http://backend:5001/health
   
   # From backend container
   docker-compose exec backend curl http://frontend:80/
   ```

#### **4. Backend Service Issues**

**Symptoms:**
- Backend container exits or crashes
- Health check fails
- API returns 500 errors

**Debug Steps:**

1. **Check backend logs:**
   ```bash
   docker-compose logs backend
   ```

2. **Check backend health:**
   ```bash
   docker-compose exec backend curl http://localhost:5001/health
   ```

3. **Check Python dependencies:**
   ```bash
   docker-compose exec backend python -c "import cv2, flask, numpy; print('Dependencies OK')"
   ```

### **🔧 Debugging Tools**

#### **1. Test Production API (HTML)**
Access `http://your-domain/test-production-api.html` to run interactive tests:
- Health check test
- API endpoint test
- Environment information

#### **2. Browser Developer Tools**
1. Open browser dev tools (F12)
2. Go to Network tab
3. Try to analyze a frame
4. Look for failed requests and error messages

#### **3. Container Shell Access**
```bash
# Access frontend container
docker-compose exec frontend sh

# Access backend container
docker-compose exec backend bash
```

### **📋 Common Fixes**

#### **Fix 1: Restart Services**
```bash
docker-compose down
docker-compose up --build -d
```

#### **Fix 2: Clear Docker Cache**
```bash
docker-compose down
docker system prune -f
docker-compose up --build -d
```

#### **Fix 3: Check Port Conflicts**
```bash
# Check if port 80 is in use
sudo netstat -tlnp | grep :80
sudo lsof -i :80
```

#### **Fix 4: Verify Environment Variables**
```bash
# Check if NODE_ENV is set correctly in production
docker-compose exec frontend env | grep NODE_ENV
```

### **🚨 Emergency Recovery**

If the application is completely broken:

1. **Stop all services:**
   ```bash
   docker-compose down
   ```

2. **Remove all containers and images:**
   ```bash
   docker-compose down --rmi all --volumes
   ```

3. **Rebuild from scratch:**
   ```bash
   docker-compose up --build -d
   ```

### **📊 Monitoring Commands**

```bash
# Check service status
docker-compose ps

# Monitor logs in real-time
docker-compose logs -f

# Check resource usage
docker stats

# Check network connectivity
docker-compose exec frontend ping backend
docker-compose exec backend ping frontend
```

### **🔍 Log Analysis**

**Frontend (nginx) logs:**
- Look for 502/503/504 errors (backend unavailable)
- Check for proxy errors

**Backend (Python) logs:**
- Look for import errors
- Check for API request processing errors
- Monitor memory usage

**Docker logs:**
- Check for container startup errors
- Look for network connectivity issues

### **📞 Getting Help**

If you're still having issues:

1. **Collect logs:**
   ```bash
   docker-compose logs > debug-logs.txt
   ```

2. **Check system resources:**
   ```bash
   docker system df
   free -h
   df -h
   ```

3. **Test with minimal setup:**
   ```bash
   # Test just the backend
   docker-compose up backend
   
   # Test just the frontend
   docker-compose up frontend
   ```
