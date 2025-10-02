# Droplet Analysis Application - Deployment Guide

This guide covers deploying the Droplet Analysis application to Google Cloud Engine (GCE) using Docker containers.

## Prerequisites

1. **Google Cloud SDK**: Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install)
2. **Docker**: Install [Docker](https://docs.docker.com/get-docker/)
3. **Google Cloud Project**: Create a project in the [Google Cloud Console](https://console.cloud.google.com/)

## Quick Deployment

### 1. Set Environment Variables

```bash
export PROJECT_ID="your-project-id"
export ZONE="us-central1-a"
export VM_NAME="droplet-analysis-vm"
```

### 2. Run Deployment Script

```bash
./deploy-gce.sh
```

The script will:
- Create a GCE VM instance
- Install Docker and Docker Compose
- Copy your application files
- Build and deploy the containers
- Set up firewall rules
- Provide you with the application URL

## Manual Deployment

### 1. Create GCE VM

```bash
gcloud compute instances create droplet-analysis-vm \
    --zone=us-central1-a \
    --machine-type=e2-standard-2 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB \
    --tags=droplet-analysis
```

### 2. Install Docker on VM

```bash
gcloud compute ssh droplet-analysis-vm --zone=us-central1-a
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

### 3. Copy Application Files

```bash
gcloud compute scp --recurse . droplet-analysis-vm:~/droplet-analysis --zone=us-central1-a
```

### 4. Deploy Application

```bash
gcloud compute ssh droplet-analysis-vm --zone=us-central1-a
cd ~/droplet-analysis
docker-compose build
docker-compose up -d
```

### 5. Create Firewall Rule

```bash
gcloud compute firewall-rules create allow-droplet-analysis \
    --allow tcp:80,tcp:443,tcp:5001 \
    --source-ranges 0.0.0.0/0 \
    --target-tags droplet-analysis
```

## Application Architecture

The deployment includes:

- **Frontend**: React/Vite application served as static files
- **Backend**: Python Flask API with OpenCV and Tesseract
- **Reverse Proxy**: Nginx for load balancing and SSL termination
- **Container Orchestration**: Docker Compose

## Configuration

### Environment Variables

- `PORT`: Application port (default: 5001)
- `HOST`: Bind address (default: 0.0.0.0)
- `DEBUG`: Debug mode (default: false)

### Resource Limits

- **Memory**: 2GB limit, 1GB reserved
- **CPU**: 1.0 limit, 0.5 reserved

## Monitoring and Maintenance

### View Logs

```bash
gcloud compute ssh droplet-analysis-vm --zone=us-central1-a
cd ~/droplet-analysis
docker-compose logs -f
```

### Restart Application

```bash
gcloud compute ssh droplet-analysis-vm --zone=us-central1-a
cd ~/droplet-analysis
docker-compose restart
```

### Update Application

```bash
# 1. Make changes locally
# 2. Copy files to VM
gcloud compute scp --recurse . droplet-analysis-vm:~/droplet-analysis --zone=us-central1-a

# 3. Rebuild and restart
gcloud compute ssh droplet-analysis-vm --zone=us-central1-a
cd ~/droplet-analysis
docker-compose up -d --build
```

## Security Considerations

### Firewall Rules

The deployment creates a firewall rule allowing:
- Port 80 (HTTP)
- Port 443 (HTTPS)
- Port 5001 (Direct API access)

### SSL/TLS

For production, configure SSL certificates:

1. Place certificates in `./ssl/` directory
2. Uncomment HTTPS configuration in `nginx.conf`
3. Update domain name in nginx configuration

### Access Control

Consider implementing:
- API authentication
- Rate limiting (configured in nginx)
- IP whitelisting
- VPN access

## Scaling

### Horizontal Scaling

To scale the application:

1. Create multiple VM instances
2. Use a load balancer
3. Configure shared storage for logs

### Vertical Scaling

To increase resources:

1. Stop the VM
2. Change machine type
3. Restart the VM

```bash
gcloud compute instances stop droplet-analysis-vm --zone=us-central1-a
gcloud compute instances set-machine-type droplet-analysis-vm \
    --machine-type=e2-standard-4 --zone=us-central1-a
gcloud compute instances start droplet-analysis-vm --zone=us-central1-a
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 80, 443, and 5001 are available
2. **Memory issues**: Increase VM memory or optimize application
3. **Docker build failures**: Check Dockerfile and dependencies
4. **Permission issues**: Ensure proper file permissions

### Health Checks

The application includes health checks:
- Container health check: `http://localhost:5001/health`
- Application health check: `http://your-vm-ip/health`

### Logs

Check logs in order:
1. Docker container logs: `docker-compose logs`
2. Nginx logs: `docker-compose logs nginx`
3. System logs: `journalctl -u docker`

## Cost Optimization

### VM Sizing

- **Development**: e2-micro (1 vCPU, 1GB RAM)
- **Production**: e2-standard-2 (2 vCPU, 8GB RAM)
- **High Load**: e2-standard-4 (4 vCPU, 16GB RAM)

### Preemptible Instances

For cost savings, use preemptible instances:

```bash
gcloud compute instances create droplet-analysis-vm \
    --preemptible \
    --zone=us-central1-a \
    --machine-type=e2-standard-2
```

### Auto-shutdown

Set up auto-shutdown for development environments:

```bash
gcloud compute instances add-metadata droplet-analysis-vm \
    --metadata=shutdown-script='#!/bin/bash
    cd ~/droplet-analysis
    docker-compose down'
```

## Support

For issues and questions:
1. Check the logs first
2. Review this deployment guide
3. Check the main README.md for application-specific issues
