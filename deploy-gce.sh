#!/bin/bash

# Deploy Droplet Analysis Application to Google Cloud Engine
# This script sets up the application on a GCE VM

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-your-project-id}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-droplet-analysis-vm}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-micro}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2204-lts-arm64}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
DISK_SIZE="${DISK_SIZE:-10GB}"

echo "🚀 Deploying Droplet Analysis Application to GCE"
echo "================================================"
echo "Project ID: $PROJECT_ID"
echo "Zone: $ZONE"
echo "VM Name: $VM_NAME"
echo "Machine Type: $MACHINE_TYPE"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    echo "   Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set the project
echo "📋 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com

# Check if VM already exists
echo "🔍 Checking if VM instance already exists..."
if gcloud compute instances describe $VM_NAME --zone=$ZONE --quiet &> /dev/null; then
    echo "⚠️  VM instance '$VM_NAME' already exists in zone '$ZONE'"
    echo ""
    echo "What would you like to do?"
    echo "1) Update the application on the existing VM"
    echo "2) Delete the existing VM and create a new one"
    echo "3) Exit without making changes"
    echo ""
    read -p "Please choose an option (1-3): " choice
    
    case $choice in
        1)
            echo "🔄 Updating application on existing VM..."
            UPDATE_EXISTING=true
            ;;
        2)
            echo "🗑️  Deleting existing VM..."
            gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet
            echo "✅ Existing VM deleted"
            
            # Also delete the static IP if it exists
            # STATIC_IP_NAME="${VM_NAME}-ip"
            # if gcloud compute addresses describe $STATIC_IP_NAME --region=${ZONE%-*} --quiet &> /dev/null; then
            #     echo "🗑️  Deleting static IP address..."
            #     gcloud compute addresses delete $STATIC_IP_NAME --region=${ZONE%-*} --quiet
            #     echo "✅ Static IP deleted"
            # fi
            
            UPDATE_EXISTING=false
            ;;
        3)
            echo "👋 Exiting without changes"
            exit 0
            ;;
        *)
            echo "❌ Invalid option. Exiting."
            exit 1
            ;;
    esac
else
    echo "✅ VM instance does not exist. Will create new instance."
    UPDATE_EXISTING=false
fi

# Create the VM (only if not updating existing)
if [ "$UPDATE_EXISTING" = false ]; then
    # Create static IP address
    echo "🌐 Creating static IP address..."
    STATIC_IP_NAME="${VM_NAME}-ip"
    
    if gcloud compute addresses describe $STATIC_IP_NAME --region=${ZONE%-*} --quiet &> /dev/null; then
        echo "✅ Static IP '$STATIC_IP_NAME' already exists"
    else
        gcloud compute addresses create $STATIC_IP_NAME \
            --region=${ZONE%-*} \
            --description="Static IP for droplet analysis VM"
        echo "✅ Static IP '$STATIC_IP_NAME' created"
    fi
    
    # Get the static IP address
    STATIC_IP=$(gcloud compute addresses describe $STATIC_IP_NAME --region=${ZONE%-*} --format='get(address)')
    echo "📍 Static IP address: $STATIC_IP"
    
    echo "🖥️  Creating VM instance with static IP..."
    gcloud compute instances create $VM_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --boot-disk-size=$DISK_SIZE \
        --boot-disk-type=pd-balanced \
        --address=$STATIC_IP \
        --tags=droplet-analysis \
        --metadata=startup-script='#!/bin/bash
            apt-get update
            apt-get install -y docker.io docker-compose git curl
            systemctl start docker
            systemctl enable docker
            usermod -aG docker $USER
            curl -fsSL https://get.docker.com -o get-docker.sh
            sh get-docker.sh
            curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            chmod +x /usr/local/bin/docker-compose
        '
else
    echo "🔄 Using existing VM instance..."
fi

# Wait for VM to be ready (only if we just created it)
if [ "$UPDATE_EXISTING" = false ]; then
    echo "⏳ Waiting for VM to be ready..."
    sleep 60
else
    echo "✅ Using existing VM (no wait needed)"
fi

# Get the external IP
if [ "$UPDATE_EXISTING" = false ]; then
    # Use the static IP we just created
    EXTERNAL_IP=$STATIC_IP
    echo "🌐 VM External IP (Static): $EXTERNAL_IP"
else
    # Get the current IP of the existing VM
    EXTERNAL_IP=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    echo "🌐 VM External IP: $EXTERNAL_IP"
fi

# Create firewall rule (if it doesn't exist)
echo "🔥 Creating firewall rule..."
if gcloud compute firewall-rules describe allow-droplet-analysis --quiet &> /dev/null; then
    echo "✅ Firewall rule 'allow-droplet-analysis' already exists"
else
    gcloud compute firewall-rules create allow-droplet-analysis \
        --allow tcp:80 \
        --source-ranges 0.0.0.0/0 \
        --target-tags droplet-analysis \
        --description "Allow access to droplet analysis application and SSH"
    echo "✅ Firewall rule created"
fi

# Copy application files to VM (using tar for faster transfer)
echo "📁 Preparing application files for transfer..."
# Create a temporary tar.gz file excluding unnecessary files
tar --exclude='.git' \
    --exclude='node_modules' \
    --exclude='.venv' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    --exclude='*.log' \
    --exclude='logs' \
    --exclude='test-*' \
    --exclude='*.md' \
    --exclude='README.md' \
    --exclude='VIDEO_CONVERSION_GUIDE.md' \
    --exclude='DEPLOYMENT_GUIDE.md' \
    -czf droplet-analysis-temp.tar.gz .

echo "📦 Copying compressed application files to VM..."
gcloud compute scp droplet-analysis-temp.tar.gz $VM_NAME:~/droplet-analysis.tar.gz --zone=$ZONE

echo "🗜️  Extracting application files on VM..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    rm -rf ~/droplet-analysis
    mkdir -p ~/droplet-analysis
    tar -xzf ~/droplet-analysis.tar.gz -C ~/droplet-analysis
    rm ~/droplet-analysis.tar.gz
    echo '✅ Application files extracted successfully'
"

# Clean up local tar file
rm droplet-analysis-temp.tar.gz
echo "✅ Application files transferred and extracted"

# Deploy the application
echo "🐳 Deploying application on VM..."
if [ "$UPDATE_EXISTING" = true ]; then
    echo "🔄 Updating existing application..."
    gcloud compute ssh $VM_NAME --zone=$ZONE --command="
        cd ~/droplet-analysis
        echo 'Stopping existing containers...'
        sudo docker-compose down || true
        echo 'Building new containers...'
        sudo docker-compose build
        echo 'Starting updated application...'
        sudo docker-compose up -d
        echo 'Showing recent logs...'
        sudo docker-compose logs --tail=20
    "
else
    echo "🚀 Deploying new application..."
    gcloud compute ssh $VM_NAME --zone=$ZONE --command="
        cd ~/droplet-analysis
        sudo docker-compose build
        sudo docker-compose up -d
        sudo docker-compose logs -f --tail=50
    "
fi

echo ""
if [ "$UPDATE_EXISTING" = true ]; then
    echo "✅ Application update completed!"
else
    echo "✅ Deployment completed!"
fi
echo "🌐 Application URL: http://$EXTERNAL_IP"
echo "🔍 Health Check: http://$EXTERNAL_IP/health"
echo ""
echo "📋 Useful commands:"
echo "  View logs: gcloud compute ssh $VM_NAME --zone=$ZONE --command='cd ~/droplet-analysis && docker-compose logs -f'"
echo "  Restart app: gcloud compute ssh $VM_NAME --zone=$ZONE --command='cd ~/droplet-analysis && docker-compose restart'"
echo "  Stop app: gcloud compute ssh $VM_NAME --zone=$ZONE --command='cd ~/droplet-analysis && docker-compose down'"
echo "  SSH to VM: gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "🔧 To update the application:"
echo "  1. Make your changes locally"
echo "  2. Run this script again and choose option 1 (Update existing VM)"
echo "  OR manually:"
echo "     gcloud compute scp --recurse . $VM_NAME:~/droplet-analysis --zone=$ZONE"
echo "     gcloud compute ssh $VM_NAME --zone=$ZONE --command='cd ~/droplet-analysis && docker-compose up -d --build'"