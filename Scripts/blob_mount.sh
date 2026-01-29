#!/bin/bash
set -e

echo "🔹 Updating system & installing blobfuse2..."
sudo apt update
sudo apt install -y blobfuse2

# -----------------------------
# Paths
# -----------------------------
BLOB_MOUNT=/mnt/blob
CACHE_DIR=/home/azureuser/blobfuse_cache
CONFIG_DIR=/home/azureuser/.blobfuse
CONFIG_FILE=$CONFIG_DIR/config.yaml
USER_NAME=azureuser

# -----------------------------
# Create required directories
# -----------------------------
echo "🔹 Creating directories..."

sudo mkdir -p $BLOB_MOUNT
sudo mkdir -p $CACHE_DIR
mkdir -p $CONFIG_DIR

# -----------------------------
# Fix permissions (CRITICAL)
# -----------------------------
echo "🔹 Fixing permissions..."

sudo chown -R $USER_NAME:$USER_NAME $BLOB_MOUNT
sudo chown -R $USER_NAME:$USER_NAME $CACHE_DIR
sudo chmod 755 $BLOB_MOUNT
sudo chmod 755 $CACHE_DIR

# -----------------------------
# Sanity check (cache write)
# -----------------------------
echo "🔹 Verifying cache write access..."
touch $CACHE_DIR/.cache_test
rm $CACHE_DIR/.cache_test

# -----------------------------
# Unmount if already mounted
# -----------------------------
echo "🔹 Unmounting existing mount (if any)..."
fusermount -u $BLOB_MOUNT 2>/dev/null || true

# -----------------------------
# Mount blob storage
# -----------------------------
echo "🔹 Mounting Azure Blob Storage..."
blobfuse2 mount $BLOB_MOUNT --config-file $CONFIG_FILE

# -----------------------------
# Verify mount
# -----------------------------
echo "🔹 Verifying mount..."
mount | grep $BLOB_MOUNT

touch $BLOB_MOUNT/.mount_test
rm $BLOB_MOUNT/.mount_test

echo "✅ Blob storage successfully mounted at $BLOB_MOUNT"
