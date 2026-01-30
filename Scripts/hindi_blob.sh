#!/bin/bash
set -e

# ====== VARIABLES (change these) ======
CONTAINER_NAME="hindiCheckpoint"
MOUNT_POINT="/mnt/blob-${CONTAINER_NAME}"
CACHE_DIR="/mnt/blobfuse_cache_${CONTAINER_NAME}"
TMP_DIR="/mnt/blobfuse_tmp_${CONTAINER_NAME}"
CONFIG_FILE="$HOME/.blobfuse/config-${CONTAINER_NAME}.yaml"
USER_NAME="azureuser"
# =====================================

echo "🔹 Updating system & installing blobfuse2..."
sudo apt update
sudo apt install -y blobfuse2

echo "🔹 Ensuring blobfuse config directory exists..."
mkdir -p ~/.blobfuse

echo "🔹 Checking config file..."
if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Config file not found: $CONFIG_FILE"
  echo "Create it first before running this script."
  exit 1
fi

echo "🔹 Creating mount, cache, and temp directories..."
sudo mkdir -p "$MOUNT_POINT"
sudo mkdir -p "$CACHE_DIR"
sudo mkdir -p "$TMP_DIR"

echo "🔹 Fixing ownership..."
sudo chown -R ${USER_NAME}:${USER_NAME} "$MOUNT_POINT"
sudo chown -R ${USER_NAME}:${USER_NAME} "$CACHE_DIR"
sudo chown -R ${USER_NAME}:${USER_NAME} "$TMP_DIR"

echo "🔹 Fixing permissions..."
sudo chmod 700 "$CACHE_DIR" "$TMP_DIR"

echo "🔹 Cleaning cache & temp directories..."
rm -rf "$CACHE_DIR"/*
rm -rf "$TMP_DIR"/*

echo "🔹 Unmounting existing mount (if any)..."
fusermount3 -u "$MOUNT_POINT" 2>/dev/null || true

echo "🔹 Mounting Azure Blob container: $CONTAINER_NAME"
blobfuse2 mount "$MOUNT_POINT" --config-file "$CONFIG_FILE"

echo "✅ Blob container '$CONTAINER_NAME' mounted at $MOUNT_POINT"
