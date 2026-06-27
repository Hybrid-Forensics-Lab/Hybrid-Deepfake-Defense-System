#!/usr/bin/env bash
# Redeploy the Hybrid Deepfake Defense System web app on the GCP VM.
#   - rebuilds the frontend (uses frontend/.env.production -> VITE_API_URL)
#   - publishes it to /var/www/deepfake (served by nginx on :80)
#   - (re)installs the systemd unit and restarts the API on :8000
#
# Run from the repo root:  bash deploy/deploy.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/4] Building frontend..."
cd frontend
npm install --no-audit --no-fund
npm run build
cd "$REPO_ROOT"

echo "[2/4] Publishing frontend to /var/www/deepfake..."
sudo mkdir -p /var/www/deepfake
sudo rsync -a --delete frontend/dist/ /var/www/deepfake/

echo "[3/4] Installing nginx + systemd config..."
sudo cp deploy/nginx-deepfake.conf /etc/nginx/sites-available/deepfake
sudo ln -sf /etc/nginx/sites-available/deepfake /etc/nginx/sites-enabled/deepfake
sudo rm -f /etc/nginx/sites-enabled/default
sudo cp deploy/deepfake-api.service /etc/systemd/system/deepfake-api.service
sudo systemctl daemon-reload

echo "[4/4] Restarting services..."
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl enable deepfake-api
sudo systemctl restart deepfake-api

echo "Done. UI: http://34.135.192.253  | API: http://34.135.192.253:8000/health"
