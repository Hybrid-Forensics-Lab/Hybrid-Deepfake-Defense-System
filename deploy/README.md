# Deployment — Hybrid Deepfake Defense System

Live demo (while the VM is running):

- **UI:** http://34.135.192.253
- **API:** http://34.135.192.253:8000  (Swagger at `/docs`)

## Architecture

- **Backend**: FastAPI (`backend/`) served by uvicorn on port 8000, run as the
  systemd service `deepfake-api` in the `webapp-env` conda environment. Loads
  CLIP ViT-L/14 + the raw-feature adv-aware probe (`/detect`) and ArcFace + MTCNN
  for PGD cloaking (`/protect`). Single worker so models load once.
- **Frontend**: Vite + React, built to `frontend/dist/`, published to
  `/var/www/deepfake`, served by nginx on port 80 (SPA fallback to `index.html`).
- **Firewall**: GCP allows TCP ingress on 80 and 8000.

## Redeploy

```bash
bash deploy/deploy.sh
```

## Service management

```bash
sudo systemctl status deepfake-api      # API status
sudo journalctl -u deepfake-api -f      # API logs
sudo systemctl restart deepfake-api     # restart API
sudo systemctl reload nginx             # reload web server
```

## Notes

- If the VM's external IP changes, update `frontend/.env.production`
  (`VITE_API_URL`) and the URLs above, then rerun `deploy/deploy.sh`.
- **Stop the VM when not in use** to save GCP credit (billing alert at $75).
  The systemd service auto-starts the API on boot.
