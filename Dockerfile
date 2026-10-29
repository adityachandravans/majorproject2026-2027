# ==========================================
# Multi-stage Dockerfile for Multi-Agent System
# ==========================================

# Stage 1: Build React 18 + Vite Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend + Static Hosting
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Optimize PyTorch size: install CPU-only wheel first to keep image lightweight (~150MB instead of ~2.5GB CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source directories
COPY config/ ./config/
COPY env/ ./env/
COPY agents/ ./agents/
COPY baseline/ ./baseline/
COPY marl/ ./marl/
COPY metrics_logging/ ./metrics_logging/
COPY backend/ ./backend/
COPY checkpoints/ ./checkpoints/

# Copy compiled frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Default port
ENV PORT=8000
EXPOSE 8000

# Start Uvicorn respecting dynamic PORT env (Railway, Render, Cloud Run, HF Spaces)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
