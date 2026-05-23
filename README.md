# LAPUK App Backend

## Overview

Flask backend for **LAPUK: Waste Classification App**, developed as a CCS6 (Applications Development and Emerging Technologies) project and structured as an API service for computer vision workflows. This service handles waste-image inference using a YOLO model and generates data-driven heatmap images consumed by the Android frontend.

## Features

- REST API for mobile integration
- Base64 image ingestion and object detection pipeline
- YOLO-powered inference using a trained local model file (`ai/gd_l_e15.pt`)
- Structured detection response (class + confidence)
- Rendered prediction image returned as base64
- Heatmap image generation from dataset CSV (`data/LAPUK_heatmapDataset.csv`)
- CORS support for cross-origin client access

## API Endpoints

- `GET /`  
  Connection test endpoint (`connected!`)

- `POST /detect`  
  Accepts base64 image payload and returns:
  - plotted detection image (base64)
  - list of detections and confidence scores

- `GET /render-heatmap`  
  Returns generated heatmap as PNG image

## Tech Stack

- **Language:** Python 3.11
- **Framework:** Flask, Flask-Cors, Gunicorn
- **ML/CV:** Ultralytics YOLO, Torch, Torchvision, OpenCV
- **Data/Visualization:** NumPy, Pandas, Matplotlib

## Requirements

- Python 3.11
- `requirements.txt` dependencies
- Model file at `ai/gd_l_e15.pt`
