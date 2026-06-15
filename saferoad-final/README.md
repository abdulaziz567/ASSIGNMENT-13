---
title: SafeRoad AI — Seatbelt Violation Detection
emoji: 🚦
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---

# 🚦 SafeRoad AI — Seatbelt Violation Detection & License Plate Logging

An intelligent traffic monitoring system that:
- Detects drivers with/without seatbelts using **YOLOv8**
- Identifies violations and finds the nearest **license plate**
- Extracts plate text using **Tesseract OCR**
- Stores all violations in **PostgreSQL**
- Displays results in a **Streamlit** dashboard

---

## 📋 Project Objective (Assignment)

> Design and implement an intelligent traffic monitoring system that detects drivers,
> classifies them into `Person_with_seatbelt` and `Person_without_seatbelt`,
> identifies violations, detects license plates, extracts plate text using OCR,
> stores data in a database, and provides a user interface.

---

## 🔄 Workflow Pipeline

```
Input Image/Video
    ↓
YOLO Model 1 — Seatbelt Detection
    ├── Person_with_seatbelt
    └── Person_without_seatbelt
    ↓
Step 1: Identify all detected persons
    ↓
Step 2: Filter persons WITHOUT seatbelt
    ↓
Step 3: For each violating person:
    ├── YOLO Model 2 — License Plate Detection
    ├── Find nearest license plate (bounding box distance)
    ├── Crop plate region
    ├── Apply Tesseract OCR
    └── Extract plate text
    ↓
Step 4: Store results in PostgreSQL
    ↓
Step 5: Display output in Streamlit UI
    ↓
Step 6: Deployed on HuggingFace Spaces
```

---

## 📁 Project Structure

```
saferoad-ai/
├── app.py                  ← Main Streamlit application
├── line_crossing.py        ← Virtual line crossing tracker (bonus feature)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file (HuggingFace Spaces config)
│
├── models/
│   ├── __init__.py
│   └── detector.py         ← SeatbeltDetector class (YOLO + OCR pipeline)
│
├── db/
│   ├── __init__.py
│   └── database.py         ← ViolationDB class (PostgreSQL integration)
│
└── train/
    ├── __init__.py
    └── train_models.py     ← Training script for both YOLO models
```

---

## 🚀 Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tesseract OCR binary** also required:
> - Ubuntu/Debian: `sudo apt install tesseract-ocr`
> - macOS: `brew install tesseract`
> - Windows: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Train the models

**Download datasets and train seatbelt model:**
```bash
python train/train_models.py \
    --task seatbelt \
    --api_key YOUR_ROBOFLOW_API_KEY \
    --epochs 50 \
    --imgsz 640
```

**Train license plate model:**
```bash
python train/train_models.py \
    --task plate \
    --api_key YOUR_ROBOFLOW_API_KEY \
    --epochs 50 \
    --imgsz 640
```

**Copy trained weights to models/ folder:**
```bash
cp runs/train/seatbelt_yolov8/weights/best.pt models/seatbelt_yolov8.pt
cp runs/train/plate_yolov8/weights/best.pt     models/license_plate_yolov8.pt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📊 Datasets

| Dataset | Source |
|---------|--------|
| Seatbelt Detection | https://universe.roboflow.com/renuka-ed2ia/seat-belt-detection-c2nia |
| License Plate Detection | https://universe.roboflow.com/manasa-n-n/car-license-plate-detection-maehj |

---

## 🗃 Database Schema

```sql
CREATE TABLE violations (
    id             SERIAL PRIMARY KEY,
    plate_number   TEXT,                          -- Extracted OCR plate text
    violation_type TEXT DEFAULT 'No Seatbelt',   -- Type of violation
    confidence     REAL,                          -- YOLO detection confidence
    timestamp      TIMESTAMPTZ DEFAULT NOW(),     -- Auto-recorded time
    image_name     TEXT,                          -- Source file name
    frame_number   INT                            -- Video frame number
);
```

### Connect PostgreSQL

Enter your PostgreSQL connection URL in the sidebar:
```
postgresql://username:password@host:5432/database_name
```

If no DB is connected, violations are stored in session memory and can be exported as CSV.

---

## 🖥 UI Features

| Tab | Description |
|-----|-------------|
| 📷 Image Detection | Upload image → Run YOLO+OCR → See violations |
| 🎬 Video Detection | Upload video → Frame-by-frame processing → Live preview |
| 🗃 Violation Log | View/export all violations from DB or session |
| 📏 Line Crossing | Virtual detection line → Trigger inspection on crossing |

---

## ⚙ Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| OCR | Tesseract OCR |
| Database | PostgreSQL (psycopg2) |
| UI | Streamlit |
| Deployment | HuggingFace Spaces |
| Image Processing | OpenCV, Pillow |

---

## 📝 Notes

- Without trained `.pt` model files, the app runs in **demo mode** with simulated detections.
- Place `seatbelt_yolov8.pt` and `license_plate_yolov8.pt` in the `models/` folder to enable real inference.
- The app auto-detects model availability on startup and shows a status banner.
"# assignment" 
