# 🐄 Cattle-Vision

**Drone & Computer Vision — transformando imagens aéreas em informação prática de campo.**

Uso drones e visão computacional para transformar imagens aéreas em informação prática de campo — começando pela **contagem automática de gado**, com **detecção de vegetação** e **qualidade de pasto** como próximos passos.

---

## 🎯 What It Does

Cattle-Vision detects individual cattle in high-resolution drone imagery using a deep-learning object-detection pipeline.

> **Detect individual cattle in aerial imagery.**

The project starts simple — one class, one task — and grows from there. Later stages will distinguish calves and bulls, estimate cattle locations, and integrate detections with geographic information.

---

## 🖼️ Result

<p align="center">
  <img src="assets/detection3.png" alt="Cattle detection on drone imagery" width="720"/>
</p>

---

## 🔄 Workflow

Every stage is a separate, inspectable step — from raw drone footage to a trained model.

```text
Drone Imagery (DJI Air 3S)
      │
      ▼
Metadata Extraction          ← ExifTool · EXIF/XMP
      │
      ▼
Dataset Processing           ← Python · PyProj
      │
      ▼
Flight Group Analysis        ← Python · Shapely
      │
      ▼
Dataset Inspection           ← Python · Pillow
      │
      ▼
Cattle Annotation            ← CVAT
      │
      ▼
YOLO Dataset                 ← Python · Pillow (validation)
      │
      ▼
Object Detection             ← PyTorch · YOLO
      │
      ▼
GeoAI Applications           ← GeoPandas · GIS
