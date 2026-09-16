# Cattle-Vision

**Computer Vision and GeoAI for Cattle Detection in Drone Imagery**

Cattle-Vision is an experimental **Computer Vision / GeoAI project** for detecting cattle in high-resolution drone imagery acquired over a rural cattle farm.

The project explores how aerial imagery, geospatial information and deep learning can be combined to support **automated cattle inventory and monitoring**.

The project is being developed incrementally, starting with a carefully structured dataset and annotation workflow before moving to machine-learning experiments.

---

## Project Overview

The current workflow starts with images captured by a **DJI Air 3S** drone and progressively transforms them into a dataset suitable for computer-vision experiments.

```text
Drone Imagery
      │
      ▼
Metadata Extraction
      │
      ▼
Dataset Processing
      │
      ▼
Flight Group Analysis
      │
      ▼
Dataset Inspection
      │
      ▼
Cattle Annotation
      │
      ▼
YOLO Dataset
      │
      ▼
Object Detection
      │
      ▼
GeoAI Applications
```

The initial objective is deliberately simple:

> **Detect individual cattle in aerial imagery.**

The project will later explore more advanced tasks such as distinguishing calves and bulls, estimating cattle locations, and integrating detections with geographic information.

---

## Current Dataset

The initial dataset consists of **30 aerial photographs** collected during multiple drone flights.

The images contain different acquisition conditions, including:

* Nadir imagery
* Oblique imagery
* Different flight altitudes
* Different cattle densities
* Partially occluded animals
* Small cattle in the image
* Vegetation and other visually similar objects

All images in the current dataset were captured using the same drone and camera configuration.

### Current dataset characteristics

| Property               | Value           |
| ---------------------- | --------------- |
| Drone                  | DJI Air 3S      |
| Images                 | 30              |
| Resolution             | 4096 × 3072     |
| Focal length           | 8.67 mm         |
| 35 mm equivalent       | 24 mm           |
| Approx. horizontal FOV | 73.7°           |
| Image formats          | JPEG            |
| GPS metadata           | Available       |
| Camera orientations    | Nadir / Oblique |
| Flight groups          | 18              |

The dataset is intentionally small at this stage. The initial goal is to validate the **data pipeline, annotation methodology and model-training workflow** before expanding the dataset.

---

# Data Processing Workflow

## 1. Metadata Extraction

The first step is extracting the information embedded in the original drone images.

The project uses **ExifTool** to extract EXIF/XMP metadata.

Relevant information includes:

* Acquisition date and time
* GPS coordinates
* Absolute and relative altitude
* Camera orientation
* Gimbal pitch/yaw/roll
* Focal length
* Image dimensions
* Camera model
* Flight information

The raw metadata is preserved and is not modified by subsequent processing.

```text
Raw Images
    │
    ▼
ExifTool
    │
    ├── images_metadata.json
    └── images_metadata.csv
```

This separation keeps the original metadata reproducible and allows later processing steps to be repeated without accessing the images again.

---

## 2. Dataset Processing

The raw metadata is transformed into a **canonical clean dataset**.

`process_dataset.py` performs validation and derives additional information from the camera metadata.

Examples include:

* Camera orientation classification
* Off-nadir angle
* Vertical field of view
* Nadir ground coverage
* Nadir-equivalent GSD

The resulting dataset contains **one record per image**.

```text
images_metadata.csv
        │
        ▼
process_dataset.py
        │
        ├── dataset_clean.csv
        └── dataset_clean.json
```

The clean dataset becomes the main metadata source for the rest of the pipeline.

---

## 3. Flight Group Analysis

Images captured during the same flight are grouped together using their temporal and spatial relationships.

The current heuristic considers:

* Time between consecutive images
* Geographic distance between consecutive images
* Changes between nadir and oblique camera orientations

This produces a `flight_group` identifier for every image.

The grouping is important later because randomly splitting individual images can cause **data leakage** when very similar images from the same flight appear in both training and validation datasets.

Instead, the project will eventually perform dataset splitting at the **flight-group level**.

---

## 4. Dataset Visualization

Before machine learning, the dataset is visually inspected.

`visualize_dataset.py` generates an overview containing the original images together with relevant acquisition information:

* Image ID
* Date/time
* Flight group
* Camera orientation
* Altitude
* Camera angle
* GSD
* Focal length
* Field of view

This provides a quick way to inspect the diversity and structure of the dataset.

---

# Cattle Annotation

The first computer-vision task is **object detection**.

The annotation workflow uses **CVAT (Computer Vision Annotation Tool)**.

The current annotation class is:

```text
cattle
```

All identifiable bovines are assigned to this single class:

* cows
* bulls
* calves
* different colors
* small animals
* partially occluded animals

Each animal receives an individual **bounding box**.

### Annotation principles

```text
One animal → one bounding box

Overlapping animals → separate bounding boxes

Partially visible animal → box the visible animal

Ambiguous object → do not annotate

Stone / vegetation / shadow → background
```

Camera orientation is **not encoded as a class**. Nadir and oblique images remain part of the same cattle-detection task.

This information is retained as image metadata and can later be used to analyze model performance under different acquisition conditions.

---

# YOLO Annotation Format

The annotations are exported from CVAT in **YOLO format**.

Each annotation follows:

```text
class_id center_x center_y width height
```

with coordinates normalized between `0` and `1`.

For example:

```text
0 0.554750 0.688599 0.012034 0.019131
```

The current class mapping is:

```text
0 → cattle
```

The exported annotations are visually validated by converting the normalized coordinates back into pixel coordinates and drawing them over the original images.

```text
CVAT
 │
 ▼
YOLO annotations
 │
 ▼
visualize_yolo_annotations.py
 │
 ▼
Visual validation
```

This step ensures that the annotation files are correctly aligned with the original imagery before training begins.

---

# Technology Stack

## Current

### Python

The main data-processing language.

Used for:

* Metadata processing
* Dataset construction
* Geospatial calculations
* Flight analysis
* Dataset visualization
* Annotation validation

### ExifTool

Used to extract detailed EXIF/XMP metadata from DJI imagery.

### PyProj

Used for geospatial calculations based on geographic coordinates and the WGS84 ellipsoid.

Current applications include:

* Geodesic distances
* Geographic bearings

### Shapely

Planned/current geospatial geometry dependency for operations involving geometric objects.

### Pillow

Used to inspect and visualize YOLO bounding-box annotations directly on the original images.

### CVAT

Used for manual object annotation and creation of the initial training dataset.

### Git / GitHub

Used for version control and documenting the development of the project.

---

# Planned Machine Learning Stack

The next stage is to introduce deep-learning-based object detection.

## PyTorch

**PyTorch** will provide the underlying deep-learning framework for experimentation and model training.

It will be used to explore:

* Neural-network training
* GPU acceleration
* Dataset pipelines
* Model evaluation
* Transfer learning

## YOLO

The first object-detection experiments will use the **YOLO family of models**.

YOLO is particularly suitable for this project because the initial task is:

> Locate individual cattle in an image using bounding boxes.

The initial model will be trained as a **single-class detector**:

```text
cattle
```

The first training experiment will use the six images that have already been annotated. This is not intended to measure final model performance; it is a **pipeline validation experiment**.

The goal is to verify that:

```text
CVAT
  ↓
YOLO annotations
  ↓
Dataset
  ↓
PyTorch / YOLO
  ↓
Training
  ↓
Inference
```

works correctly end-to-end.

---

# Future Development

The project is intended to evolve beyond simple cattle detection.

Potential future directions include:

### Multi-class cattle detection

The current single-class model may later distinguish between:

```text
cattle
bull
calf
```

This will only be introduced after the single-class detection pipeline has been validated.

### Larger training dataset

The current 30 images represent an initial experimental dataset.

Future datasets will include:

* More flights
* More geographic areas
* Different altitudes
* Different illumination conditions
* More cattle densities
* More occlusion cases
* More challenging backgrounds

### Geospatial cattle positioning

Because the source imagery contains GPS and camera information, detections could eventually be connected to geographic coordinates.

This opens the possibility of producing:

```text
Drone Image
     │
     ▼
Cattle Detection
     │
     ▼
Pixel Position
     │
     ▼
Camera / Flight Geometry
     │
     ▼
Geographic Position
     │
     ▼
GIS / Farm Map
```

The long-term goal is therefore not simply image classification, but **geospatially-aware cattle detection and analysis**.

### Farm management integration

A future version could connect the computer-vision pipeline with farm-management systems, allowing detected cattle information to be combined with:

* Farm maps
* Pastures
* Water bodies
* Roads
* Animal records
* Historical observations
* Other agricultural data

This would move the project toward a practical **GeoAI application for precision livestock management**.

---

# Project Structure

```text
cattle-vision/
│
├── data/
│   ├── raw/
│   │   └── images/
│   │
│   ├── metadata/
│   │   ├── images_metadata.json
│   │   ├── images_metadata.csv
│   │   ├── dataset_clean.csv
│   │   ├── dataset_clean.json
│   │   └── flight_groups_summary.csv
│   │
│   ├── inspection/
│   │   └── yolo_annotations/
│   │
│   └── annotations/
│       └── test_sample_yolo/
│
├── docs/
│   └── annotation_guidelines.md
│
├── scripts/
│   ├── extract_metadata.py
│   ├── process_dataset.py
│   ├── analyze_flights.py
│   ├── visualize_dataset.py
│   ├── visualize_yolo_annotations.py
│   ├── utils.py
│   └── geo.py
│
└── README.md
```

The project is structured so that the data-processing stages are separated from the machine-learning stage.

This makes it possible to independently inspect and validate each step of the pipeline.

---

# Project Status

### Completed

* [x] Drone image acquisition
* [x] EXIF/XMP metadata extraction
* [x] Metadata cleaning and normalization
* [x] Camera orientation analysis
* [x] Flight grouping
* [x] Dataset visualization
* [x] Annotation workflow in CVAT
* [x] Initial cattle annotation
* [x] YOLO annotation export
* [x] YOLO annotation visual validation

### In Progress

* [ ] Initial YOLO object-detection experiment
* [ ] PyTorch training pipeline
* [ ] Model inference and inspection
* [ ] Annotation of remaining images

### Future

* [ ] Larger annotated dataset
* [ ] Flight-group-based train/validation/test split
* [ ] Model evaluation
* [ ] Detection of calves and bulls
* [ ] Geospatial localization of detections
* [ ] Integration with GIS and farm-management data

---

# Why This Project?

Cattle monitoring is traditionally dependent on manual observation and field work.

Drone imagery provides a way to observe large areas efficiently, while Computer Vision can potentially automate part of the process.

The interesting aspect of Cattle-Vision is the combination of three areas:

**Computer Vision**

Detecting and characterizing objects directly from aerial imagery.

**Geospatial Computing**

Maintaining the geographic context of the imagery and eventually converting image detections into geographic information.

**Artificial Intelligence**

Using deep-learning models to transform raw aerial imagery into actionable spatial information.

The project therefore serves both as a practical experiment and as a study of how **GeoAI can be applied to real-world agricultural problems**.

---

## Author

**Marcelo Galvão**

Computer Science · Geoinformatics · GeoAI · Computer Vision · GIS

This project is part of an ongoing exploration of intelligent geospatial applications combining software engineering, geographic data and artificial intelligence.
