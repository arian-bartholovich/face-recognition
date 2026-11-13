# Face Recognition with OpenCV

A **computer vision** application in **Python** that detects and identifies faces in images using **OpenCV**. It trains an **LBPH (Local Binary Patterns Histograms)** face recognizer on a labeled image dataset, detects faces with a **Haar cascade classifier**, and displays each result with a bounding box, predicted identity, and confidence score.

## Overview

This project implements a complete classical face recognition pipeline: dataset preparation, model training, face detection, and identity prediction, with a simple desktop file picker for choosing test images.

## How It Works

1. **Dataset preparation:** each subfolder of `faces_dataset/` represents one person. Images are loaded in grayscale and assigned numeric labels mapped back to folder names.
2. **Training:** an `LBPHFaceRecognizer` is trained on the labeled face images.
3. **Face detection:** a pre-trained Haar cascade (`haarcascade_frontalface_default.xml`) locates faces in the selected test image.
4. **Recognition:** each detected face region is passed to the recognizer, which returns a predicted label and a confidence value (for LBPH, lower values mean a closer match).
5. **Visualization:** the image is displayed with bounding boxes and annotated labels, and results are printed to the console.

## Key Features

- **End-to-end pipeline:** data loading, training, detection, and recognition in one script
- **Multi-person support:** add a new person by adding a new folder of images
- **Multi-face detection:** identifies every face found in a single image
- **Annotated output:** bounding boxes with name and confidence overlaid on the image
- **Interactive image selection:** native file dialog (Tkinter) for choosing test images
- **Robust input handling:** unreadable images are skipped with a warning instead of crashing

## Tech Stack

| Area              | Technologies                                   |
| ----------------- | ---------------------------------------------- |
| Language          | Python 3                                       |
| Computer vision   | OpenCV (`opencv-contrib-python`), Haar cascades, LBPH |
| Numerical         | NumPy                                          |
| UI                | Tkinter (file dialog)                          |

## Project Structure

```
face-recognition/
├── main.py            # Training, detection, and recognition pipeline
├── download.py        # [describe: e.g., helper for collecting dataset images]
├── requirements.txt   # Python dependencies
├── faces_dataset/     # Training images, one subfolder per person
└── test_images/       # Images for testing the recognizer
```

## Getting Started

### Prerequisites

- Python 3.8+
- A desktop environment (the app uses a GUI file dialog and an image window)

### Installation

```bash
git clone https://github.com/arian-bartholovich/face-recognition.git
cd face-recognition
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `cv2.face` requires the **`opencv-contrib-python`** package, not just `opencv-python`.

### Dataset Setup

Organize training images with one folder per person:

```
faces_dataset/
├── person_a/
│   ├── img1.jpg
│   └── img2.jpg
└── person_b/
    ├── img1.jpg
    └── img2.jpg
```

Only use images you have the right to use, and respect the privacy of the people shown.

### Run

```bash
python main.py
```

Select a test image when prompted. The recognizer prints each prediction and opens a window with the annotated result. Press any key to close it.

## Limitations and Future Work

- LBPH always returns the closest known identity, so adding a **confidence threshold** to label faces as "unknown" would reduce false matches
- Upgrading to **deep-learning embeddings** (e.g., FaceNet or dlib) for higher accuracy under varied lighting and angles
- Model persistence (`recognizer.save()` / `read()`) to avoid retraining on every run
- Command-line arguments, evaluation metrics, and unit tests
- Real-time recognition from a webcam feed

## Ethical Use

Face recognition involves biometric data. This project is for **educational purposes**. Obtain consent before using images of private individuals, and comply with applicable privacy laws such as GDPR.

## Author

**Arian Bartholovich**
GitHub: [@arian-bartholovich](https://github.com/arian-bartholovich)
