# Real-time Emotion and Hand Tracking with MediaPipe and DeepFace

![Project Status](https://img.shields.io/badge/Status-In%20Progress-blue)
![License](https://img.shields.io/github/license/MEGHNAKORIM/emotion_detection)

---

## 📸 Project Overview

This repository contains Python scripts for real-time computer vision applications, leveraging the power of **OpenCV**, **MediaPipe**, and **DeepFace**. The project demonstrates:

1.  **Real-time Emotion Recognition:** Detects and displays dominant emotions (e.g., happy, sad, angry) from faces in a live webcam feed.
2.  **Real-time Face Mesh Detection:** Visualizes a detailed 3D face mesh on detected faces, showing hundreds of facial landmarks.
3.  **Real-time Hand Tracking & Distance Measurement:** Detects hands, draws landmarks, calculates the approximate distance of the hand from the camera, and triggers an audio alert if a hand is too close.

This project is ideal for understanding basic real-time computer vision techniques, utilizing pre-trained models from popular libraries.

---

## ✨ Features

* **Emotion Detection:** Utilizes DeepFace for accurate emotion analysis.
* **Face Mesh Visualization:** Implements MediaPipe Face Mesh for granular facial landmark detection.
* **Hand Tracking:** Uses MediaPipe Hands for robust hand and finger tracking.
* **Distance Estimation:** Calculates approximate hand-to-camera distance using perceived width and a calibrated focal length.
* **Proximity Alert:** Triggers an audible beep (Windows only) when a hand is too close to the camera.
* **Real-time Processing:** All functionalities operate on a live webcam feed.
* **Clear Visualizations:** Displays bounding boxes, landmarks, and text overlays for intuitive understanding.
* **Modular Code:** Organized into functions for better readability and maintainability.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.7+**
* **pip** (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MEGHNAKORIM/emotion_detection.git](https://github.com/MEGHNAKORIM/emotion_detection.git)
    cd emotion_detection
    ```

2.  **Install the required Python packages:**
    It's highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

    Now install the libraries:
    ```bash
    pip install opencv-python mediapipe deepface numpy winsound # winsound is Windows-specific
    ```
    * **Note for macOS/Linux users:** The `winsound` library is specific to Windows. If you are on macOS or Linux, `pip` might skip it, or you might get a warning. The hand distance alert feature will not work, but the rest of the project will function. You can replace `winsound.Beep` with an alternative sound library like `playsound` if needed.

---

## 🏃 How to Run

Each functionality is encapsulated in its own script (assuming you name them accordingly, e.g., `emotion_detection.py`, `face_mesh.py`, `hand_distance.py`).

Navigate to the project root directory (`emotion_detection`) in your terminal.

### 1. Run Emotion Recognition

This script will open your webcam feed and display the dominant emotion detected on faces.

```bash
python emotion_recognition.py # Or whatever you named your emotion script
