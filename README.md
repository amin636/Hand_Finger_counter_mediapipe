# Hand Finger Counter with MediaPipe

A real-time finger counter application using **Google MediaPipe** and **OpenCV**.

![Demo](demo.gif)

## ✨ Features

- Real-time hand detection and landmark tracking using MediaPipe
- Accurate finger counting
- Live counter display on video feed
- Supports webcam and video file input
- Fast and lightweight performance
- Simple and clean interface

## 🛠️ Technologies Used

- Python 3.8+
- [MediaPipe](https://mediapipe.dev)
- OpenCV
- NumPy

## 📥 Installation & Setup

### Prerequisites
```bash
pip install mediapipe opencv-python numpy
```
### Clone the Repository
```bash
git clone https://github.com/amin636/hand-finger-counter-mediapipe.git
cd hand-finger-counter-mediapipe
```
### Download Model
Before running, download the model:
```bash
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
###Run the Application
```bash
python finger_counter.py
