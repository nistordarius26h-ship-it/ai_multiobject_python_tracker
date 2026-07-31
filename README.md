# Real-Time AI Object Detection & Tracking System
![Python](https://img.shields.io/badge/Python-3.11-blue)
![AI](https://img.shields.io/badge/AI-Computer%20Vision-green)
![GPU](https://img.shields.io/badge/GPU-CUDA%2FROCm-orange)
![License](https://img.shields.io/badge/License-MIT-red)

## Overview

This project presents a **real-time computer vision system** for object detection, multi-object tracking, and motion analytics on any live video stream, including aerial drone footage.

Originally built and tested using live footage from a DJI Mini 3 drone, the system was later adapted into the core vision pipeline for an autonomous ground robot's person-following capability.

> **Disclaimer**
>
> This project is intended for educational and research purposes. Ensure you have the right to capture and process any live video source you connect it to.

---

## Features

- Real-time object detection & tracking (YOLO + ByteTrack)
- Works with any live video source (drone, YouTube Live, RTSP, webcam)
- Automatic GPU hardware detection (NVIDIA CUDA / AMD ROCm / CPU fallback)
- Motion trail visualization per tracked object
- Real-time speed estimation (km/h) with exponential smoothing
- Occlusion handling to prevent bounding-box flicker
- Multi-class filtering (people, vehicles, animals)
- Live on-screen telemetry (FPS, device, object counts, runtime)

---

## System Architecture

- Live Video Source (drone / stream / camera)
- Frame Capture (streamlink / OpenCV)
- YOLO Detection Model
- ByteTrack Multi-Object Tracker
- Kinematics Engine (speed, trails, occlusion handling)
- Annotated Video Output

---

## Technologies

- Python
- OpenCV
- Ultralytics YOLO
- ByteTrack
- PyTorch (CUDA / ROCm)
- Streamlink

---

## Requirements

- Python 3.11+
- Windows or Linux
- NVIDIA GPU (CUDA), AMD GPU (ROCm/HIP), or CPU
- Internet connection if using live streams (YouTube Live, RTSP, etc.)

---

## Hardware

- GPU-accelerated workstation (NVIDIA CUDA or AMD ROCm/HIP)
> **Keep in mind**
>
> This project was developed and tested on an **AMD Radeon RX 7800 XT**. Setting up GPU acceleration on AMD hardware (ROCm/HIP) can be considerably more challenging than on NVIDIA CUDA due to differences in software support and compatibility across operating systems, PyTorch builds, and drivers. If GPU acceleration is unavailable or difficult to configure on your system, the application will automatically fall back to CPU execution, although with reduced performance.
- Any live video source (drone, IP camera, webcam, streaming platform)

---

## Project Structure

```
tracker.py           # Main detection/tracking pipeline
requirements.txt      # Dependencies
media/                # Demo clips, screenshots
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nistordarius26h-ship-it/ai_multiobject_python_tracker.git
cd ai_multiobject_python_tracker
```

Install the required packages:

```bash
pip install -r requirements.txt
```

> **AMD GPU (ROCm) users**
>
> PyTorch with ROCm is **not installed automatically** through `requirements.txt`. Install the appropriate ROCm-enabled PyTorch build for your ROCm version from the official PyTorch installation guide before installing the remaining dependencies.

Install PyTorch:

> Install the appropriate PyTorch build for your hardware:

- **NVIDIA:** CUDA
- **AMD:** ROCm
- **CPU:** CPU-only

See the official PyTorch installation guide:
https://pytorch.org/get-started/locally/

Then verify GPU support:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run the application:

```bash
python tracker.py
```

---

## Future Work

- Command-line arguments for source/model selection
- Multi-camera support
- On-device (edge) inference for lower latency
- Re-identification across camera handoffs
- Web dashboard for remote monitoring

---

## License

MIT License

---

## Author

Nistor Darius

Artificial Intelligence • Computer Vision • Robotics
