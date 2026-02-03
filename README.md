# HandFlow

**Real-Time Hand Gesture Recognition for Touchless Human-Computer Interaction**

> **Note:** This is an early-stage prototype demonstrating core capabilities. The project is under active development with significant features and optimizations planned. Current implementation serves as a proof-of-concept for the underlying technology.

[Demo Video](#demo) | [Features](#features) | [Architecture](#architecture) | [Getting Started](#getting-started)

---

## Overview

HandFlow is an end-to-end gesture recognition system that enables touchless interaction with computers through free-space hand gestures. The system combines deep learning-based gesture classification with computer vision-based spatial calibration to support:

- **Free-space gesture control** — Navigate, click, scroll, and trigger shortcuts without touching any device
- **Virtual touchscreen** — Transform any non-touch display into a touch-enabled surface using ArUco marker calibration
- **Paper macro pads** — Use printed ArUco markers as physical button interfaces mapped to OS-level controls

## Demo

<!-- Replace with your actual demo video link -->
[![Demo Video](https://img.shields.io/badge/Demo-Watch%20Video-red?style=for-the-badge&logo=youtube)](YOUR_DEMO_LINK_HERE)

*Click above to watch the prototype demonstration*

---

## Features

### Gesture Recognition
- **11 gesture classes** including swipes, clicks, touch, and navigation gestures
- **Per-hand detection** — Independent left/right hand tracking with gesture-specific actions
- **Customizable mappings** — Map any gesture to keyboard shortcuts, mouse actions, or application launches

### Spatial Calibration
- **ArUco-based homography** — 4-corner marker system for precise screen coordinate mapping
- **Missing marker recovery** — Estimates occluded corners using remaining visible markers
- **Sub-pixel accuracy** — Per-corner offset calibration for precise touch mapping

### Real-Time Performance
- **~25 FPS** on standard consumer hardware (CPU-only inference)
- **TFLite quantization** for optimized model size (~0.6 MB) and inference speed
- **Smoothing filters** (OneEuro) for stable cursor tracking and reduced jitter

---

## Architecture

### System Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Camera    │───▶│  MediaPipe   │───▶│   Feature   │───▶│   Temporal   │
│   Input     │    │  Hand Track  │    │  Engineer   │    │     CNN      │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                              │                   │
                                              ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   OS-Level  │◀───│    Action    │◀───│  Smoothing  │◀───│   Gesture    │
│   Control   │    │   Executor   │    │  & Voting   │    │ Prediction   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### Model Architecture

**Temporal Convolutional Network (TCN)** with velocity-gated attention:

- **Input:** 12-frame sequences × 88 engineered features
- **Velocity Gating:** Custom `SoftMotionWeighting` layer that dynamically weights features based on motion magnitude, suppressing noise during idle periods
- **Temporal Processing:** Dilated causal convolutions (dilation rates: 1, 2, 4) with residual connections
- **Aggregation:** Combined average + max pooling for temporal summarization
- **Output:** 11-class softmax classification

**Alternative architectures available:** LSTM, GRU, 1D-CNN, Transformer

### Feature Engineering (88 features)

| Feature Type | Dimensions | Description |
|-------------|------------|-------------|
| Relative Positions | 63 | Wrist-normalized (x, y, z) for 21 landmarks |
| Inter-finger Distances | 5 | Thumb to each fingertip + index PIP |
| Raw Positions | 9 | Absolute coords for thumb tip, index MCP/tip |
| Velocity Features | 6 | Frame-to-frame motion of index finger |
| Finger Angles | 5 | Bending angle at PIP joint per finger |

### Data Augmentation

Online geometric augmentation during training:
- Gaussian noise injection
- Time warping
- Landmark dropout
- Uniform scaling & 2D rotation
- Z-axis transformations (scale, shift, proportional stretch)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | TensorFlow/Keras + TFLite |
| Hand Tracking | MediaPipe Hands |
| Computer Vision | OpenCV + ArUco |
| Experiment Tracking | Weights & Biases |
| GUI | Tkinter |
| Testing | pytest |

---

## Project Structure

```
HandFlow/
├── config/                 # YAML configuration files
│   ├── config.yaml        # Model & training settings
│   └── handflow_setting.yaml  # User preferences & gesture mappings
├── src/handflow/          # Main package
│   ├── app/               # GUI application
│   ├── data/              # Data loading & augmentation
│   ├── features/          # Feature engineering
│   ├── models/            # Neural network architectures
│   ├── detector/          # Gesture & ArUco detection
│   └── actions/           # Mouse/keyboard control
├── scripts/               # Training & preprocessing scripts
├── models/                # Trained model artifacts
├── Experiment_notebooks/  # Jupyter notebooks for exploration
└── tests/                 # Unit tests
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Webcam
- (Optional) Printed ArUco markers for spatial calibration

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/HandFlow.git
cd HandFlow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Quick Start

```bash
# Run the application
python -m handflow

# Or run with GUI
python -m handflow --gui
```

### Training Custom Models

```bash
# Train with default configuration
python scripts/train.py

# Train with custom config
python scripts/train.py --config config/config.yaml
```

### Benchmarking Model Inference

Compare inference times across different architectures:

```bash
# Basic benchmark (all architectures)
python scripts/benchmark_inference.py

# More iterations for accurate results
python scripts/benchmark_inference.py --iterations 500 --warmup 50

# Include TFLite comparison
python scripts/benchmark_inference.py --include-tflite

# Benchmark specific architectures only
python scripts/benchmark_inference.py --architectures lstm tcn transformer

# Different batch size
python scripts/benchmark_inference.py --batch-size 8
```

Output includes mean/std/min/p95 inference times (ms) and parameter counts for each model.

---

## Supported Gestures

| Gesture | Default Action | Description |
|---------|---------------|-------------|
| `none` | — | Baseline (no gesture) |
| `horizontal_swipe` | Navigate | Left/right swipe motion |
| `swipeup` / `swipedown` | Scroll | Vertical swipe gestures |
| `thumb_left` / `thumb_right` | Back/Forward | Directional thumb gestures |
| `pointyclick` | Left Click | Index finger point + click |
| `middleclick` | Middle Click | Middle finger click |
| `touch` | Click | Tap gesture on virtual surface |
| `touch_hover` | Cursor Move | Move cursor without clicking |
| `touch_hold` | Drag | Hold for drag operations |

---

## Roadmap

This prototype demonstrates the foundational capabilities. Planned enhancements include:

- [ ] Multi-hand gesture combinations
- [ ] Expanded gesture vocabulary (20+ gestures)
- [ ] 3D spatial gestures with depth sensing
- [ ] Mobile deployment (Android/iOS)
- [ ] Plugin system for application-specific controls
- [ ] Improved low-light and varying background robustness
- [ ] Extended macro pad support (24+ button layouts)

---

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand landmark detection
- [OpenCV](https://opencv.org/) for computer vision utilities
- [Weights & Biases](https://wandb.ai/) for experiment tracking

---

## License

This project is for educational and demonstration purposes.

---

<p align="center">
  <i>Built with passion for touchless interaction</i>
</p>
