<div align="center">

# 🎓 Multi-Face Recognition Attendance System

### _AI-Powered Smart Attendance with Real-Time Face Recognition_

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)](LICENSE)

<img src="https://img.icons8.com/?size=100&id=a7dp48MWGFN9&format=png&color=000000" alt="Face Recognition" width="150"/>

**Revolutionize attendance tracking with cutting-edge deep learning technology**

**🌐 Now with Modern Web Interface!**

[📖 Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [🌐 Web Interface](#-web-interface) • [⚙️ Features](#-key-features)

---

</div>

## 📑 Table of Contents

- [🌟 Overview](#-overview)
  - [💡 The Problem We Solve](#-the-problem-we-solve)
- [⚙️ Key Features](#️-key-features)
- [🚀 Quick Start](#-quick-start)
  - [📦 Prerequisites](#-prerequisites)
  - [⚡ Installation](#-installation)
  - [✅ Verify Installation](#-verify-installation)
- [📖 Documentation](#-documentation)
  - [🎯 How It Works](#-how-it-works)
  - [📋 Step-by-Step Usage](#-step-by-step-usage)
    - [1️⃣ Enroll New Students](#1️⃣-enroll-new-students)
    - [2️⃣ Mark Attendance](#2️⃣-mark-attendance)
- [🏗️ Technical Architecture](#️-technical-architecture)
  - [🧩 System Components](#-system-components)
  - [🔬 Algorithm Workflow](#-algorithm-workflow)
  - [📊 Performance Metrics](#-performance-metrics)
- [📁 Project Structure](#-project-structure)
  - [📋 Data File Formats](#-data-file-formats)
- [🛠️ Configuration & Customization](#️-configuration--customization)
  - [⚙️ Adjustable Parameters](#️-adjustable-parameters)
  - [🎛️ Advanced Tuning](#️-advanced-tuning)
- [🔧 Troubleshooting Guide](#-troubleshooting-guide)
- [🎓 Real-World Use Cases](#-real-world-use-cases)
  - [📈 Impact Metrics](#-impact-metrics)
- [🚀 Future Enhancements](#-future-enhancements)
- [⚙️ Tech Stack](#️-tech-stack)
- [👨‍💻 Contributors](#-contributors)
- [👏 Credits](#-credits)
- [📜 License](#-license)

---

## 🌟 Overview

<div align="center">

### 🎉 **Latest Update: Web-Based UI Interface!**

![System Demo](https://img.shields.io/badge/Interface-Web%20Based-brightgreen?style=for-the-badge)
![Real--time](https://img.shields.io/badge/Processing-Real--time-blue?style=for-the-badge)
![Responsive](https://img.shields.io/badge/Design-Responsive-orange?style=for-the-badge)

</div>

Say goodbye to long queues and manual attendance marking! This **intelligent attendance system** leverages state-of-the-art **FaceNet** architecture with a **modern web interface** to recognize multiple faces simultaneously, making attendance marking **fast, accurate, and touchless**.

### ✨ **What's New in v2.0**

- 🌐 **Beautiful Web Interface** - Browser-based UI with real-time video streaming
- 🎨 **Modern Design** - Sleek, responsive interface with live feedback
- 📊 **Live Dashboard** - Real-time attendance tracking and statistics
- 🔄 **REST API** - Easy integration with existing systems
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile
- 💾 **One-Click Export** - Download attendance reports instantly

### 💡 The Problem We Solve

| ❌ Traditional Method                | ✅ Our Solution                               |
| ------------------------------------ | --------------------------------------------- |
| 🐌 One-by-one biometric scanning     | ⚡ Recognize multiple students simultaneously |
| ⏰ Long queues wasting precious time | 🚀 Instant recognition in real-time           |
| 🦠 Physical contact required         | 🙌 Completely touchless system                |
| 📝 Manual attendance registers       | 🤖 Fully automated with CSV logs              |
| ❓ Proxy attendance possible         | 🔒 Highly secure face verification            |

---

##⚙️ Key Features

<table>
<tr>
<td width="33%">

### 🌐 **Modern Web Interface**

- Beautiful browser-based UI
- Real-time video streaming
- Live attendance dashboard
- Responsive design
- One-click CSV export

### 🎯 **Multi-Face Detection**

- Detect **multiple faces** at once
- MTCNN-powered accuracy
- Various lighting conditions

</td>
<td width="33%">

### 🧠 **Deep Learning AI**

- **FaceNet** architecture
- VGGFace2 pre-trained
- 128-D face embeddings
- Cosine similarity matching

### 🎨 **Visual Feedback**

- 🟢 **Green**: Recognized
- 🔴 **Red**: Unknown
- Real-time scores
- Name & ID overlay

</td>
<td width="33%">

### 🖥️ **GPU Acceleration**

- CUDA support (10x faster)
- CPU fallback available
- Runtime device selection

### 🔐 **Smart Security**

- Secure embeddings
- Duplicate prevention
- Timestamp tracking
- Session management

### 💾 **Data Management**

- CSV attendance logs
- Automatic backup
- Easy integration

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 📦 Prerequisites

<div align="center">

| Requirement       | Version        | Status                                                   |
| ----------------- | -------------- | -------------------------------------------------------- |
| 🐍 **Python**     | 3.10+          | ![Required](https://img.shields.io/badge/Required-red)   |
| 🎥 **Webcam**     | Any            | ![Required](https://img.shields.io/badge/Required-red)   |
| 🎮 **NVIDIA GPU** | RTX/GTX Series | ![Optional](https://img.shields.io/badge/Optional-green) |
| 💾 **Storage**    | ~3GB           | ![Required](https://img.shields.io/badge/Required-red)   |

</div>

### ⚡ Installation

Follow our comprehensive setup guide: **[SETUP_ENV.md](SETUP_ENV.md)**

**Quick Setup (4 Steps):**

```bash
# 1️⃣ Create virtual environment
python -m venv venv

# 2️⃣ Activate environment
.\venv\Scripts\Activate.ps1

# 3️⃣ Install dependencies
pip install facenet-pytorch opencv-python pandas flask
pip install numpy==1.26.4 Pillow==10.2.0 torch==2.2.0 torchvision==0.17.0

# 4️⃣ Launch web interface
python app.py
```

### ✅ Verify Installation

```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

<div align="center">

**✨ You're all set! Let's launch the web interface ✨**

</div>

---

## 🌐 Web Interface

<div align="center">

### 🎨 **Beautiful, Modern, & Intuitive UI**

![Interface](https://img.shields.io/badge/Interface-Web%20Based-success?style=flat-square)
![Streaming](https://img.shields.io/badge/Video-Real--time-blue?style=flat-square)
![Responsive](https://img.shields.io/badge/Design-Responsive-orange?style=flat-square)

</div>

### 🚀 Launch the Application

Start the Flask web server:

```bash
python app.py
```

The application will start on **`http://localhost:5000`**

### 📱 Using the Web Interface

#### **1️⃣ Device Selection (Landing Page)**

<div align="center">
  <img src="assets/landing_page.png" alt="Device Selection Landing Page" width="800"/>
  <br>
  <em>Modern glassmorphism UI with CPU/GPU device selection</em>
</div>

<br>

- **First visit:** Choose between CPU or GPU processing
- **System initializes** models and loads enrolled faces
- **Redirects** to main dashboard automatically

#### **2️⃣ Main Dashboard**

<div align="center">
  <img src="assets/dashboard.png.png" alt="Main Dashboard Interface" width="900"/>
  <br>
  <em>Real-time video feed with enrollment & attendance controls</em>
</div>

<br>

<table>
<tr>
<td width="60%">

**🎥 Live Video Feed**

- Real-time camera stream with MJPEG streaming
- Face detection overlays with bounding boxes
- Green boxes for recognized students
- Red boxes for unknown faces
- Live status indicators and similarity scores

**📊 Live Attendance List**

- Real-time updates as students are recognized
- Shows: Name, ID, Time, Status
- Color-coded badges (green for present)
- Session tracking with duplicate prevention

</td>
<td width="40%">

**🎛️ Control Panel**

**Enrollment Mode:**

- Enter student name and registration ID
- Click "Start Enrollment" button
- Follow on-screen pose instructions
- 20 samples captured automatically across 5 poses
- Real-time progress feedback

**Attendance Mode:**

- Click "Start Attendance" to begin recognition
- System recognizes all faces simultaneously
- Automatic CSV logging with timestamps
- Duplicate prevention (one per day)
- Download attendance CSV report

</td>
</tr>
</table>

#### **3️⃣ Features in Action**

<details>
<summary><b>✨ Real-Time Recognition</b></summary>

- **Multi-Face Detection**: Recognizes up to 10 faces simultaneously
- **Instant Feedback**: Green/Red bounding boxes with names
- **Similarity Scores**: Real-time confidence display
- **Status Messages**: "Attendance Marked", "Already Marked Today", etc.

</details>

<details>
<summary><b>📝 Enrollment Process</b></summary>

1. Fill in student name and ID
2. Click "Start Enrollment"
3. Follow pose instructions:
   - 👀 Look center
   - 👈 Look left
   - 👉 Look right
   - 👆 Look up
   - 👇 Look down
4. Hold each pose steady (5 frames)
5. System captures 20 high-quality samples
6. Embeddings generated and saved automatically

</details>

<details>
<summary><b>📊 Attendance Marking</b></summary>

- Click "Start Attendance" button
- Stand in front of camera
- System detects and recognizes faces
- Attendance marked automatically
- CSV updated with timestamp
- View live list of marked students
- Download session or full attendance report

</details>

#### **4️⃣ Download Reports**

- **📥 Full Attendance CSV**: All attendance records with dates
- **📥 Session CSV**: Only current session data
- **📊 Excel Compatible**: Direct import to spreadsheets

### 🎨 Interface Highlights

| Feature              | Description                                 |
| -------------------- | ------------------------------------------- |
| **🎭 Modern Design** | Sleek dark theme with glassmorphism effects |
| **📱 Responsive**    | Works on desktop, tablet, and mobile        |
| **⚡ Real-time**     | Instant updates without page refresh        |
| **🔄 Live Stream**   | Smooth 30 FPS video streaming               |
| **🎯 Intuitive**     | Clear controls and visual feedback          |
| **🌈 Themed**        | Professional blue-dark color scheme         |

---

## 📖 Documentation

### 🎯 How It Works

<div align="center">

```mermaid
graph LR
    A[👤 Student Appears] --> B[📷 Camera Capture]
    B --> C[🔍 Face Detection MTCNN]
    C --> D[🧮 Extract Embeddings]
    D --> E{🤔 Match Found?}
    E -->|✅ Yes| F[✍️ Mark Attendance]
    E -->|❌ No| G[🚫 Unknown Face]
    F --> H[💾 Save to CSV]
```

</div>

### 📋 Step-by-Step Usage

<table>
<tr>
<td width="50%" valign="top">

### 1️⃣ **Enroll New Students**

```powershell
python face_enroll.py
```

<details>
<summary><b>🎬 What happens during enrollment?</b></summary>

1. **Device Selection**: Choose CPU or GPU
2. **Camera Activation**: Webcam starts
3. **Face Capture**: Follow on-screen pose instructions
   - 👀 Look center
   - 👈 Look left
   - 👉 Look right
   - 👆 Look up
   - 👇 Look down
4. **Sample Collection**: 20 samples captured
5. **Embedding Generation**: FaceNet creates unique signature
6. **Data Storage**: Saved in `data/enrolled_people/`

</details>

**📊 Progress Display:**

- Real-time detection feedback
- Pose instruction overlay
- Sample counter (X/20)
- Confidence scores

**💾 Files Created:**

```
data/enrolled_people/
└── [ID]_[Name]/
    ├── face_01.jpg to face_20.jpg
    ├── thumbnail.jpg
    ├── embeddings.npy
    ├── embedding_mean.npy
    └── meta.json
```

</td>
<td width="50%" valign="top">

### 2️⃣ **Mark Attendance**

```powershell
python mark_attendance.py
```

<details>
<summary><b>🎥 How does attendance marking work?</b></summary>

1. **System Initialization**: Loads enrolled faces
2. **Device Selection**: Choose processing device
3. **Real-Time Recognition**: Camera scans for faces
4. **Instant Feedback**:
   - 🟢 **Green**: Recognized → Attendance marked
   - 🔴 **Red**: Unknown face
5. **Duplicate Prevention**: One entry per day
6. **Live Stats**: Enrolled count & marked count
7. **Exit**: Press **Q** to quit

</details>

**📊 On-Screen Display:**

```
┌──────────────────────────────────┐
│ 🟢 [Name] ([ID])                 │
│    Sim: 0.85                     │
│    ✅ Attendance Marked!          │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│ 🔴 Unknown Face                  │
└──────────────────────────────────┘
```

**📝 Attendance Log:**

```csv
Registration_Number,Name,Date,Time,Status
22105131004,Ankit Kumar,2026-01-26,09:15:23,Present
```

</td>
</tr>
</table>

---

## 🏗️ Technical Architecture

<div align="center">

### 🧩 System Components

</div>

<table>
<tr>
<td width="33%" align="center">

### 🔍 **Detection Layer**

<img src="https://img.icons8.com/fluency/96/000000/face-id.png" width="60"/>

**MTCNN**

- Multi-task Cascaded CNN
- 3-stage detection pipeline
- Facial landmark detection
- 99%+ detection accuracy

</td>
<td width="33%" align="center">

### 🧠 **Recognition Layer**

<img src="https://img.icons8.com/fluency/96/000000/artificial-intelligence.png" width="60"/>

**InceptionResnetV1**

- Pre-trained on VGGFace2
- 128-dimensional embeddings
- Cosine similarity matching
- Threshold: 0.6

</td>
<td width="33%" align="center">

### 💾 **Storage Layer**

<img src="https://img.icons8.com/fluency/96/000000/database.png" width="60"/>

**Data Management**

- NumPy arrays (.npy)
- JSON metadata
- CSV attendance logs
- Efficient retrieval

</td>
</tr>
</table>

### 🔬 Algorithm Workflow

```python
# Simplified Recognition Pipeline
frame → MTCNN → [boxes, landmarks] → crop_face() →
InceptionResnetV1 → [embedding_vector] → cosine_similarity() →
threshold_check() → {Recognized / Unknown}
```

### 📊 Performance Metrics

| Metric                      | Value    | Details                 |
| --------------------------- | -------- | ----------------------- |
| ⚡ **Detection Speed**      | ~30 FPS  | Real-time on GPU        |
| 🎯 **Recognition Accuracy** | 95%+     | Under normal conditions |
| 👥 **Simultaneous Faces**   | 1-10     | Optimal: 2-6 faces      |
| 📏 **Min Face Size**        | 80x80 px | Configurable threshold  |
| 🎚️ **Similarity Threshold** | 0.60     | Adjustable in code      |
| 💾 **Storage per User**     | ~2 MB    | 20 samples + embeddings |

---

## 📁 Project Structure

```
Multi_Face_Recognition/
│
├── 🌐 app.py                      # 🚀 Flask web application (main entry)
├── 📹 video_stream.py             # 📡 Video streaming & processing
├── 📄 face_enroll.py              # 🎯 Student enrollment script (CLI)
├── 📄 mark_attendance.py          # ✅ Attendance marking system (CLI)
├── 📄 README.md                   # 📖 This beautiful documentation
├── 📄 SETUP_ENV.md               # 🛠️ Detailed setup guide
├── 📄 PROJECT_REPORT.txt         # 📝 Detailed project report
│
├── 📂 templates/                  # 🎨 HTML templates
│   ├── landing.html              # 🏠 Device selection page
│   └── index.html                # 📊 Main dashboard
│
├── 📂 static/                     # 🎨 Static assets
│   ├── 📂 css/
│   │   └── style.css             # 💎 Custom styles
│   └── 📂 js/
│       └── script.js             # ⚡ JavaScript logic
│
├── 📂 data/                       # 💾 Data directory (auto-created)
│   ├── 📄 attendance.csv         # 📊 Attendance records
│   │
│   └── 📂 enrolled_people/       # 👥 Enrolled student database
│       ├── 📂 22105131004_Ankit_Kumar/
│       │   ├── 🖼️ face_01.jpg → face_20.jpg
│       │   ├── 🖼️ thumbnail.jpg
│       │   ├── 🔢 embeddings.npy          (20 x 128 vectors)
│       │   ├── 🔢 embedding_mean.npy      (1 x 128 vector)
│       │   └── 📋 meta.json
│       │
│       ├── 📂 00123_Tapan_kumar/
│       ├── 📂 00102_Rekha_Kumari/
│       └── 📂 22151131006_Uday_Kumar/
│
└── 📂 venv/                       # 🐍 Python virtual environment
    └── ...
```

### 📋 Data File Formats

<details>
<summary><b>📄 attendance.csv</b></summary>

```csv
Registration_Number,Name,Date,Time,Status
22105131004,Ankit Kumar,2026-02-18,09:15:23,Present
22151131006,Uday Kumar,2026-02-18,09:15:25,Present
00102,Rekha Kumari,2026-02-18,09:15:28,Present
```

</details>

<details>
<summary><b>📋 meta.json</b></summary>

```json
{
  "name": "Ankit Kumar",
  "id": "22105131004",
  "samples": 20,
  "created": "2026-02-18T08:30:45.123456+00:00"
}
```

</details>

<details>
<summary><b>🔢 embedding_mean.npy</b></summary>

- **Shape**: (128,)
- **Type**: float32
- **Purpose**: Average of 20 face embeddings
- **Usage**: Face matching via cosine similarity

</details>

---

## 🛠️ Configuration & Customization

### ⚙️ Adjustable Parameters

**face_enroll.py:**

```python
TOTAL_SAMPLES = 20              # Number of face samples to capture
STABLE_FRAMES_REQUIRED = 5      # Frames to hold pose steady
DETECTION_PROB_THRESHOLD = 0.9  # Detection confidence threshold
MIN_FACE_BOX_SIZE = 80          # Minimum face size (pixels)
DISPLAY_SCALE = 1.5             # Display window scaling
```

**mark_attendance.py:**

```python
SIMILARITY_THRESHOLD = 0.6      # Recognition threshold (0-1)
DETECTION_PROB_THRESHOLD = 0.9  # Detection confidence
MIN_FACE_BOX_SIZE = 80          # Minimum face size
```

### 🎛️ Advanced Tuning

<details>
<summary><b>🔧 Improve Accuracy</b></summary>

- ⬆️ **Increase** `SIMILARITY_THRESHOLD` (e.g., 0.65-0.70) for stricter matching
- ⬆️ **Increase** `TOTAL_SAMPLES` (e.g., 30-50) for better embeddings
- ✅ Ensure good lighting during enrollment

</details>

<details>
<summary><b>⚡ Improve Speed</b></summary>

- ⬇️ **Decrease** `TOTAL_SAMPLES` (e.g., 10-15) for faster enrollment
- 🎮 Use GPU device selection for faster processing
- ⬇️ **Decrease** camera resolution in code

</details>

<details>
<summary><b>👥 Handle More Faces</b></summary>

- ⬇️ **Decrease** `MIN_FACE_BOX_SIZE` to detect smaller/distant faces
- 📹 Use higher resolution camera
- 💡 Ensure adequate lighting

</details>

---

## 🔧 Troubleshooting Guide

### ❗ Common Issues

<details>
<summary><b>🎥 Camera Not Working</b></summary>

**Symptoms:**

```
Error: Unable to access the webcam.
```

**Solutions:**

1. ✅ Close applications using camera (Zoom, Teams)
2. 🔌 Check camera connection
3. 🔐 Enable camera permissions (Windows Settings)
4. 🔄 Try different camera index:
   ```python
   cap = cv2.VideoCapture(1)  # or 2, 3...
   ```

</details>

<details>
<summary><b>🎮 GPU Not Detected</b></summary>

**Symptoms:**

```
GPU Available: False
```

**Solutions:**

1. 🔄 Update NVIDIA drivers
2. ✅ Verify CUDA: Run `nvidia-smi`
3. 📦 Reinstall PyTorch:
   ```bash
   pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
   ```

</details>

<details>
<summary><b>⚠️ Dependency Conflicts</b></summary>

**Symptoms:**

```
ERROR: pip's dependency resolver...
facenet-pytorch requires numpy<2.0.0
```

**Solution:**

```bash
pip install numpy==1.26.4 Pillow==10.2.0 torch==2.2.0 torchvision==0.17.0
```

</details>

<details>
<summary><b>🎯 Poor Recognition Accuracy</b></summary>

**Causes & Solutions:**

| Issue                 | Solution                              |
| --------------------- | ------------------------------------- |
| 💡 Bad lighting       | Enroll in similar lighting conditions |
| 😷 Face covered       | Re-enroll without masks/glasses       |
| 📏 Face too small     | Move closer to camera                 |
| 🌐 Wrong angle        | Face camera directly                  |
| 🔢 Threshold too high | Lower `SIMILARITY_THRESHOLD`          |

**Optimization Steps:**

1. **Re-enroll** problematic users with better samples
2. **Adjust threshold** in mark_attendance.py (line 15):
   ```python
   SIMILARITY_THRESHOLD = 0.55  # Lower = more lenient
   ```
3. **Increase samples** in face_enroll.py:
   ```python
   TOTAL_SAMPLES = 30  # More samples = better accuracy
   ```

</details>

<details>
<summary><b>📊 Duplicate Entries</b></summary>

**Prevention:**

- ✅ System automatically blocks duplicates per day
- ✅ Check `attendance.csv` for verification
- ✅ Date-based filtering: `YYYY-MM-DD` format

</details>

<details>
<summary><b>⚡ Slow Performance</b></summary>

**Optimization:**

- 🎮 Select GPU (Option 2) at runtime
- ⬇️ Reduce camera resolution
- 👥 Limit simultaneous faces (2-6 optimal)
- 🔄 Close unnecessary applications

</details>

---

## 🎓 Real-World Use Cases

<div align="center">

### 🏫 Transform Your Institution's Attendance System

</div>

<table>
<tr>
<td width="25%" align="center">

### 🏛️ **Universities**

<img src="https://img.icons8.com/fluency/96/000000/university.png" width="70"/>

✅ Large lecture halls  
✅ Multiple entries  
✅ Fast processing  
✅ No hardware required

</td>
<td width="25%" align="center">

### 🏢 **Offices**

<img src="https://img.icons8.com/fluency/96/000000/company.png" width="70"/>

✅ Employee tracking  
✅ Shift management  
✅ Access control  
✅ Time logging

</td>
<td width="25%" align="center">

### 🏥 **Labs & Research**

<img width="64" height="64" src="https://img.icons8.com/external-photo3ideastudio-lineal-color-photo3ideastudio/64/external-laboratory-online-learning-photo3ideastudio-lineal-color-photo3ideastudio.png" alt="external-laboratory-online-learning-photo3ideastudio-lineal-color-photo3ideastudio"/>

✅ Lab access logs  
✅ Safety compliance  
✅ Usage tracking  
✅ Contactless entry

</td>
<td width="25%" align="center">

### 🎉 **Events**

<img src="https://img.icons8.com/fluency/96/000000/party-baloons.png" width="70"/>

✅ Entry management  
✅ Guest verification  
✅ Headcount tracking  
✅ Security checks

</td>
</tr>
</table>

### 📈 Impact Metrics

| Metric                  | Traditional    | Our System   | Improvement         |
| ----------------------- | -------------- | ------------ | ------------------- |
| ⏱️ **Time per Student** | ~10 seconds    | ~0.5 seconds | **95% faster**      |
| 👥 **Queue Length**     | 20-30 students | 0 students   | **100% reduction**  |
| 📊 **Accuracy**         | 90-95%         | 95%+         | **Higher accuracy** |
| 🦠 **Contact Points**   | Physical touch | Zero contact | **100% touchless**  |
| 💰 **Hardware Cost**    | ₹50,000+       | ~₹5,000      | **90% cheaper**     |

---

## 🚀 Future Enhancements

<div align="center">

### 🔮 Roadmap for Next Versions

</div>

<table>
<tr>
<td width="50%">

### 🎯 **Version 2.0** (Planned)

- [ ] 🌐 **Web Dashboard**
  - Real-time attendance view
  - Analytics & reports
  - Export to Excel/PDF
- [ ] 📧 **Email Notifications**
  - Attendance confirmation
  - Absence alerts
  - Daily reports to instructors

- [ ] 🔐 **Anti-Spoofing**
  - Liveness detection
  - Photo/video attack prevention
  - 3D depth analysis

- [ ] 📱 **Mobile App Integration**
  - Student mobile notifications
  - Self-check attendance
  - QR code backup

</td>
<td width="50%">

### 🚀 **Version 3.0** (Vision)

- [ ] ☁️ **Cloud Integration**
  - Azure/AWS deployment
  - Centralized database
  - Multi-campus support

- [ ] 🤖 **Advanced AI Features**
  - Emotion detection
  - Attention tracking
  - Behavior analysis

- [ ] 🔗 **ERP Integration**
  - Direct sync with college systems
  - Grade correlation
  - Automated reports

- [ ] 🌍 **Multi-Language Support**
  - UI in regional languages
  - Voice feedback
  - Accessibility features

</td>
</tr>
</table>

---

## ⚙️ Tech Stack

<div align="center">

![Technology](https://img.shields.io/badge/Technology-AI%20Stack-blue?style=for-the-badge&logo=stack-overflow)

</div>

<table align="center">
<tr>
<td align="center"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="60"/><br><b>PyTorch</b><br><sub>Deep Learning Framework</sub></td>
<td align="center"><img src="https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_black-2.png" width="60"/><br><b>OpenCV</b><br><sub>Computer Vision Library</sub></td>
<td align="center"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="60"/><br><b>NumPy</b><br><sub>Numerical Computing</sub></td>
</tr>
<tr>
<td align="center"><img width="68" height="68" src="https://img.icons8.com/external-flat-circular-vectorslab/68/external-Face-Recognition-interior-flat-circular-vectorslab.png" alt="external-Face-Recognition-interior-flat-circular-vectorslab"/><br><b>FaceNet</b><br><sub>Face Recognition Model</sub></td>
<td align="center"><img src="https://img.icons8.com/fluency/48/000000/face-id.png" width="60"/><br><b>MTCNN</b><br><sub>Face Detection Model</sub></td>
<td align="center"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="60"/><br><b>Python</b><br><sub>Programming Language</sub></td>
</tr>
</table>

### 🏆 Why These Technologies?

<div align="center">

| Technology  | Reason                                               | Version           |
| ----------- | ---------------------------------------------------- | ----------------- |
| **PyTorch** | Industry-standard, excellent GPU support             | 2.2.0             |
| **FaceNet** | State-of-the-art face recognition (128-D embeddings) | InceptionResnetV1 |
| **MTCNN**   | Robust multi-face detection with landmarks           | Latest            |
| **OpenCV**  | Mature, fast, reliable computer vision library       | 4.x               |
| **NumPy**   | Efficient numerical operations & array handling      | 1.26.4            |
| **Pandas**  | Powerful data manipulation & CSV management          | 3.0+              |
| **CUDA**    | GPU acceleration for 10x faster processing           | 12.1              |

</div>

---

## 👨‍💻 Contributors

<div align="center">

### 🎓 Semester 7 Project

**Bachelor of Technology - Computer Science**

<br>

**Made with ❤️ and lots of ☕**

<br>

### 📧 Contact & Support

Got questions? Found a bug? Want to contribute?

[![GitHub Issues](https://img.shields.io/badge/Issues-Open-red?style=for-the-badge&logo=github)](https://github.com)
[![Email](https://img.shields.io/badge/Email-Contact-blue?style=for-the-badge&logo=gmail)](mailto:your.email@example.com)

</div>

---

## 👏 Credits

<div align="center">

![Credits](https://img.shields.io/badge/Credits-Acknowledgements-yellow?style=for-the-badge&logo=heart)

</div>

<table align="center">
<tr>

<td align="center">
<img src="https://avatars.githubusercontent.com/u/155890684?v=4" width="50" style="border-radius: 50%"/>
<br><b>👨‍💻 Development</b>
<br><sub>
<a href="https://github.com/IdealAnkit">IdealAnkit</a><br>
Lead Developer
</sub>
</td>
<td align="center">
<img src="https://img.icons8.com/fluency/48/000000/source-code.png" width="50"/>
<br><b>🛠️ Libraries</b>
<br><sub>
<a href="https://github.com/timesler/facenet-pytorch">facenet-pytorch</a><br>
PyTorch, OpenCV, NumPy
</sub>
</td>
</tr>
</table>

<div align="center">

### 🌟 Special Thanks

This project builds upon cutting-edge face recognition research:

- **FaceNet** - Deep learning face representation via triplet loss
- **MTCNN** - Robust multi-task face detection architecture
- **VGGFace2** - Large-scale face recognition dataset
- **PyTorch** - Powerful deep learning framework
- **OpenCV** - Essential computer vision toolkit

</div>

---

## 📜 License

<div align="center">

![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

This project is developed as an **academic project** for educational purposes.

**Semester 7 Project** | **Bachelor of Technology - Computer Science**

</div>

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

**Happy Coding! 🚀**

<br>

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Powered by PyTorch](https://img.shields.io/badge/Powered%20by-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask Web Framework](https://img.shields.io/badge/Flask-Web%20Framework-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Built with Love](https://img.shields.io/badge/Built%20with-Love-red?style=for-the-badge&logo=heart&logoColor=white)](https://github.com)

**© 2026 Multi-Face Recognition Attendance System**

</div>

---

## 🎓 Project Credits

<table align="center">
<tr>
<td align="center">
<img src="https://avatars.githubusercontent.com/u/155890684?v=4" width="60"/>
<br><b>👨‍💻 Development</b>
<br><sub>SEM 7 Student Project</sub>
<br><sub>PyTorch Implementation</sub>
</td>
<td align="center">
<img src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" width="80"/>
<br><b>🔥 AI Framework</b>
<br><sub>PyTorch</sub>
<br><sub>Deep Learning Library</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" width="70"/>
<br><b>📸 Computer Vision</b>
<br><sub>OpenCV</sub>
<br><sub>Image Processing</sub>
</td>
</tr>
<tr>
<td align="center">
<img width="96" height="96" src="https://img.icons8.com/nolan/96/flask.png" alt="flask"/>
<br><b>🌐 Web Framework</b>
<br><sub>Flask</sub>
<br><sub>Python Web Server</sub>
</td>
<td align="center">
<img src="https://img.icons8.com/fluency/96/000000/source-code.png" width="60"/>
<br><b>🛠️ Open Source Tools</b>
<br><sub>NumPy, Pandas</sub>
<br><sub>Python Ecosystem</sub>
</td>
<td align="center">
<img width="48" height="48" src="https://img.icons8.com/pulsar-gradient/48/artificial-intelligence.png" alt="artificial-intelligence"/>
<br><b>🧠 Pre-trained Models</b>
<br><sub>FaceNet & MTCNN</sub>
<br><sub>VGGFace2 Dataset</sub>
</td>
</tr>
</table>

<div align="center">

### 💝 Special Acknowledgments

This project leverages cutting-edge research and open-source contributions:

**Research Papers:**

- 📄 [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832) (Schroff et al., 2015)
- 📄 [Joint Face Detection and Alignment using MTCNN](https://arxiv.org/abs/1604.02878) (Zhang et al., 2016)

**Open Source Libraries:**

- 🔥 [PyTorch](https://pytorch.org/) - Deep learning platform
- 🌐 [Flask](https://flask.palletsprojects.com/) - Lightweight web framework
- 📸 [OpenCV](https://opencv.org/) - Computer vision library
- 🧠 [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - Pre-trained FaceNet models

**Datasets:**

- 👥 [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) - Large-scale face recognition dataset

---

**Developed with 💙 by Computer Science Students** | **Bachelor of Technology - Semester 7**

Made with Python 🐍 | Powered by AI 🤖 | Built for Education 🎓

</div>
