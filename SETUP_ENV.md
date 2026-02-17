# 🛠️ Environment Setup Guide - Multi-Face Recognition System

This guide provides a detailed, step-by-step procedure to set up the development environment for the Multi-Face Recognition Attendance System. It is designed to ensure compatibility with Windows and NVIDIA GPUs.

## 📋 Prerequisites

- **Python 3.10 or newer** installed and added to PATH.
- **NVIDIA GPU** (Optional but recommended) with updated drivers.
- **Internet connection** for downloading packages (~3GB).

---

## 🚀 Step-by-Step Setup

### 1. Open Terminal

Open your project folder in VS Code or File Explorer.

- **VS Code**: View > Terminal (`Ctrl + ` `)
- **File Explorer**: Right-click > "Open in Terminal" or type `cmd` in address bar.

### 2. Create Virtual Environment

A virtual environment isolates this project's dependencies from your system.

```powershell
python -m venv venv
```

_Why?_ To prevent conflicts with other Python projects on your system.

### 3. Activate Environment

You must activate the environment every time you work on the project.

```powershell
.\venv\Scripts\Activate.ps1
```

_Verification:_ You should see `(venv)` at the start of your command line.

### 4. Install Core Dependencies

We need to install packages in a specific order to avoid version conflicts.

**Step 4.1: Install FaceNet-PyTorch and compatible versions**

```powershell
pip install facenet-pytorch
```

- **What it does:** Installs FaceNet library with MTCNN and InceptionResnetV1 models.
- _Note:_ This automatically installs compatible versions of PyTorch, torchvision, numpy, and Pillow.

**Step 4.2: Verify PyTorch CUDA Support**

```powershell
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

- If GPU is **NOT available**, install PyTorch with CUDA support:

```powershell
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

**Step 4.3: Install remaining dependencies**

```powershell
pip install opencv-python pandas numpy==1.26.4 Pillow==10.2.0 requests tqdm Flask
```

- **opencv-python:** For camera access and image processing.
- **pandas:** For handling attendance CSV files.
- **numpy:** For numerical calculations (embeddings).
- **Pillow:** For image handling.
- **requests & tqdm:** Helper utilities.
- **Flask:** Web framework for the web interface.

### 5. Ensure Compatible Versions

To avoid dependency conflicts, ensure you have these specific versions:

```powershell
pip install numpy==1.26.4 Pillow==10.2.0 torch==2.2.0 torchvision==0.17.0
```

- **Why?** facenet-pytorch 2.6.0 requires specific versions to work correctly.
- This command will downgrade if you have newer incompatible versions installed.

---

## ✅ Verification

Run the following command to verify everything is working and your GPU is detected:

```powershell
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

**Expected Output (with GPU):**

```text
GPU Available: True
Device Name: NVIDIA GeForce RTX 3050 Laptop GPU
```

**Expected Output (without GPU):**

```text
GPU Available: False
Device Name: CPU only
```

---

## 🏃 Running the System

### Option 1: Web Interface (Recommended ⭐)

The modern web interface provides a user-friendly dashboard with real-time video streaming, enrollment controls, and attendance management.

**1. Start the Flask Web Server**

```powershell
python app.py
```

**2. Access the Dashboard**

- Open your browser and navigate to: **http://localhost:5000**
- Or: **http://127.0.0.1:5000**

**3. Select Device**

- On the landing page, choose **CPU** or **GPU** for processing.
- Click **Continue** to proceed to the dashboard.

**4. Use the Dashboard**

The dashboard provides two main functions:

**Enroll New User:**

- Enter name and registration number in the enrollment form.
- Click **Start Enrollment**.
- Follow on-screen pose instructions (center, left, right, up, down).
- System will capture 20 samples automatically.

**Mark Attendance:**

- Click **Start Attendance** button.
- Stand in front of the camera for recognition.
- Attendance is marked automatically with green bounding boxes.
- View marked attendees in the real-time list.
- Click **Stop Attendance** when done.
- Download CSV file using **Download Attendance CSV** button.

**5. Stop the Server**

- Press `Ctrl + C` in the terminal to stop the Flask server.

---

### Option 2: Command Line Interface (CLI - Legacy)

For advanced users or automated scripts, the CLI version is still available.

**1. Enroll a New User**

```powershell
python face_enroll.py
```

- **Device Selection:** The script will prompt you to select CPU (1) or GPU (2).
- Select Option **2** (GPU) if you have a CUDA-capable GPU, otherwise select **1** (CPU).
- Follow on-screen instructions to capture face samples.

**2. Mark Attendance**

```powershell
python mark_attendance.py
```

- **Device Selection:** Choose CPU (1) or GPU (2) when prompted.
- Stand in front of the camera for face recognition.
- Press **Q** to quit the system.

---

## 📦 Installed Packages Summary

| Package         | Version | Purpose                                       |
| --------------- | ------- | --------------------------------------------- |
| torch           | 2.2.0   | Deep learning framework with CUDA support     |
| torchvision     | 0.17.0  | Computer vision utilities for PyTorch         |
| facenet-pytorch | 2.6.0   | Face detection (MTCNN) and recognition models |
| numpy           | 1.26.4  | Numerical computing and array operations      |
| Pillow          | 10.2.0  | Image processing library                      |
| opencv-python   | Latest  | Camera access and image manipulation          |
| pandas          | Latest  | CSV file handling for attendance records      |
| Flask           | Latest  | Web framework for dashboard interface         |
| requests        | Latest  | HTTP library for downloading models           |
| tqdm            | Latest  | Progress bar utilities                        |

---

## 🐛 Troubleshooting

### Issue 1: Dependency Conflicts

**Error:** `pip's dependency resolver does not currently take into account all the packages...`

**Solution:** Run the version compatibility command:

```powershell
pip install numpy==1.26.4 Pillow==10.2.0 torch==2.2.0 torchvision==0.17.0
```

### Issue 2: GPU Not Detected

**Symptoms:** `GPU Available: False` even though you have an NVIDIA GPU.

**Solutions:**

1. Update NVIDIA GPU drivers from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Verify CUDA installation: `nvidia-smi` in terminal
3. Reinstall PyTorch with CUDA:
   ```powershell
   pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
   ```

### Issue 3: Webcam Not Working

**Error:** "Error: Unable to access the webcam."

**Solutions:**

1. Close any other applications using the webcam (Zoom, Teams, etc.)
2. Check if camera permissions are enabled in Windows Settings
3. Try changing camera index in code from `0` to `1` or `2`

### Issue 4: FutureWarning Messages

**Symptoms:** Long warning messages from `facenet_pytorch` about `torch.load`.

**Explanation:** These are warnings from the library, not errors. Your application works fine despite them. They appear because the library hasn't updated to newer PyTorch security standards yet.

### Issue 5: Flask Server Won't Start

**Error:** `Address already in use` or `OSError: [WinError 10048]`

**Solution:** Port 5000 is already in use by another application.

1. Find and close the application using port 5000:
   ```powershell
   netstat -ano | findstr :5000
   ```
2. Or run Flask on a different port:
   ```powershell
   set FLASK_RUN_PORT=5001
   python app.py
   ```

### Issue 6: Camera Not Accessible in Web Browser

**Error:** Browser shows "Cannot access camera" or black video feed.

**Solutions:**

1. Make sure no other application is using the camera (close CLI scripts).
2. For HTTPS requirement: Modern browsers require HTTPS for camera access on non-localhost. Use `localhost` or `127.0.0.1` instead of your IP address for local testing.
3. Grant camera permissions in browser when prompted.

### Issue 7: Web Dashboard Shows 404 Not Found

**Error:** Browser cannot load the dashboard page.

**Solutions:**

1. Ensure Flask server is running (`python app.py`).
2. Check the terminal for Flask startup messages.
3. Verify you're using the correct URL: `http://localhost:5000` or `http://127.0.0.1:5000`
4. Clear browser cache and refresh the page.

---

## 📁 Project Structure

```
Multi_Face_Recognition/
├── app.py                  # Flask web server (main entry point)
├── video_stream.py         # Video streaming and processing engine
├── face_enroll.py          # CLI script to enroll new users
├── mark_attendance.py      # CLI script to mark attendance
├── README.md               # Project overview and documentation
├── SETUP_ENV.md           # This file - setup instructions
├── PROJECT_REPORT.txt     # Detailed technical report
├── templates/              # HTML templates for web interface
│   ├── landing.html        # Device selection page
│   └── index.html          # Main dashboard
├── static/                 # Static assets for web interface
│   ├── css/
│   │   └── style.css       # Custom styling
│   └── js/
│       └── script.js       # Frontend JavaScript
├── data/
│   ├── attendance.csv      # Attendance records
│   └── enrolled_people/    # Stored face embeddings
│       ├── 00102_Rekha_Kumari/
│       ├── 00123_Tapan_kumar/
│       ├── 22105131004_Ankit_Kumar/
│       └── 22151131006_Uday_Kumar/
└── venv/                   # Virtual environment (created by you)
```
