import cv2
import torch
import numpy as np
import time
import os
import json
import threading
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from face_enroll import evaluate_pose, crop_face, save_face_data, TOTAL_SAMPLES, STABLE_FRAMES_REQUIRED, CAPTURE_DELAY_SECONDS, POSE_SEQUENCE, DETECTION_PROB_THRESHOLD, MIN_FACE_BOX_SIZE, scale_boxes_landmarks
from mark_attendance import load_enrolled_people, crop_and_tensorize_face, recognize_face, mark_attendance, is_already_marked_today, initialize_attendance_csv

class VideoCamera:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VideoCamera, cls).__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        
        self.cap = None
        self.mode = "STOPPED"  # STOPPED, ENROLLMENT, ATTENDANCE
        self.system_ready = False # Flag for device initialization
        
        # Placeholders for models (loaded in initialize_system)
        self.device = None
        self.mtcnn = None
        self.embedder = None
        self.enrolled_people = {}
        
        # Enrollment State
        self.enroll_name = ""
        self.enroll_id = ""
        self.samples = []
        self.pose_index = 0
        self.current_prompt = POSE_SEQUENCE[0]
        self.steady_frame_counter = 0
        self.prompt_cycle = None 
        self.last_capture_time = 0

        # Attendance State
        self.marked_session = set()
        self.frame_count = 0
        self.last_boxes = None
        self.last_names = None
        self.last_statuses = None
        
        self.initialized = True

    def initialize_system(self, device_choice='cpu'):
        """
        Initialize models with selected device.
        device_choice: 'cpu' or 'cuda'
        """
        if self.system_ready:
            print("System already initialized.")
            return True

        try:
            if device_choice == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
                
            print(f"Initializing models on {self.device}...")
            
            # Load Models
            self.mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=self.device)
            self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            # Load Data
            self.enrolled_people = load_enrolled_people(self.embedder, self.device)
            initialize_attendance_csv()
            
            self.system_ready = True
            print("System initialization complete.")
            return True
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False

    def start_camera(self):
        if not self.system_ready:
            return # Don't start if not initialized
            
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def stop_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def start_enrollment(self, name, user_id):
        self.mode = "ENROLLMENT"
        self.enroll_name = name
        self.enroll_id = user_id
        self.samples = []
        self.pose_index = 0
        self.prompt_cycle = iter(POSE_SEQUENCE * (TOTAL_SAMPLES // len(POSE_SEQUENCE) + 1)) # Ensure enough items
        self.current_prompt = next(self.prompt_cycle)
        self.steady_frame_counter = 0
        self.last_capture_time = 0
        self.start_camera()

    def start_attendance(self):
        self.mode = "ATTENDANCE"
        if self.system_ready:
             # Reload enrolled people to catch any new enrollments
             self.enrolled_people = load_enrolled_people(self.embedder, self.device)
        self.start_camera()

    def stop_mode(self):
        self.mode = "STOPPED"
        self.stop_camera()

    def get_frame(self):
        # Debugging print to trace status
        # print(f"DEBUG: get_frame | Mode: {self.mode} | Ready: {self.system_ready}")

        if not self.system_ready or self.mode == "STOPPED":
             blank_image = np.zeros((480, 640, 3), np.uint8)
             msg = "System Not Initialized" if not self.system_ready else "Camera Stopped"
             cv2.putText(blank_image, msg, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             ret, jpeg = cv2.imencode('.jpg', blank_image)
             return jpeg.tobytes()

        # Debugging camera start
        if self.cap is None or not self.cap.isOpened():
            print(f"Starting camera in get_frame because mode is {self.mode}")
            self.start_camera() 
            
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        display_frame = frame.copy()

        if self.mode == "ENROLLMENT":
            display_frame = self._process_enrollment(display_frame, rgb_frame)
        elif self.mode == "ATTENDANCE":
            display_frame = self._process_attendance(display_frame, rgb_frame)

        ret, jpeg = cv2.imencode('.jpg', display_frame)
        return jpeg.tobytes()
        
    def get_session_csv(self):
        """Returns a CSV string of students marked in current session"""
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Registration_Number', 'Name', 'Status']) # Header
        
        for reg_id in self.marked_session:
            name = "Unknown"
            if reg_id in self.enrolled_people:
                name = self.enrolled_people[reg_id]['name']
            writer.writerow([reg_id, name, 'Present'])
            
        return output.getvalue()

    def _process_enrollment(self, display_frame, rgb_frame):
        # Stop if we have enough samples
        if len(self.samples) >= TOTAL_SAMPLES:
            cv2.putText(display_frame, "Enrollment Complete!", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Save data automatically if complete
            try:
                 save_face_data(self.samples, self.enroll_name, self.enroll_id, self.embedder, self.device)
                 self.mode = "STOPPED" # Stop after saving
            except Exception as e:
                print(f"Error saving data: {e}")
            return display_frame

        # Detect
        boxes, probs, landmarks = self.mtcnn.detect(rgb_frame, landmarks=True)

        best_box = None
        best_landmarks = None
        best_prob = 0.0

        if boxes is not None:
             # Find best face
            for i, prob in enumerate(probs):
                if prob > best_prob:
                    best_prob = prob
                    best_box = boxes[i]
                    best_landmarks = landmarks[i]

        detection_ready = False
        if best_box is not None and best_prob >= DETECTION_PROB_THRESHOLD:
            width = best_box[2] - best_box[0]
            height = best_box[3] - best_box[1]
            
            if width >= MIN_FACE_BOX_SIZE and height >= MIN_FACE_BOX_SIZE:
                 # Since we flipped the frame (mirror), "Look Left" in reality becomes "Look Left" in the image (Negative offset).
                 # But evaluate_pose expects "Look Left" to be "Look Right" in image (Positive offset) because it assumes non-mirrored webcam.
                 # So we must swap the prompt we send to the logic.
                 logic_prompt = self.current_prompt
                 if self.current_prompt == "Look left":
                     logic_prompt = "Look right"
                 elif self.current_prompt == "Look right":
                     logic_prompt = "Look left"
                     
                 detection_ready = evaluate_pose(best_landmarks, best_box, logic_prompt)
        
        if detection_ready:
            self.steady_frame_counter += 1
        else:
            self.steady_frame_counter = 0

        # Draw UI
        color = (0, 255, 0) if detection_ready else (0, 0, 255)
        
        if best_box is not None:
            x1, y1, x2, y2 = [int(x) for x in best_box]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

        # Text overlays in Yellow/Green/Red
        cv2.putText(display_frame, f"Pose: {self.current_prompt}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Progress: {len(self.samples)}/{TOTAL_SAMPLES}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        if detection_ready:
             cv2.putText(display_frame, f"Hold Steady: {self.steady_frame_counter}/{STABLE_FRAMES_REQUIRED}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
             cv2.putText(display_frame, "Align Face", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Capture logic
        current_time = time.time()
        if detection_ready and self.steady_frame_counter >= STABLE_FRAMES_REQUIRED:
            if current_time - self.last_capture_time >= CAPTURE_DELAY_SECONDS:
                 face_rgb, face_tensor = crop_face(rgb_frame, best_box)
                 if face_rgb is not None:
                     self.samples.append({"rgb": face_rgb, "tensor": face_tensor, "prompt": self.current_prompt})
                     self.last_capture_time = current_time
                     self.steady_frame_counter = 0
                     try:
                        self.current_prompt = next(self.prompt_cycle)
                     except StopIteration:
                        pass # Should handle via length check

        return display_frame

    
    def _process_attendance(self, display_frame, rgb_frame):
        self.frame_count += 1
        
        # Process detection only every 4th frame to reduce lag
        if self.frame_count % 4 == 0 or self.last_boxes is None:
            boxes, probs, _ = self.mtcnn.detect(rgb_frame, landmarks=True)
            self.last_boxes = boxes if boxes is not None else []
            self.last_names = []
            self.last_statuses = []

            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob < 0.9: 
                        self.last_names.append(None)
                        self.last_statuses.append(None)
                        continue

                    face_tensor = crop_and_tensorize_face(rgb_frame, box)
                    if face_tensor is None:
                        self.last_names.append(None)
                        self.last_statuses.append(None) 
                        continue

                    reg_number, name, similarity = recognize_face(face_tensor, self.enrolled_people, self.embedder, self.device)
                    
                    if reg_number:
                        # Mark attendance logic
                        already_marked_today = is_already_marked_today(reg_number)
                        status = ""
                        if already_marked_today:
                            status = "Marked Today"
                        elif reg_number in self.marked_session:
                            status = "Marked Session"
                        else:
                            if mark_attendance(reg_number, name):
                                self.marked_session.add(reg_number)
                                status = "Marked Now!"
                            else:
                                 status = "Already Marked"
                        
                        self.last_names.append(name)
                        self.last_statuses.append(status)
                    else:
                        self.last_names.append("Unknown")
                        self.last_statuses.append(None)

        # Draw results (either fresh or cached)
        if self.last_boxes is not None and len(self.last_boxes) > 0:
             for i, box in enumerate(self.last_boxes):
                 if i >= len(self.last_names): break # Safety check
                 
                 name = self.last_names[i]
                 status = self.last_statuses[i]
                 
                 if name is None: continue # skipped low prob face
                 
                 x1, y1, x2, y2 = [int(v) for v in box]
                 
                 if name != "Unknown":
                     color = (0, 255, 0)
                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                     cv2.putText(display_frame, name, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                     if status:
                        cv2.putText(display_frame, status, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                 else:
                     color = (0, 0, 255)
                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                     cv2.putText(display_frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return display_frame

    def get_marked_students(self):
        # Get list of students marked in this session
        students = []
        for reg_id in self.marked_session:
            # Find name from enrolled data
            name = "Unknown"
            if reg_id in self.enrolled_people:
                name = self.enrolled_people[reg_id]['name']
            
            # Since we don't have exact time for session marks in memory, 
            # we can look up from CSV or just show "Present"
            # For better UX, let's just show Name and ID
            students.append({"id": reg_id, "name": name, "status": "Present"})
        return students

# Global instance
camera = VideoCamera()
