import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


ENROLLED_ROOT = "data/enrolled_people"
ATTENDANCE_CSV = "data/attendance.csv"
DETECTION_PROB_THRESHOLD = 0.9
MIN_FACE_BOX_SIZE = 80
SIMILARITY_THRESHOLD = 0.6  # Cosine similarity threshold for recognition
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_enrolled_people(embedder: InceptionResnetV1, device: torch.device) -> Dict[str, Dict]:
    """Load all enrolled people and their face embeddings."""
    enrolled = {}
    
    if not os.path.exists(ENROLLED_ROOT):
        print(f"Warning: Enrolled people directory not found: {ENROLLED_ROOT}")
        return enrolled
    
    for person_folder in os.listdir(ENROLLED_ROOT):
        folder_path = os.path.join(ENROLLED_ROOT, person_folder)
        if not os.path.isdir(folder_path):
            continue
        
        meta_path = os.path.join(folder_path, "meta.json")
        embedding_path = os.path.join(folder_path, "embedding_mean.npy")
        
        if os.path.exists(meta_path) and os.path.exists(embedding_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                embedding = np.load(embedding_path)
                
                enrolled[meta['id']] = {
                    'name': meta['name'],
                    'id': meta['id'],
                    'embedding': embedding,
                    'folder': person_folder
                }
                print(f"Loaded: {meta['name']} ({meta['id']})")
            except Exception as e:
                print(f"Error loading {person_folder}: {e}")
    
    return enrolled


def initialize_attendance_csv():
    """Initialize attendance CSV if it doesn't exist."""
    os.makedirs(os.path.dirname(ATTENDANCE_CSV), exist_ok=True)
    
    if not os.path.exists(ATTENDANCE_CSV):
        df = pd.DataFrame(columns=['Registration_Number', 'Name', 'Date', 'Time', 'Status'])
        df.to_csv(ATTENDANCE_CSV, index=False)


def is_already_marked_today(reg_number: str) -> bool:
    """Check if student has already been marked present today."""
    if not os.path.exists(ATTENDANCE_CSV):
        return False
    
    df = pd.read_csv(ATTENDANCE_CSV, dtype={'Registration_Number': str})
    today = datetime.now().strftime('%Y-%m-%d')
    
    existing = df[(df['Registration_Number'] == reg_number) & (df['Date'] == today)]
    return len(existing) > 0


def mark_attendance(reg_number: str, name: str) -> bool:
    """Mark attendance for a student."""
    if is_already_marked_today(reg_number):
        return False
    
    df = pd.read_csv(ATTENDANCE_CSV)
    
    now = datetime.now()
    new_record = pd.DataFrame({
        'Registration_Number': [reg_number],
        'Name': [name],
        'Date': [now.strftime('%Y-%m-%d')],
        'Time': [now.strftime('%H:%M:%S')],
        'Status': ['Present']
    })
    
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(ATTENDANCE_CSV, index=False)
    
    return True


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def recognize_face(
    face_tensor: torch.Tensor,
    enrolled_people: Dict[str, Dict],
    embedder: InceptionResnetV1,
    device: torch.device
) -> Tuple[str, str, float]:
    """
    Recognize a face by comparing with enrolled embeddings.
    Returns: (reg_number, name, similarity) or (None, None, 0.0)
    """
    with torch.no_grad():
        face_embedding = embedder(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]
    
    best_match_id = None
    best_match_name = None
    best_similarity = 0.0
    
    for person_id, person_data in enrolled_people.items():
        similarity = cosine_similarity(face_embedding, person_data['embedding'])
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = person_id
            best_match_name = person_data['name']
    
    if best_similarity >= SIMILARITY_THRESHOLD:
        return best_match_id, best_match_name, best_similarity
    
    return None, None, 0.0


def crop_and_tensorize_face(rgb_frame: np.ndarray, box: np.ndarray, target_size: int = 160) -> torch.Tensor:
    """Crop face from frame and convert to tensor."""
    x1, y1, x2, y2 = [int(v) for v in box]
    height, width = rgb_frame.shape[:2]
    
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, width)
    y2 = min(y2, height)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    face_region = rgb_frame[y1:y2, x1:x2]
    if face_region.size == 0:
        return None
    
    resized_face = cv2.resize(face_region, (target_size, target_size))
    face_tensor = torch.from_numpy(resized_face).permute(2, 0, 1).float() / 255.0
    
    return face_tensor


def start_attendance_system():
    """Main function to start the attendance marking system."""
    torch.set_grad_enabled(False)
    
    # Device selection
    print("Select processing device:")
    print("1. CPU")
    if torch.cuda.is_available():
        print(f"2. GPU ({torch.cuda.get_device_name(0)})")
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "2" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = f"{torch.cuda.get_device_name(0)} (cuda:0)"
    else:
        device = torch.device("cpu")
        device_name = "cpu"
    
    print("\n" + "="*60)
    print("MULTI-FACE RECOGNITION ATTENDANCE SYSTEM")
    print("FaceNet-based Recognition")
    print("="*60)
    print(f"Device: {device_name}")
    print()
    
    # Initialize models
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=device)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    
    # Load enrolled people
    enrolled_people = load_enrolled_people(embedder, device)
    
    if not enrolled_people:
        print("\nERROR: No enrolled people found!")
        print(f"Please enroll faces first using enroll_face.py")
        print("="*60 + "\n")
        return
    
    print(f"Loaded {len(enrolled_people)} enrolled people")
    print("="*60 + "\n")
    
    # Initialize attendance CSV
    initialize_attendance_csv()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    print("Press 'Q' to quit\n")
    
    # Track marked students in this session
    marked_in_session = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam!")
            break
        
        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
        
        if boxes is not None and probs is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob is None or prob < DETECTION_PROB_THRESHOLD:
                    continue
                
                width = box[2] - box[0]
                height = box[3] - box[1]
                
                if width < MIN_FACE_BOX_SIZE or height < MIN_FACE_BOX_SIZE:
                    continue
                
                # Crop and prepare face for recognition
                face_tensor = crop_and_tensorize_face(rgb_frame, box)
                
                if face_tensor is None:
                    continue
                
                # Recognize face
                reg_number, name, similarity = recognize_face(face_tensor, enrolled_people, embedder, device)
                
                x1, y1, x2, y2 = [int(v) for v in box]
                
                if reg_number is not None:
                    # Recognized face - GREEN BOX
                    color = (0, 255, 0)
                    label = f"{name} ({reg_number})"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1 - 30),
                               FONT, 0.6, color, 2)
                    cv2.putText(display_frame, f"Sim: {similarity:.2f}", (x1, y1 - 10),
                               FONT, 0.5, color, 1)
                    
                    # Check if already marked
                    already_marked = is_already_marked_today(reg_number)
                    
                    if already_marked:
                        status = "Already Marked Today"
                        status_color = (0, 165, 255)  # Orange
                    elif reg_number in marked_in_session:
                        status = "Marked in Session"
                        status_color = (0, 255, 255)  # Yellow
                    else:
                        # Mark attendance
                        if mark_attendance(reg_number, name):
                            marked_in_session.add(reg_number)
                            status = "Attendance Marked!"
                            status_color = (0, 255, 0)  # Green
                            print(f"✓ Attendance marked: {name} ({reg_number}) at {datetime.now().strftime('%H:%M:%S')}")
                        else:
                            status = "Already Marked"
                            status_color = (0, 165, 255)  # Orange
                    
                    cv2.putText(display_frame, status, (x1, y2 + 20),
                               FONT, 0.5, status_color, 2)
                else:
                    # Unknown face - RED BOX
                    color = (0, 0, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, "Unknown Face", (x1, y1 - 10),
                               FONT, 0.6, color, 2)
        
        # Display info panel
        info_y = 30
        cv2.putText(display_frame, f"Enrolled: {len(enrolled_people)} | Marked Today: {len(marked_in_session)}",
                   (10, info_y), FONT, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'Q' to quit", (10, info_y + 30),
                   FONT, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Attendance System - FaceNet Recognition', display_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("ATTENDANCE SESSION ENDED")
    print(f"Total marked in this session: {len(marked_in_session)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    start_attendance_system()
