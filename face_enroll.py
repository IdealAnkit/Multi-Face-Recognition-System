import os
import re
import json
import time
from datetime import datetime, timezone
from itertools import cycle
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='facenet_pytorch')


TOTAL_SAMPLES = 20
STABLE_FRAMES_REQUIRED = 5
CAPTURE_DELAY_SECONDS = 0.6
DETECTION_PROB_THRESHOLD = 0.9
MIN_FACE_BOX_SIZE = 80
OUTPUT_ROOT = "data/enrolled_people"
DISPLAY_SCALE = 1.5  # Scale factor for display (fits screen better)
POSE_SEQUENCE = [
    "Look center",
    "Look left",
    "Look right",
    "Look up",
    "Look down",
]


def safe_folder_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    return cleaned or "user"


def evaluate_pose(landmarks: np.ndarray, box: np.ndarray, prompt: str) -> bool:
    left_eye, right_eye, nose, mouth_left, mouth_right = landmarks
    box_width = max(box[2] - box[0], 1.0)
    box_height = max(box[3] - box[1], 1.0)

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    nose_horizontal_offset = (nose[0] - eye_mid_x) / box_width

    box_center_y = (box[1] + box[3]) / 2.0
    nose_vertical_offset = (nose[1] - box_center_y) / box_height

    if prompt == "Look center":
        return abs(nose_horizontal_offset) <= 0.05 and abs(nose_vertical_offset) <= 0.06
    if prompt == "Look left":
        return nose_horizontal_offset >= 0.07
    if prompt == "Look right":
        return nose_horizontal_offset <= -0.07
    if prompt == "Look up":
        return nose_vertical_offset <= -0.05 and abs(nose_horizontal_offset) <= 0.15
    if prompt == "Look down":
        return nose_vertical_offset >= 0.08 and abs(nose_horizontal_offset) <= 0.15
    return False


def clip_box(box: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    height, width = frame_shape
    x1 = max(int(box[0]), 0)
    y1 = max(int(box[1]), 0)
    x2 = min(int(box[2]), width)
    y2 = min(int(box[3]), height)
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    return x1, y1, x2, y2


def crop_face(rgb_frame: np.ndarray, box: np.ndarray, target_size: int = 160) -> Tuple[np.ndarray, torch.Tensor]:
    x1, y1, x2, y2 = clip_box(box, rgb_frame.shape[:2])
    if x2 <= x1 or y2 <= y1:
        return None, None
    face_region = rgb_frame[y1:y2, x1:x2]
    if face_region.size == 0:
        return None, None
    resized_face = cv2.resize(face_region, (target_size, target_size))
    face_tensor = torch.from_numpy(resized_face).permute(2, 0, 1).float() / 255.0
    return resized_face, face_tensor


def scale_boxes_landmarks(boxes, landmarks, scale_x, scale_y):
    """Scale bounding boxes and landmarks to display coordinates"""
    if boxes is not None:
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
    
    if landmarks is not None:
        landmarks = landmarks.copy()
        landmarks[:, :, 0] *= scale_x
        landmarks[:, :, 1] *= scale_y
    
    return boxes, landmarks


def capture_face_samples(
    cap: cv2.VideoCapture,
    mtcnn: MTCNN,
    total_samples: int,
    stable_frames_required: int,
    capture_delay: float,
) -> Tuple[List[Dict[str, object]], bool]:
    samples: List[Dict[str, object]] = []
    prompt_cycle = cycle(POSE_SEQUENCE)
    current_prompt = next(prompt_cycle)
    steady_frame_counter = 0
    aborted = False

    font = cv2.FONT_HERSHEY_SIMPLEX

    print("Starting capture. Follow on-screen prompts and hold each pose steady.")

    while len(samples) < total_samples:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from webcam.")
            break

        # Get original frame dimensions
        orig_height, orig_width = frame.shape[:2]
        
        # Process detection on original frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
        
        # Scale display by fixed factor (no padding, maintains aspect ratio)
        display_width = int(orig_width * DISPLAY_SCALE)
        display_height = int(orig_height * DISPLAY_SCALE)
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        # Calculate scaling factors
        scale_x = DISPLAY_SCALE
        scale_y = DISPLAY_SCALE
        x_offset = 0
        y_offset = 0
        
        # Scale boxes and landmarks to display coordinates
        display_boxes, display_landmarks = scale_boxes_landmarks(boxes, landmarks, scale_x, scale_y)
        
        best_box = None
        best_landmarks = None
        best_prob = 0.0
        display_best_box = None

        if boxes is not None and probs is not None:
            idx = int(np.argmax(probs))
            if probs[idx] is not None:
                best_prob = float(probs[idx])
                best_box = boxes[idx]
                best_landmarks = landmarks[idx] if landmarks is not None else None
                display_best_box = display_boxes[idx]

        detection_ready = False
        if best_box is not None and best_prob >= DETECTION_PROB_THRESHOLD and best_landmarks is not None:
            width = best_box[2] - best_box[0]
            height = best_box[3] - best_box[1]
            if width >= MIN_FACE_BOX_SIZE and height >= MIN_FACE_BOX_SIZE:
                detection_ready = evaluate_pose(best_landmarks, best_box, current_prompt)

        if detection_ready:
            steady_frame_counter += 1
        else:
            steady_frame_counter = 0

        # Draw on display frame
        if display_best_box is not None:
            x1, y1, x2, y2 = [int(v) for v in display_best_box]
            color = (0, 255, 0) if detection_ready else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display_frame,
                f"Confidence: {best_prob:.2f}",
                (x1, max(y1 - 10, 20)),
                font,
                0.6,
                color,
                2,
            )

        # Simple instruction display
        cv2.putText(display_frame, f"Instruction: {current_prompt}", (20, 40), font, 1.0, (0, 30, 255), 2)
        cv2.putText(
            display_frame,
            f"Progress: {len(samples)}/{total_samples}",
            (20, 80),
            font,
            0.8,
            (255, 0, 0),
            2,
        )

        if detection_ready:
            cv2.putText(
                display_frame,
                f"Hold steady: {steady_frame_counter}/{stable_frames_required}",
                (20, 120),
                font,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                display_frame,
                "Align your face with the instruction",
                (20, 120),
                font,
                0.7,
                (0, 255, 51),
                2,
            )

        cv2.putText(display_frame, "Press 'Q' to quit", (20, display_frame.shape[0] - 20), font, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Enrollment", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            aborted = True
            print("Capture aborted by user.")
            break

        if detection_ready and steady_frame_counter >= stable_frames_required and best_box is not None:
            face_rgb, face_tensor = crop_face(rgb_frame, best_box)
            if face_rgb is not None and face_tensor is not None:
                samples.append({"rgb": face_rgb, "tensor": face_tensor, "prompt": current_prompt})
                print(f"Captured sample {len(samples)}/{total_samples} ({current_prompt}).")
                steady_frame_counter = 0
                current_prompt = next(prompt_cycle)
                time.sleep(capture_delay)

    return samples, aborted


def save_face_data(
    samples: List[Dict[str, object]],
    name: str,
    user_id: str,
    embedder: InceptionResnetV1,
    device: torch.device,
) -> str:
    sanitized_name = safe_folder_name(name)
    session_folder = f"{user_id}_{sanitized_name}"
    output_dir = os.path.join(OUTPUT_ROOT, session_folder)
    os.makedirs(output_dir, exist_ok=True)

    for idx, sample in enumerate(samples, start=1):
        image_rgb = sample["rgb"]
        image_path = os.path.join(output_dir, f"face_{idx:02d}.jpg")
        cv2.imwrite(image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    thumbnail_rgb = samples[0]["rgb"]
    cv2.imwrite(os.path.join(output_dir, "thumbnail.jpg"), cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2BGR))

    face_stack = torch.stack([sample["tensor"] for sample in samples]).to(device)
    with torch.no_grad():
        embeddings = embedder(face_stack).cpu().numpy().astype(np.float32)

    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "embedding_mean.npy"), embeddings.mean(axis=0))

    meta = {
        "name": name,
        "id": user_id,
        "samples": len(samples),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, indent=2)

    return output_dir


def main() -> None:
    torch.set_grad_enabled(False)
    
    # Device selection
    print("Select processing device:")
    print("1. CPU")
    if torch.cuda.is_available():
        print(f"2. GPU ({torch.cuda.get_device_name(0)})")
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "2" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using device: {torch.cuda.get_device_name(0)} (cuda:0)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=device)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    try:
        samples, aborted = capture_face_samples(
            cap=cap,
            mtcnn=mtcnn,
            total_samples=TOTAL_SAMPLES,
            stable_frames_required=STABLE_FRAMES_REQUIRED,
            capture_delay=CAPTURE_DELAY_SECONDS,
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if aborted or len(samples) < TOTAL_SAMPLES:
        print("Enrollment incomplete. No data saved.")
        return

    name = input("Enter the user's name: ").strip()
    while not name:
        name = input("Name cannot be empty. Enter the user's name: ").strip()

    user_id = input("Enter the user's ID: ").strip()
    while not user_id:
        user_id = input("ID cannot be empty. Enter the user's ID: ").strip()

    output_folder = save_face_data(samples, name, user_id, embedder, device)
    print(f"Enrollment completed successfully. Data saved to: {output_folder}")


if __name__ == "__main__":
    main()
