import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp
import threading
import math
import time

# Load YOLO model for bracelet detection
model = YOLO("runs/detect/train/weights/best.pt")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Create canvas
ret, frame = cap.read()
canvas_height, canvas_width = frame.shape[:2]
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Cursor smoothing
alpha = 0.3
smoothed_cursor = None

# Brush & Eraser settings
brush_size = 5
eraser_size = 50
prev_position = None
color = (0, 0, 255)  # Default Red
cursor_position = None

# Distance threshold (8 cm ≈ 80 pixels)
BRACELET_HAND_DISTANCE_THRESHOLD = 80  

# Global variables for YOLO results
bracelet_detected = False
bracelet_center = None
yolo_running = True  # Flag to keep YOLO running

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def detect_gesture(hand_landmarks):
    """Detect drawing or erasing gestures based on fingers."""
    finger_states = [
        hand_landmarks[4].x < hand_landmarks[3].x,  # Thumb
        hand_landmarks[8].y < hand_landmarks[6].y,  # Index
        hand_landmarks[12].y < hand_landmarks[10].y,  # Middle
        hand_landmarks[16].y < hand_landmarks[14].y,  # Ring
        hand_landmarks[20].y < hand_landmarks[18].y   # Pinky
    ]

    if finger_states == [0, 1, 0, 0, 0]:
        return "drawing"
    elif all(finger_states):
        return "erasing"
    return None  

def update_canvas(gesture, position):
    """Draw or erase based on gesture."""
    global prev_position, color, brush_size
    if gesture == "drawing":
        if prev_position:
            cv2.line(canvas, prev_position, position, color, brush_size)
        prev_position = position  
    elif gesture == "erasing":
        cv2.circle(canvas, position, eraser_size, (255, 255, 255), -1)
        prev_position = None  
    else:
        prev_position = None

def process_yolo():
    """Runs YOLO in a separate thread for speed."""
    global bracelet_center, bracelet_detected, yolo_running
    while yolo_running:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        results = model(frame)

        bracelet_detected = False
        bracelet_center = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if box.conf[0] > 0.7:  # Confidence threshold
                    bracelet_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    bracelet_detected = True
        
        time.sleep(0.05)  # Small delay to prevent CPU overload

# Start YOLO in a separate thread
yolo_thread = threading.Thread(target=process_yolo, daemon=True)
yolo_thread.start()

def process_frame():
    global smoothed_cursor, cursor_position, prev_position, canvas, yolo_running, color, brush_size
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            continue  # Skip frame if read failed
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(frame_rgb)

        if bracelet_detected and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                wrist = (int(hand_landmarks.landmark[0].x * frame.shape[1]), 
                         int(hand_landmarks.landmark[0].y * frame.shape[0]))

                if euclidean_distance(wrist, bracelet_center) > BRACELET_HAND_DISTANCE_THRESHOLD:
                    continue  

                index_tip = (int(hand_landmarks.landmark[8].x * frame.shape[1]), 
                             int(hand_landmarks.landmark[8].y * frame.shape[0]))
                
                # Apply cursor smoothing
                if smoothed_cursor is None:
                    smoothed_cursor = index_tip
                else:
                    smoothed_cursor = (
                        int(alpha * index_tip[0] + (1 - alpha) * smoothed_cursor[0]),
                        int(alpha * index_tip[1] + (1 - alpha) * smoothed_cursor[1])
                    )

                cursor_position = smoothed_cursor
                gesture = detect_gesture(hand_landmarks.landmark)
                if gesture:
                    update_canvas(gesture, smoothed_cursor)
                else:
                    prev_position = None

        # Display results
        display_canvas = canvas.copy()
        if cursor_position:
            cv2.circle(display_canvas, cursor_position, 5, (0, 0, 0), -1)  

        cv2.imshow("Gesture Whiteboard", display_canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            color = (0, 0, 255)  # Red
        elif key == ord('g'):
            color = (0, 255, 0)  # Green
        elif key == ord('b'):
            color = (255, 0, 0)  # Blue
        elif key == ord('w'):
            color = (255, 255, 255)  # White
        elif key == ord('+'):
            brush_size = min(brush_size + 2, 20)
            print(f"Brush Size Increased: {brush_size}")
        elif key == ord('-'):
            brush_size = max(brush_size - 2, 2)
            print(f"Brush Size Decreased: {brush_size}")
        elif key == ord('s'):
            cv2.imwrite("canvas_output.png", canvas)
            print("Canvas saved as canvas_output.png")
        elif key == ord('q'):
            yolo_running = False  # Stop YOLO thread
            break

    cap.release()
    cv2.destroyAllWindows()

# Start main frame processing
frame_thread = threading.Thread(target=process_frame)
frame_thread.start()
frame_thread.join()
