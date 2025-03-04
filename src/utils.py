import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp

# Load YOLO model for bracelet detection
model = YOLO("runs/detect/train/weights/best.pt")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create a transparent canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
brush_size = 5
eraser_size = 50
prev_position = None  # This will reset correctly

# Open webcam
cap = cv2.VideoCapture(0)
bracelet_detected = False
bracelet_center = None
bracelet_hand = None  # Tracks which hand is wearing the bracelet

# Function to detect gestures
def detect_gesture(hand_landmarks):
    finger_states = []

    # Thumb
    finger_states.append(hand_landmarks[4].x < hand_landmarks[3].x)
    # Other fingers (Index, Middle, Ring, Pinky)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        finger_states.append(hand_landmarks[tip].y < hand_landmarks[pip].y)

    if finger_states == [0, 1, 0, 0, 0]:  
        return "drawing"
    elif finger_states == [1, 1, 1, 1, 1]:  
        return "erasing"
    return None  

# Function to update the canvas
def update_canvas(gesture, position):
    global prev_position

    if gesture == "drawing":
        if prev_position is not None:
            cv2.line(canvas, prev_position, position, (0, 0, 255), brush_size)
        prev_position = position  
    elif gesture == "erasing":
        cv2.circle(canvas, position, eraser_size, (0, 0, 0), thickness=-1)
        prev_position = None  
    else:
        prev_position = None  # Reset on gesture stop

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Step 1: Run object detection every frame
    results = model(frame)
    bracelet_detected = False
    bracelet_center = None
    bracelet_hand = None  

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()

            # Calculate bracelet center
            bracelet_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            bracelet_detected = True

            # Determine if the bracelet is on the left or right side of the frame
            bracelet_hand = "right" if bracelet_center[0] > frame.shape[1] // 2 else "left"

            # Draw bounding box for bracelet
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Bracelet {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Step 2: If bracelet is detected, check for hand gestures
    if bracelet_detected:
        results_hands = hands.process(frame_rgb)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                hand_label = handedness.classification[0].label.lower()  

                # Ignore the hand that does not wear the bracelet
                if hand_label != bracelet_hand:
                    continue  

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get wrist position
                wrist = (int(hand_landmarks.landmark[0].x * frame.shape[1]), 
                         int(hand_landmarks.landmark[0].y * frame.shape[0]))

                # Measure distance between bracelet and wrist
                if bracelet_center:
                    distance = np.linalg.norm(np.array(bracelet_center) - np.array(wrist))
                    
                    # Only allow gestures if bracelet is within 100 pixels of the wrist
                    if distance > 100:
                        cv2.putText(frame, "Bracelet too far", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        continue  

                # Get index fingertip position
                index_tip = (int(hand_landmarks.landmark[8].x * frame.shape[1]), 
                             int(hand_landmarks.landmark[8].y * frame.shape[0]))

                # Detect gesture
                gesture = detect_gesture(hand_landmarks.landmark)

                if gesture:
                    update_canvas(gesture, index_tip)
                    cv2.putText(frame, f"Mode: {gesture}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    prev_position = None  # Reset so new strokes don't connect

    # Blend canvas over frame with 60% transparency
    blended = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)

    cv2.imshow("Object + Gesture Control", blended)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
