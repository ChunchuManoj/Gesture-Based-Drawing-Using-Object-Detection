import cv2
import numpy as np
import mediapipe as mp
import threading
import queue

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Create a white canvas
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  
brush_size = 7
eraser_size = 50
prev_position = None  

gesture_queue = queue.Queue()
position_queue = queue.Queue()
frame_queue = queue.Queue(maxsize=1)  # Limit to prevent lag

# Gesture Detection Function
def detect_gesture(hand_landmarks):
    """Recognizes gestures based on finger states."""
    finger_states = [
        hand_landmarks[4].x < hand_landmarks[3].x  # Thumb
    ]
    
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        finger_states.append(hand_landmarks[tip].y < hand_landmarks[pip].y)

    if finger_states == [0, 1, 0, 0, 0]:  
        return "drawing"
    elif finger_states == [1, 1, 1, 1, 1]:  
        return "erasing"
    elif finger_states == [0, 1, 1, 0, 0]:  
        return "idle"  
    return None  

# Canvas Update Thread
def update_canvas():
    """Continuously updates the canvas in a separate thread."""
    global prev_position

    while True:
        try:
            gesture = gesture_queue.get(timeout=0.05)
            position = position_queue.get(timeout=0.05)
            
            if gesture == "drawing":
                if prev_position is not None:
                    cv2.line(canvas, prev_position, position, (0, 0, 255), brush_size, cv2.LINE_AA)
                prev_position = position
            elif gesture == "erasing":
                cv2.circle(canvas, position, eraser_size, (255, 255, 255), -1)
                prev_position = None
            else:
                prev_position = None  
        except queue.Empty:
            continue

# Start the canvas updating thread
canvas_thread = threading.Thread(target=update_canvas, daemon=True)
canvas_thread.start()

# Capture Video in a Separate Thread
def capture_video(cap):
    """Reads frames from webcam and adds them to the queue."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

cap = cv2.VideoCapture(0)

# Start video capture thread
video_thread = threading.Thread(target=capture_video, args=(cap,), daemon=True)
video_thread.start()

while True:
    if not frame_queue.empty():
        frame = frame_queue.get()
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                detected_gesture = detect_gesture(landmarks)
                tip_position = (int(landmarks[8].x * 640), int(landmarks[8].y * 480))

                gesture_queue.put(detected_gesture)
                position_queue.put(tip_position)

                cv2.putText(frame, f"Mode: {detected_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        cv2.imshow("Gesture Drawing", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
