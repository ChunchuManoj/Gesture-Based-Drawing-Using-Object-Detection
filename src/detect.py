import cv2
import time
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('runs/detect/train/weights/best.pt')  # Update the path to your trained model

# Initialize the webcam (0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Set frame interval for smoother detection (adjust as needed)
frame_interval = 0.1  
last_time = time.time()

while True:
    if time.time() - last_time < frame_interval:
        continue  # Skip frames for performance improvement
    last_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from the webcam.")
        break

    # Perform inference with confidence threshold
    results = model(frame, conf=0.7, imgsz=320)

    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            class_name = result.names[class_id]  # Get class name

            if confidence > 0.7 and class_name.lower() == "bracelet":  # Filter for bracelets
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add label and confidence
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Bracelet Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
