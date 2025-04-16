import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import time

# Load the YOLO TFLite model
model_path = r"C:\yolo_test_2\owais_yolo_model_float16.tflite"
model = YOLO(model_path, task="detect")

# Define class names
class_names = ["drowning", "floating"]

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (800, 800)  # match model input size
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print("🚀 Running real-time detection... Press Ctrl+C to stop.")

try:
    while True:
        # Capture image from camera
        frame = picam2.capture_array()

        # Run YOLO inference
        results = model(frame, imgsz=(800, 800), conf=0.2)

        # Draw results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"

                color = (0, 0, 255) if class_name == "drowning" else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                label = f"{class_name}: {confidence:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y), color, cv2.FILLED)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display frame
        cv2.imshow("YOLO PiCam Detection", frame)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("🛑 Detection stopped.")

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
