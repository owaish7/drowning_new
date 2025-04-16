import cv2
import pathlib
from ultralytics import YOLO
import numpy as np

# Specify model path and task explicitly to avoid the warning
model_path = r"C:\yolo_test_2\owais_yolo_model_float16.tflite"
image_path = r"C:\yolo_test_2\img78.jpg"

# Define model task explicitly and specify the input size to match your model
model = YOLO(model_path, task='detect')

# Read the image
original_img = cv2.imread(image_path)
if original_img is None:
    print(f"Could not read {pathlib.Path(image_path).name}")
    exit()

# Get original dimensions for later
original_height, original_width = original_img.shape[:2]

# Run inference with explicit size parameter matching your model's expected input size (800x800)
results = model(original_img, imgsz=(800, 800), conf=0.2)

# Class names for your model
class_names = ["drowning", "floating"]

# Draw boxes on the image
result_img = original_img.copy()

# Process each detection
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    
    for box in boxes:
        # Get box coordinates, confidence and class
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # xyxy format (x1, y1, x2, y2)
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        
        # Get class name
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        
        # Draw rectangle
        color = (0, 255, 0) if class_name == "floating" else (0, 0, 255)  # Green for floating, Red for drowning
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label = max(y, label_height)
        cv2.rectangle(result_img, (x, y_label - label_height - baseline), (x + label_width, y_label), color, cv2.FILLED)
        cv2.putText(result_img, label, (x, y_label - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Save the result
output_path = r"C:\yolo_test_2\result_ultralytics_tflite.jpg"
cv2.imwrite(output_path, result_img)
print(f"Result saved to {output_path}")

# Print detection information
print(f"Detected {len(results[0])} objects")
for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x, y = int(x1), int(y1)
    w, h = int(x2 - x1), int(y2 - y1)
    confidence = float(box.conf[0])
    class_id = int(box.cls[0])
    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
    print(f"Detection {i+1}: {class_name} with confidence {confidence:.2f} at position [{x}, {y}, {w}, {h}]")