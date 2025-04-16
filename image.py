import cv2
import pathlib
import pygame
import time
from ultralytics import YOLO
import numpy as np

# Initialize pygame for audio playback
pygame.init()
pygame.mixer.init()

# Load the beep sound
beep_sound_path = r"267555__alienxxx__beep_sequence_02.wav"
beep_sound = pygame.mixer.Sound(beep_sound_path)

# Specify model path and task explicitly
model_path = r"owais_yolo_model_float32.tflite"

# Define model task explicitly and specify the input size to match your model
model = YOLO(model_path, task='detect')

# Class names for your model (reversed based on previous feedback)
class_names = ["drowning", "floating"]

# Flag to track if sound is currently playing to avoid overlapping sounds
sound_playing = False
last_sound_time = 0
sound_cooldown = 2  # Seconds between alerts to avoid constant beeping

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Starting webcam detection. Press 'q' to quit.")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Run model inference on the frame
    results = model(frame, imgsz=(800, 800), conf=0.2)
    
    # Track if drowning was detected in this frame
    drowning_detected = False
    
    # Process results
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates, confidence and class
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Get class name
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            
            # Check if drowning is detected
            if class_name == "drowning":
                drowning_detected = True
            
            # Draw rectangle
            color = (0, 255, 0) if class_name == "floating" else (0, 0, 255)  # Green for floating, Red for drowning
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label = max(y, label_height)
            cv2.rectangle(frame, (x, y_label - label_height - baseline), (x + label_width, y_label), color, cv2.FILLED)
            cv2.putText(frame, label, (x, y_label - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Play alert sound if drowning is detected and sound cooldown has passed
    current_time = time.time()
    if drowning_detected and (current_time - last_sound_time > sound_cooldown):
        beep_sound.play()
        last_sound_time = current_time
        
        # Add visual alert on screen
        cv2.putText(frame, "DROWNING ALERT!", (width//2 - 150, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    # Display the frame
    cv2.imshow("Drowning Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Quit pygame
pygame.quit()

# import cv2
# import pathlib
# import pygame
# import time
# import threading
# import queue
# from ultralytics import YOLO
# import numpy as np

# # Initialize pygame for audio playback
# pygame.init()
# pygame.mixer.init()

# # Class to implement multithreaded video detection
# class VideoDetector:
#     def __init__(self, model_path, sound_path, cam_id=0, queue_size=10):
#         # Load the model
#         print("Loading YOLO model...")
#         self.model = YOLO(model_path, task='detect')
        
#         # Load the beep sound
#         self.beep_sound = pygame.mixer.Sound(sound_path)
        
#         # Class labels
#         self.class_names = ["drowning", "floating"]
        
#         # Initialize camera
#         self.cam_id = cam_id
#         self.cap = None
        
#         # Thread control
#         self.stopped = False
#         self.frame_queue = queue.Queue(maxsize=queue_size)
#         self.result_queue = queue.Queue(maxsize=queue_size)
        
#         # Alert control
#         self.last_sound_time = 0
#         self.sound_cooldown = 2  # Seconds between alerts
        
#         # Frame skipping for performance
#         self.process_every_n_frames = 2  # Process every 2nd frame

#     def start(self):
#         # Create threads
#         self.stopped = False
        
#         print("Starting camera thread...")
#         # Start camera thread
#         camera_thread = threading.Thread(target=self.camera_thread)
#         camera_thread.daemon = True
#         camera_thread.start()
        
#         print("Starting inference thread...")
#         # Start inference thread
#         inference_thread = threading.Thread(target=self.inference_thread)
#         inference_thread.daemon = True
#         inference_thread.start()
        
#         return self
    
#     def camera_thread(self):
#         # Initialize camera
#         self.cap = cv2.VideoCapture(self.cam_id)
#         if not self.cap.isOpened():
#             print("Error: Could not open webcam")
#             self.stopped = True
#             return
        
#         # Set camera properties for better performance
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
#         # Process frames
#         frame_count = 0
#         while not self.stopped:
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("Failed to grab frame")
#                 self.stopped = True
#                 break
            
#             # Only process every nth frame for better performance
#             frame_count += 1
#             if frame_count % self.process_every_n_frames != 0:
#                 continue
            
#             # Clear queue if it's full to avoid lag
#             if self.frame_queue.full():
#                 try:
#                     self.frame_queue.get_nowait()
#                 except queue.Empty:
#                     pass
            
#             # Add new frame to queue
#             self.frame_queue.put(frame)
    
#     def inference_thread(self):
#         while not self.stopped:
#             if not self.frame_queue.empty():
#                 # Get frame from queue
#                 frame = self.frame_queue.get()
                
#                 # Run inference at original 800x800 resolution as the model requires
#                 results = self.model(frame, imgsz=(800, 800), conf=0.4)
                
#                 # Clear queue if full
#                 if self.result_queue.full():
#                     try:
#                         self.result_queue.get_nowait()
#                     except queue.Empty:
#                         pass
                
#                 # Add results to queue
#                 self.result_queue.put((frame.copy(), results))
#             else:
#                 # Small sleep to reduce CPU usage, but shorter than before
#                 time.sleep(0.005)
    
#     def get_next_frame(self):
#         if self.result_queue.empty():
#             return None, None
        
#         frame, results = self.result_queue.get()
#         return frame, results
    
#     def stop(self):
#         self.stopped = True
#         if self.cap is not None:
#             self.cap.release()
    
#     def process_frame(self, frame, results):
#         # Track if drowning was detected in this frame
#         drowning_detected = False
#         height, width = frame.shape[:2]
        
#         # Process results
#         for result in results:
#             boxes = result.boxes
            
#             for box in boxes:
#                 # Get box coordinates, confidence and class
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 x, y = int(x1), int(y1)
#                 w, h = int(x2 - x1), int(y2 - y1)
#                 confidence = float(box.conf[0])
#                 class_id = int(box.cls[0])
                
#                 # Get class name
#                 class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                
#                 # Check if drowning is detected
#                 if class_name == "drowning":
#                     drowning_detected = True
                
#                 # Draw rectangle
#                 color = (0, 255, 0) if class_name == "floating" else (0, 0, 255)  # Green for floating, Red for drowning
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
#                 # Draw label
#                 label = f"{class_name}: {confidence:.2f}"
#                 (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#                 y_label = max(y, label_height)
#                 cv2.rectangle(frame, (x, y_label - label_height - baseline), (x + label_width, y_label), color, cv2.FILLED)
#                 cv2.putText(frame, label, (x, y_label - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         # Play alert sound if drowning is detected and sound cooldown has passed
#         current_time = time.time()
#         if drowning_detected and (current_time - self.last_sound_time > self.sound_cooldown):
#             self.beep_sound.play()
#             self.last_sound_time = current_time
            
#             # Add visual alert on screen
#             cv2.putText(frame, "DROWNING ALERT!", (width//2 - 150, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
#         return frame

# # Main execution
# def main():
#     # File paths
#     model_path = r"owais_yolo_model_float32.tflite"
#     sound_path = r"267555__alienxxx__beep_sequence_02.wav"
    
#     print("Initializing drowning detection system...")
    
#     # Initialize and start the detector
#     detector = VideoDetector(model_path, sound_path)
#     detector.start()
    
#     # Wait a bit for initialization to complete
#     print("Warming up camera and model...")
#     time.sleep(1)
    
#     print("Starting webcam detection. Press 'q' to quit.")
    
#     # Display loading screen while waiting for first frame
#     loading_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#     cv2.putText(loading_frame, "Loading camera feed...", (180, 240), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#     cv2.imshow("Drowning Detection", loading_frame)
#     cv2.waitKey(1)
    
#     # FPS calculation variables
#     fps_start_time = time.time()
#     fps_frame_count = 0
#     fps = 0
    
#     # Main loop
#     while not detector.stopped:
#         frame, results = detector.get_next_frame()
        
#         if frame is not None and results is not None:
#             # Process and display frame
#             processed_frame = detector.process_frame(frame, results)
            
#             # Calculate and display FPS
#             fps_frame_count += 1
#             if fps_frame_count >= 10:
#                 end_time = time.time()
#                 fps = fps_frame_count / (end_time - fps_start_time)
#                 fps_start_time = end_time
#                 fps_frame_count = 0
            
#             # Display FPS on frame
#             cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
#             cv2.imshow("Drowning Detection", processed_frame)
        
#         # Check for quit with a shorter wait time
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Clean up
#     print("Shutting down...")
#     detector.stop()
#     cv2.destroyAllWindows()
#     pygame.quit()

# if __name__ == "__main__":
#     main()