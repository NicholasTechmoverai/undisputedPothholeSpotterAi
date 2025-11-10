import tensorflow as tf
import cv2
import numpy as np
import time
import os
from datetime import datetime

# Configuration
CLASS_NAMES = ["background", "D00", "D10", "D20", "D40"]
CONF_THRESH = 0.5
LOG_FILE = "pothole_detections_log.csv"

# Create log file with headers if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as f:
        f.write("timestamp,class,confidence,x1,y1,x2,y2,image_path\n")

class RoadDetector:
    def __init__(self):
        self.road_detected = False
        self.road_confidence = 0.0
        
    def detect_road(self, image):
        """Simple road detection using color and texture analysis"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Asphalt road color range (gray to dark colors)
        lower_asphalt = np.array([0, 0, 40])
        upper_asphalt = np.array([180, 60, 200])
        
        # Create mask for road-like colors
        mask = cv2.inRange(hsv, lower_asphalt, upper_asphalt)
        
        # Calculate road coverage percentage
        road_coverage = np.sum(mask > 0) / mask.size
        
        # Additional texture analysis - road typically has more edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combined confidence score
        self.road_confidence = (road_coverage * 0.7 + edge_density * 0.3)
        self.road_detected = self.road_confidence > 0.3
        
        return self.road_detected, self.road_confidence

def log_detection(class_name, confidence, bbox, image=None):
    """Log detection to CSV file and save image if provided"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    x1, y1, x2, y2 = bbox
    
    # Save detection image
    image_path = ""
    if image is not None:
        image_path = f"detections/detection_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        os.makedirs("detections", exist_ok=True)
        cv2.imwrite(image_path, image)
    
    # Log to CSV
    with open(LOG_FILE, 'a') as f:
        f.write(f"{timestamp},{class_name},{confidence:.3f},{x1},{y1},{x2},{y2},{image_path}\n")
    
    print(f"📝 LOGGED: {class_name} ({confidence:.2f}) at {timestamp}")

def draw_detections_tflite(image, boxes, scores, labels, conf_thresh=0.5):
    """Draw detections for TFLite model output with enhanced visualization"""
    h, w = image.shape[:2]
    
    detection_count = 0
    detected_classes = set()
    
    # Enhanced colors for better visibility
    colors = {
        "D00": (0, 255, 0),      # Green - Minor
        "D10": (0, 255, 255),    # Yellow - Moderate
        "D20": (0, 165, 255),    # Orange - Severe
        "D40": (0, 0, 255)       # Red - Critical
    }
    
    severity_levels = {
        "D00": "MINOR",
        "D10": "MODERATE", 
        "D20": "SEVERE",
        "D40": "CRITICAL"
    }
    
    if len(boxes) == 0 or len(scores) == 0:
        return image, detected_classes, detection_count
    
    # Handle different output formats
    if len(boxes.shape) == 1:
        # Flattened array - assume 4 values per box
        num_boxes = len(boxes) // 4
        for i in range(num_boxes):
            if i >= len(scores):
                break
                
            score = scores[i]
            if score < conf_thresh:
                continue
                
            # Extract box coordinates
            box_idx = i * 4
            if box_idx + 3 < len(boxes):
                x1, y1, x2, y2 = boxes[box_idx:box_idx+4]
                
                # Convert to pixel coordinates if normalized
                if x1 <= 1.0 and y1 <= 1.0:
                    x1, x2 = int(x1 * w), int(x2 * w)
                    y1, y2 = int(y1 * h), int(y2 * h)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure coordinates are within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Get class label
                class_id = 0
                if len(labels) > i:
                    class_id = int(labels[i])
                
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
                
                if class_name != "background":
                    detected_classes.add(class_name)
                    color = colors.get(class_name, (255, 255, 255))
                    severity = severity_levels.get(class_name, "UNKNOWN")
                    
                    # Draw enhanced bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw filled label background
                    label = f"{class_name} {severity} ({score:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(image, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Log the detection
                    log_detection(class_name, score, (x1, y1, x2, y2), image)
                    
                    detection_count += 1
    
    else:
        # Handle 2D array format
        for i in range(min(len(scores), len(boxes))):
            score = scores[i]
            if score < conf_thresh:
                continue
                
            if len(boxes[i]) >= 4:
                x1, y1, x2, y2 = boxes[i][:4]
                
                # Convert to pixel coordinates if normalized
                if x1 <= 1.0 and y1 <= 1.0:
                    x1, x2 = int(x1 * w), int(x2 * w)
                    y1, y2 = int(y1 * h), int(y2 * h)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure coordinates are within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Get class label
                class_id = int(labels[i]) if i < len(labels) else 0
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
                
                if class_name != "background":
                    detected_classes.add(class_name)
                    color = colors.get(class_name, (255, 255, 255))
                    severity = severity_levels.get(class_name, "UNKNOWN")
                    
                    # Draw enhanced bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw filled label background
                    label = f"{class_name} {severity} ({score:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(image, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Log the detection
                    log_detection(class_name, score, (x1, y1, x2, y2), image)
                    
                    detection_count += 1
    
    return image, detected_classes, detection_count

def create_status_display(image, road_detected, road_confidence, detected_classes, detection_count, fps):
    """Create enhanced status display overlay"""
    h, w = image.shape[:2]
    
    # Create semi-transparent overlay for status
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
    
    # System title
    cv2.putText(image, "🚗 AI Pothole Detection System", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Road status with color coding
    road_status = "ROAD DETECTED" if road_detected else "NO ROAD DETECTED"
    road_color = (0, 255, 0) if road_detected else (0, 0, 255)
    cv2.putText(image, f"Road: {road_status} ({road_confidence:.2f})", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, road_color, 2)
    
    # Detection status
    if detection_count > 0:
        status_text = f"POTHOLES DETECTED: {detection_count}"
        classes_text = f"Types: {', '.join(detected_classes)}" if detected_classes else ""
        cv2.putText(image, status_text, (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if classes_text:
            cv2.putText(image, classes_text, (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    else:
        cv2.putText(image, "No Potholes Detected", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # FPS and model info
    cv2.putText(image, f"FPS: {fps:.1f} | TFLite Model", (w - 250, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Legend in corner
    legend_y = h - 10
    for i, (class_name, color) in enumerate({
        "D00-MINOR": (0, 255, 0),
        "D10-MODERATE": (0, 255, 255),
        "D20-SEVERE": (0, 165, 255),
        "D40-CRITICAL": (0, 0, 255)
    }.items()):
        cv2.putText(image, class_name, (10, legend_y - i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return image

def run_tflite_detection():
    """Enhanced TFLite detection with road detection and better presentation"""
    
    print("[INFO] Loading TFLite model...")
    try:
        interpreter = tf.lite.Interpreter(model_path="tensorflow_model/model.tflite")
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ TFLite model loaded successfully!")
        print(f"📥 Input: {input_details[0]['shape']}")
        print(f"📤 Outputs: {[out['name'] for out in output_details]}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize road detector
    road_detector = RoadDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access the webcam.")
        return
    
    print("\n🎥 AI Pothole Detection System Started!")
    print("📍 Features:")
    print("   • Real-time road detection")
    print("   • Pothole classification (D00, D10, D20, D40)")
    print("   • Automatic logging to CSV")
    print("   • Severity-based color coding")
    print("   • Press 'q' to quit, 'd' for debug info")
    print("=" * 50)
    
    fps_counter = 0
    fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            continue
        
        # Road detection
        road_detected, road_confidence = road_detector.detect_road(frame)
        
        # Model inference
        input_shape = input_details[0]['shape']
        input_data = cv2.resize(frame, (input_shape[2], input_shape[1]))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        inference_time = time.time() - start_time
        
        # Get outputs
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        scores = interpreter.get_tensor(output_details[1]['index'])[0]
        labels = interpreter.get_tensor(output_details[2]['index'])[0]
        
        # Process detections
        result_frame, detected_classes, detection_count = draw_detections_tflite(
            frame.copy(), boxes, scores, labels
        )
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter / (time.time() - fps_time)
            fps_counter = 0
            fps_time = time.time()
        else:
            fps = 1.0 / inference_time if inference_time > 0 else 0
        
        # Create enhanced display
        result_frame = create_status_display(
            result_frame, road_detected, road_confidence, 
            detected_classes, detection_count, fps
        )
        
        # Show result
        cv2.imshow("🚗 AI Pothole Detection System - ResNet50 + TFLite", result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            print("\n🔍 Debug Information:")
            print(f"   Road Detected: {road_detected} (Confidence: {road_confidence:.2f})")
            print(f"   Potholes Found: {detection_count}")
            print(f"   Classes: {list(detected_classes)}")
            print(f"   FPS: {fps:.1f}")
            print(f"   Inference Time: {inference_time*1000:.1f}ms")
        
        elif key == ord('s'):
            # Save current frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshots/system_snapshot_{timestamp}.jpg"
            os.makedirs("snapshots", exist_ok=True)
            cv2.imwrite(filename, result_frame)
            print(f"💾 System snapshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Session ended. Detection log saved to:", LOG_FILE)

# Run the enhanced detection system
if __name__ == "__main__":
    print("=" * 60)
    print("           🚗 AI POTHOLES DETECTION SYSTEM")
    print("=" * 60)
    print("Architecture: ResNet50 Backbone + Custom Detection Head")
    print("Classes: D00 (Minor), D10 (Moderate), D20 (Severe), D40 (Critical)")
    print("Features: Real-time Road Detection + Automated Logging")
    print("=" * 60)
    
    run_tflite_detection()