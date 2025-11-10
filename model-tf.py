import tensorflow as tf
import cv2
import numpy as np
import time

def draw_detections_tflite(image, boxes, scores, labels, conf_thresh=0.5):
    """Draw detections for TFLite model output with proper class labels"""
    h, w = image.shape[:2]
    
    # Pothole class names - adjust based on your model
    CLASS_NAMES = ["background", "D00", "D10", "D20", "D40"]
    
    if len(boxes) == 0 or len(scores) == 0:
        cv2.putText(image, "No detections", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return image
    
    print(f"🔍 Debug - Boxes: {boxes.shape}, Scores: {scores.shape}, Labels: {labels.shape}")
    print(f"🔍 Sample scores: {scores[:3]}")
    print(f"🔍 Sample labels: {labels[:3]}")
    
    detection_count = 0
    
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
                
            # Extract box coordinates (assuming [x1, y1, x2, y2] format)
            box_idx = i * 4
            if box_idx + 3 < len(boxes):
                x1, y1, x2, y2 = boxes[box_idx:box_idx+4]
                
                # Convert to pixel coordinates if normalized
                if x1 <= 1.0 and y1 <= 1.0:
                    x1, x2 = int(x1 * w), int(x2 * w)
                    y1, y2 = int(y1 * h), int(y2 * h)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class label
                class_id = 0
                if len(labels) > i:
                    class_id = int(labels[i])
                
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
                
                # Choose color based on class
                colors = {
                    "D00": (0, 255, 0),    # Green
                    "D10": (0, 255, 255),  # Yellow
                    "D20": (0, 165, 255),  # Orange
                    "D40": (0, 0, 255)     # Red
                }
                color = colors.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with class and confidence
                label = f"{class_name}: {score:.2f}"
                cv2.putText(image, label, (x1, max(25, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                detection_count += 1
                print(f"🎯 Detection: {class_name} {score:.2f} at [{x1},{y1},{x2},{y2}]")
    
    else:
        # Handle 2D array format
        for i in range(min(len(scores), len(boxes))):
            score = scores[i]
            if score < conf_thresh:
                continue
                
            # Extract box coordinates
            if len(boxes[i]) >= 4:
                x1, y1, x2, y2 = boxes[i][:4]
                
                # Convert to pixel coordinates if normalized
                if x1 <= 1.0 and y1 <= 1.0:
                    x1, x2 = int(x1 * w), int(x2 * w)
                    y1, y2 = int(y1 * h), int(y2 * h)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class label
                class_id = int(labels[i]) if i < len(labels) else 0
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
                
                # Choose color based on class
                colors = {
                    "D00": (0, 255, 0),    # Green
                    "D10": (0, 255, 255),  # Yellow  
                    "D20": (0, 165, 255),  # Orange
                    "D40": (0, 0, 255)     # Red
                }
                color = colors.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with class and confidence
                label = f"{class_name}: {score:.2f}"
                cv2.putText(image, label, (x1, max(25, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                detection_count += 1
                print(f"🎯 Detection: {class_name} {score:.2f} at [{x1},{y1},{x2},{y2}]")
    
    # Display detection count
    count_text = f"Detections: {detection_count}"
    cv2.putText(image, count_text, (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image

def run_tflite_detection():
    """Use TFLite model - fastest option"""
    
    print("[INFO] Loading TFLite model...")
    try:
        interpreter = tf.lite.Interpreter(model_path="tensorflow_model/model.tflite")
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ TFLite model loaded successfully!")
        print(f"📥 Input: {input_details[0]['shape']}")
        print(f"📤 Outputs: {[out['name'] for out in output_details]}")
        print(f"📤 Output shapes: {[out['shape'] for out in output_details]}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access the webcam.")
        return
    
    print("🎥 TFLite Detection - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            continue
        
        # Preprocess
        input_shape = input_details[0]['shape']
        input_data = cv2.resize(frame, (input_shape[2], input_shape[1]))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        # Run inference
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        inference_time = time.time() - start_time
        
        # Get outputs
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        scores = interpreter.get_tensor(output_details[1]['index'])[0]
        labels = interpreter.get_tensor(output_details[2]['index'])[0]
        
        print(f"📊 Raw outputs - Boxes: {boxes[:2]}, Scores: {scores[:3]}, Labels: {labels[:3]}")
        
        # Draw results
        result_frame = draw_detections_tflite(frame.copy(), boxes, scores, labels)
        
        # Display info
        fps_text = f"FPS: {1/inference_time:.1f}" if inference_time > 0 else "FPS: Calculating..."
        cv2.putText(result_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, "TFLite Pothole Detection", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("📱 TFLite Pothole Detection", result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):  # Press 'd' to debug
            print("🔍 Debug Info:")
            print(f"   Boxes shape: {boxes.shape}, sample: {boxes[:2]}")
            print(f"   Scores shape: {scores.shape}, sample: {scores[:5]}")
            print(f"   Labels shape: {labels.shape}, sample: {labels[:5]}")
    
    cap.release()
    cv2.destroyAllWindows()

def run_savedmodel_detection():
    """Fallback SavedModel version"""
    print("🔄 Falling back to SavedModel...")
    
    try:
        model = tf.saved_model.load("tensorflow_model/saved_model")
        
        if "serving_default" in model.signatures:
            infer = model.signatures["serving_default"]
        else:
            infer = list(model.signatures.values())[0]
        
        print("✅ SavedModel loaded successfully!")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            input_frame = cv2.resize(frame, (640, 640))
            input_tensor = tf.convert_to_tensor(input_frame, dtype=tf.float32)
            input_tensor = tf.expand_dims(input_tensor, 0)
            
            outputs = infer(input_tensor)
            
            boxes = outputs['boxes'].numpy()[0] if 'boxes' in outputs else np.array([])
            scores = outputs['scores'].numpy()[0] if 'scores' in outputs else np.array([])
            labels = outputs['labels'].numpy()[0] if 'labels' in outputs else np.array([])
            
            result_frame = draw_detections_tflite(frame, boxes, scores, labels)
            cv2.imshow("💾 SavedModel Detection", result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ SavedModel also failed: {e}")

# Run the detection
if __name__ == "__main__":
    print("=" * 50)
    print("🛠️ Pothole Detection System")
    print("=" * 50)
    print("Classes: D00, D10, D20, D40")
    print("Press 'q' to quit, 'd' for debug info")
    print("=" * 50)
    
    print("Choose method:")
    print("1. TFLite (Fastest)")
    print("2. SavedModel (Fallback)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        run_savedmodel_detection()
    else:
        run_tflite_detection()