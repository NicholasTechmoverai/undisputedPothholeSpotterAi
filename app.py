import cv2
import torch
import numpy as np
import torchvision
import os
import csv
import time
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

# ====================================
# CONFIGURATION
# ====================================
MODEL_PATH = "rdd2022_resnet50_fast.pt"
NUM_CLASSES = 5  # background + D00 + D10 + D20 + D40
CLASS_NAMES = ["background", "D00", "D10", "D20", "D40"]
CONF_THRESH = 0.5

SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, "detections_log.csv")

# ====================================
# CSV LOGGER SETUP
# ====================================
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "image_path", "label", "confidence", "x1", "y1", "x2", "y2"])

# ====================================
# LOAD MODEL
# ====================================
print("[INFO] Loading trained model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn(weights=None, num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"[INFO] Model loaded successfully on {device} ✅")

# ====================================
# SIMPLE HSV-BASED ROAD FILTER
# ====================================
def looks_like_road(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # gray-to-black asphalt color range
    mask = cv2.inRange(hsv, (0, 0, 40), (180, 60, 200))
    coverage = np.sum(mask > 0) / mask.size
    return coverage > 0.25

# ====================================
# DETECTION DRAW + SAVE LOGS
# ====================================
def draw_and_save_detections(frame, boxes, scores, labels, conf_thresh=CONF_THRESH):
    detections = []

    for box, score, label in zip(boxes, scores, labels):
        if score < conf_thresh:
            continue

        color = (0, 255, 0)
        x1, y1, x2, y2 = box.int().tolist()
        cls = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)

        # Draw boxes on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls} {score:.2f}", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        detections.append({
            "label": cls,
            "confidence": round(score.item(), 3),
            "bbox": [x1, y1, x2, y2]
        })

    # ✅ Save only if at least one detection is found
    if len(detections) > 0:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_name = f"pothole_{timestamp}.jpg"
        image_path = os.path.join(SAVE_DIR, image_name)

        # ✅ Save the drawn frame (with bounding boxes)
        cv2.imwrite(image_path, frame)
        print(f"[INFO] 💾 Saved detection image with boxes: {image_path}")

        # ✅ Log detections
        with open(CSV_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            for det in detections:
                writer.writerow([
                    timestamp,
                    image_path,
                    det["label"],
                    det["confidence"],
                    *det["bbox"]
                ])

    return frame

# ====================================
# MAIN CAMERA LOOP
# ====================================
def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access the webcam.")
        return

    print("🎥 Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame, skipping...")
            continue

        # Skip non-road frames
        if not looks_like_road(frame):
            cv2.putText(frame, "⛔ Non-road frame skipped", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("🚗 Live Pothole Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Run inference
        img_tensor = to_tensor(frame).to(device)
        with torch.no_grad():
            preds = model([img_tensor])[0]

        boxes = preds["boxes"]
        scores = preds["scores"]
        labels = preds["labels"]

        frame = draw_and_save_detections(frame, boxes, scores, labels)

        cv2.imshow("🚗 Live Pothole Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera stopped.")

# ====================================
# RUN
# ====================================
if __name__ == "__main__":
    run_camera()
