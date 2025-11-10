import cv2
import tensorflow as tf
import numpy as np

# --- Step 1: Load the frozen .pb graph ---
pb_file = r"From_azure/model.pb"

def load_graph(pb_file):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_file, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    return graph

graph = load_graph(pb_file)

# Start TF session
sess = tf.compat.v1.Session(graph=graph)

# --- Step 2: Inspect tensor names (optional but recommended) ---
for op in graph.get_operations():
    print(op.name)  # look for input/output tensor names

# Replace these with the actual names from the above print
input_tensor_name = "Placeholder:0"
output_tensor_name = "model_outputs:0"


input_tensor = graph.get_tensor_by_name(input_tensor_name)
output_tensor = graph.get_tensor_by_name(output_tensor_name)

# Labels
labels = ["major", "minor", "none"]

# --- Step 3: OpenCV webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame (resize to model input size)
    img = cv2.resize(frame, (224, 224)).astype(np.float32)  # adjust if your model uses different size
    img = np.expand_dims(img, axis=0)

    # Run inference
    pred = sess.run(output_tensor, feed_dict={input_tensor: img})
    pred_label = labels[np.argmax(pred)]

    # Display result
    cv2.putText(frame, f"Prediction: {pred_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
