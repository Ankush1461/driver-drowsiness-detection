import threading
import cv2
import numpy as np
import winsound

# Try LiteRT if available, else fallback to standard TFLite
try:
    import ai_edge_litert.interpreter as litert
    print("Using LiteRT (New Google TFLite Runtime)")
except ImportError:
    try:
        import tflite_runtime.interpreter as litert
        print("Using tflite_runtime")
    except ImportError:
        import tensorflow as tf
        litert = tf.lite
        print("Using tf.lite from full Tensorflow")

# Load the blazing fast .tflite model explicitly compiled by the new MobileNetV2 pipeline
interpreter = litert.Interpreter(model_path="drowsiness.tflite", num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize OpenCV DNN Face Detector (ResNet-10 SSD)
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
dna_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Set up a video capture object to capture frames from a camera or file
cap = cv2.VideoCapture(0)  # Changed to zero for live webcam testing, or set back to Media/drowsy_vid.mp4

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        
        # OpenCV DNN Face Detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        dna_net.setInput(blob)
        detections = dna_net.forward()
        
        res_label = "AWAKE"
        color = (0, 255, 0)
        
        best_idx = -1
        highest_conf = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > highest_conf:
                highest_conf = confidence
                best_idx = i
        
        if best_idx != -1:
            box = detections[0, 0, best_idx, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            
            roi = frame[max(0, y):min(h, y2), max(0, x):min(w, x2)]
            
            if roi.size > 0:
                # Convert OpenCV BGR Image natively captured from webcam into RGB
                img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                
                # Resize directly to Model's exact input shape (96x96)
                img_resized = cv2.resize(img_rgb, (96, 96))
                
                # MobileNetV3 Preprocessing expects values in [0, 255]. It automatically maps the input properly internally.
                img_batch = np.expand_dims(img_resized, axis=0).astype(np.float32)

                # Execute Extreme Edge Detection using LiteRT
                interpreter.set_tensor(input_details[0]['index'], img_batch)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])

                # Log probability for debugging
                prob = float(prediction[0][0])
                cv2.putText(frame, f"Conf: {prob:.2f}", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1)

                # Binary classification logic (sigmoid output)
                if prob > 0.5:
                    res_label = "DROWSY"
                    color = (0, 0, 255)
                    # threading.Thread(target=winsound.PlaySound, args=('alarm.wav', winsound.SND_ASYNC)).start()
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x2, y2), color, 3)
            
        cv2.putText(frame, f"STATUS: {res_label}", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

        cv2.imshow("Driver AI Test Env", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("An error occurred:", e)

finally:
    cap.release()
    cv2.destroyAllWindows()
