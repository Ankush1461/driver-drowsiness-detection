import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# SETTINGS
SOURCE_DIR = 'dataset/train'
CLEAN_DIR = 'dataset/train_cropped'
TARGET_SIZE = (96, 96)

# 1. DOWNLOAD THE MODEL BUFFER (MediaPipe Tasks requires a .task file)
# If you don't have this, download 'face_landmarker.task' from Google's MediaPipe site
model_path = 'face_landmarker.task' 

def crop_and_save_faces():
    # Configure the Task API
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
    
    detector = vision.FaceLandmarker.create_from_options(options)

    if not os.path.exists(CLEAN_DIR): os.makedirs(CLEAN_DIR)

    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path): continue
        
        save_class_path = os.path.join(CLEAN_DIR, class_name)
        os.makedirs(save_class_path, exist_ok=True)
        print(f"🚀 Cleaning: {class_name}")

        for img_name in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_name))
            if img is None: continue
            
            h, w, _ = img.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
                
                x_min, y_min = np.min(coords, axis=0)
                x_max, y_max = np.max(coords, axis=0)
                
                side = int(max(x_max - x_min, y_max - y_min) * 1.3)
                cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
                
                nx, ny = max(0, cx - side // 2), max(0, cy - side // 2)
                nx1, ny1 = min(w, nx + side), min(h, ny + side)

                face_crop = img[ny:ny1, nx:nx1]
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(os.path.join(save_class_path, img_name), face_crop)

    print("✅ DATASET PURIFIED")

if __name__ == "__main__":
    crop_and_save_faces()