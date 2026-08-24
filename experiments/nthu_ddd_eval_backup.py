import numpy as np
import os, time, sys, glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import cv2

NTHU_PATH = "external_dataset/NTHU-DDD"
MODEL_PATH = "drowsiness_robust.tflite"
SSD_PROTO = "deploy.prototxt"
SSD_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
IMG_SIZE = 96
FRAME_INTERVAL = 10
FACE_CONF_THRESH = 0.5

def load_ssd():
    return cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)

def detect_faces(net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()
    faces = []
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf > FACE_CONF_THRESH:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                fh, fw = face.shape[:2]
                side = max(fh, fw)
                ph = (side - fh) // 2
                pw = (side - fw) // 2
                face = cv2.copyMakeBorder(face, ph, side-fh-ph, pw, side-fw-pw, cv2.BORDER_REFLECT)
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                faces.append(face)
    return faces

def classify(interp, faces):
    inp = interp.get_input_details()
    out = interp.get_output_details()
    bs = inp[0]['shape'][0] if inp[0]['shape'][0] > 1 else 1
    probs = []
    for i in range(0, len(faces), bs):
        batch = np.array(faces[i:i+bs], dtype=np.float32) / 127.5 - 1.0
        interp.set_tensor(inp[0]['index'], batch)
        interp.invoke()
        ot = interp.get_tensor(out[0]['index'])
        p = ot[:, 0] if ot.ndim > 1 else ot.flatten()
        probs.extend(p.tolist())
    return np.array(probs)

def eval_video(net, interp, vpath):
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened(): return None
    idx = 0; probs = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % FRAME_INTERVAL == 0:
            faces = detect_faces(net, frame)
            if faces:
                p = classify(interp, faces)
                probs.extend(p.tolist())
        idx += 1
    cap.release()
    if not probs: return None
    pa = np.array(probs)
    preds = (pa > 0.5).astype(int)
    return {'n': len(pa), 'mean': float(np.mean(pa)), 'drowsy_pct': float(np.mean(preds))*100}

def get_label(vpath):
    name = os.path.basename(vpath).lower()
    parent = os.path.basename(os.path.dirname(vpath)).lower()
    for kw in ['drowsy', 'fatigue', 'sleepy']:
        if kw in name or kw in parent: return 1
    for kw in ['normal', 'alert', 'non']:
        if kw in name or kw in parent: return 0
    return -1

def main():
    if not os.path.exists(NTHU_PATH):
        print('ERROR: NTHU-DDD not found at', NTHU_PATH)
        print('Download videos and place in', NTHU_PATH)
        return
    print('Loading models...')
    ssd = load_ssd()
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for _ in range(5): interp.set_tensor(interp.get_input_details()[0]['index'], dummy); interp.invoke()
    vids = glob.glob(os.path.join(NTHU_PATH, '**', '*.avi'), recursive=True)
    vids += glob.glob(os.path.join(NTHU_PATH, '**', '*.mp4'), recursive=True)
    print('Found', len(vids), 'videos')
    if not vids: print('No videos found'); return
    results = []
    for vp in sorted(vids):
        label = get_label(vp)
        vname = os.path.relpath(vp, NTHU_PATH)
        print('  ', vname, 'label=', label)
        r = eval_video(ssd, interp, vp)
        if r:
            r['path'] = vname; r['label'] = label
            results.append(r)
            print('    frames:', r['n'], 'drowsy%:', round(r['drowsy_pct'],1))
    labs = [r['label'] for r in results if r['label'] >= 0]
    preds = [(1 if r['drowsy_pct'] > 50 else 0) for r in results if r['label'] >= 0]
    if labs:
        acc = sum(p==l for p,l in zip(preds,labs))/len(labs)*100
        print('Video-level Accuracy:', round(acc,1), '%')
        os.makedirs('results', exist_ok=True)
        with open('results/nthu_ddd_results.txt','w') as f:
            for r in results:
                f.write(r['path']+': label='+str(r['label'])+' drowsy%='+str(round(r['drowsy_pct'],1))+chr(10))
            f.write('Accuracy: '+str(round(acc,1))+'%'+chr(10))
        print('Saved to results/nthu_ddd_results.txt')

if __name__ == "__main__": main()
