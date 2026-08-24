import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf, numpy as np, cv2, glob, random, time
from tensorflow import keras
from tensorflow.keras import layers

IMG=96; BATCH=32; EPOCHS=5
DATA="external_dataset/combined_cropped"

active=sorted(glob.glob("%s/active/*.jpg" % DATA))[:2000]
fatigue=sorted(glob.glob("%s/fatigue/*.jpg" % DATA))[:2000]
files=[(f,0) for f in active]+[(f,1) for f in fatigue]
random.seed(42); random.shuffle(files)
val=files[:500]; train=files[500:]
print("Train:%d Val:%d" % (len(train), len(val)))

def load(fl):
  I,L=[],[]
  for fp,l in fl:
    img=cv2.imread(fp)
    if img is None: continue
    I.append(cv2.resize(img,(IMG,IMG)).astype(np.float32))
    L.append(l)
  return np.array(I),np.array(L)

print("Loading...")
t=time.time()
Xt,yt=load(train)
Xv,yv=load(val)
print("Loaded in %.0fs: train%s val%s" % (time.time()-t, Xt.shape, Xv.shape))

base=keras.applications.MobileNetV3Large(input_shape=(IMG,IMG,3),include_top=False,weights="imagenet")
base.trainable=True
for l in base.layers[:-15]: l.trainable=False
m=keras.Sequential([base,layers.GlobalAveragePooling2D(),layers.Dropout(0.3),layers.Dense(1,activation="sigmoid")])
m.compile(optimizer=keras.optimizers.Adam(1e-4),loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0),metrics=["accuracy"])

print("Training...")
m.fit(Xt,yt,validation_data=(Xv,yv),epochs=EPOCHS,batch_size=BATCH,
  callbacks=[keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True,monitor="val_accuracy")])

m.save("drowsiness_pipeline_matched.keras")
print("Saved Keras")

conv=tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations=[tf.lite.Optimize.DEFAULT]
tl=conv.convert()
open("drowsiness_pipeline_matched.tflite","wb").write(tl)
print("Saved TFLite: %.2f MB" % (len(tl)/1024/1024))

from tensorflow.lite.python.interpreter import Interpreter
ip=Interpreter(model_path="drowsiness_pipeline_matched.tflite")
ip.allocate_tensors()
ii=ip.get_input_details()[0]["index"]; oi=ip.get_output_details()[0]["index"]
c=tp=fp=tn=fn=0
for x,l in zip(Xv,yv):
  ip.set_tensor(ii,np.expand_dims(x,0)); ip.invoke()
  p=1 if float(ip.get_tensor(oi)[0][0])>0.5 else 0
  if p==l: c+=1
  if p==1 and l==1: tp+=1
  elif p==1 and l==0: fp+=1
  elif p==0 and l==0: tn+=1
  else: fn+=1
tt=tp+fp+tn+fn
print("TFLite: %d/%d = %.2f%% P=%.4f R=%.4f F1=%.4f Sp=%.4f" % (c,tt,c/tt*100,tp/max(tp+fp,1),tp/max(tp+fn,1),2*tp/max(2*tp+fp+fn,1),tn/max(tn+fp,1)))
print("TP=%d FP=%d TN=%d FN=%d" % (tp,fp,tn,fn))