import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
import os

# --- 1. GLOBAL CONFIGURATION ---
IMG_SIZE = (96, 96) 
BATCH_SIZE = 32 
DATASET_PATH = 'dataset/train_cropped' 
AUTOTUNE = tf.data.AUTOTUNE

# Enable XLA for 1.5x - 2x speedup on compatible hardware
tf.config.optimizer.set_jit(True)

def create_dataset(subset):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset=subset,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    
    # Preprocessing: Native to MobileNetV3 (Scales pixels to [-1, 1])
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    
    # AGGRESSIVE AUGMENTATION: To break the plateau
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),     # 18-degree variance
        layers.RandomTranslation(0.1, 0.1), # Vital for crop-jitter robustness
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ])

    if subset == 'training':
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    
    return ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)

# --- 2. DATA LOADERS ---
train_ds = create_dataset('training')
val_ds = create_dataset('validation')

# --- 3. ARCHITECTURE: MOBILENETV3-LARGE ---
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(96, 96, 3), include_top=False, alpha=1.0, weights='imagenet'
)
base_model.trainable = False # Start frozen

inputs = layers.Input(shape=(96, 96, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

# DEEP HEAD: For complex eyelid geometry
x = layers.Dense(1024, activation='swish', kernel_regularizer=regularizers.l2(5e-4))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(2e-4))(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=inputs, outputs=outputs)

# --- 4. TRAINING PHASES ---
if __name__ == "__main__":
    # LOSS: Focal Loss exponentially penalizes the 4.2% of images you're currently missing
    loss_fn = keras.losses.BinaryFocalCrossentropy(gamma=2.5, label_smoothing=0.05)
    
    # PHASE 1: Warmup (Top Layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    checkpointer = callbacks.ModelCheckpoint('drowsiness.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)

    print("🚀 PHASE 1: Aligning Neural Head...")
    model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[checkpointer, early_stop])

    # PHASE 2: Deep Fine-Tuning
    base_model.trainable = True
    for layer in base_model.layers[:-100]: # Unfreeze a massive section for 99.5%
        layer.trainable = False
    
    # Keep BN frozen to preserve statistics
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # Cosine Decay for smooth convergence to the absolute minimum
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-5,
        decay_steps=80 * len(train_ds),
        alpha=0.01
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['accuracy']
    )

    print("🎯 PHASE 2: Surgical Fine-Tuning...")
    model.fit(train_ds, initial_epoch=20, epochs=100, validation_data=val_ds, callbacks=[checkpointer, early_stop])

    # --- 5. BEAST EXPORT (TFLite for Hugging Face) ---
    print("📦 PHASE 3: Converting to TFLite for Deployment...")
    model.load_weights('drowsiness.keras')
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Quantization for speed
    tflite_model = converter.convert()
    
    with open('drowsiness.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✅ SUCCESS: Model saved as 'drowsiness.tflite'")