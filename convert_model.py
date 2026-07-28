import tensorflow as tf

print("Loading Keras model...")
try:
    model = tf.keras.models.load_model("dermasense_skin_model.keras")
    print("[SUCCESS] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

print("Converting to TFLite...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = "dermasense_skin_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[SUCCESS] Saved TFLite model to {tflite_path}")
except Exception as e:
    print(f"[ERROR] Conversion failed: {e}")
    exit(1)
