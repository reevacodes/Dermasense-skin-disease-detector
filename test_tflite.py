import numpy as np
from PIL import Image
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Load TFLite model
tflite_path = "dermasense_skin_model.tflite"
print(f"Loading TFLite model from '{tflite_path}'...")
if not os.path.exists(tflite_path):
    print(f"[ERROR] {tflite_path} does not exist yet. Please wait for model conversion to finish.")
    exit(1)

try:
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    print("[SUCCESS] TFLite model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load TFLite model: {e}")
    exit(1)

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Load and preprocess a test image
test_img_path = "data/test/acne/R (1).jpg"
print(f"Loading test image: {test_img_path}")
if not os.path.exists(test_img_path):
    print("Test image path not found, searching data/test/...")
    found_img = None
    for root, dirs, files in os.walk("data/test"):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                found_img = os.path.join(root, file)
                break
        if found_img:
            break
    if found_img:
        test_img_path = found_img
        print(f"Using found test image: {test_img_path}")
    else:
        print("[ERROR] No test images found in 'data/test' directory.")
        exit(1)

try:
    image = Image.open(test_img_path).convert("RGB")
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    image = image.resize((width, height))
    
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("[SUCCESS] Prediction probabilities:", output_data)
    
    class_names = ["Acne", "Melanoma", "Psoriasis"]
    pred_class_idx = np.argmax(output_data[0])
    print(f"Predicted class: {class_names[pred_class_idx]} with confidence {output_data[0][pred_class_idx]*100:.2f}%")
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
