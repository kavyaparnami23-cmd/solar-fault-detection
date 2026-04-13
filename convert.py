import tensorflow as tf

# Load the model
try:
    model = tf.keras.models.load_model('final_model .h5')
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted successfully!")
except Exception as e:
    print(f"Error: {e}")
