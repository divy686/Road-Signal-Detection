import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("road_sign_cnn_model.h5")

# Define the class names (replace with your actual class names)
class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
    'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road',
    'Slippery road', 'Road narrows on the right', 'Road work',
    'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]

# Image prediction function
def predict_image(img):
    img = img.resize((30, 30))  # Resize to match training input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 30, 30, 3)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]
    return class_name

# Gradio UI
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an image", height=400, width=400),
    outputs=gr.Textbox(label="Predicted Sign"),
    title="Road Sign Detection",
    description="Upload an image of a road sign to detect its type."
)

interface.launch()
