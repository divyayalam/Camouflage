import gradio as gr
import numpy as np
from PIL import Image

#Loading the model
from pathlib import Path
import keras

CLASS_NAMES = ["bird", "cat", "deer", "dog", "frog", "horse"]

MODEL_PATH = Path(__file__).resolve().parent / "animal_cnn_final.keras"
model = keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image):

    #Ensuring the image is in RGB format
    image = image.convert("RGB")

    #Resizing the image to 32x32 pixels
    image = image.resize((32, 32), Image.Resampling.BILINEAR)

    #Converting the pixels to numericals and normalizing the image
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    #Creating a batch containing one image
    batch = np.expand_dims(image_array, axis=0)

    return batch

def predict_animal(image):
    if image is None:
        raise gr.Error("Upload an image first.")

    batch = preprocess_image(image)
    probabilities = model(batch, training=False).numpy()[0]

    return {
        name: float(probabilities[i])
        for i, name in enumerate(CLASS_NAMES)
    }

#Updated Gradio layout
with gr.Blocks() as demo:
    gr.Markdown(
        "# CAMouflage\n"
        "Upload an image to classify it into one of six animal categories."
    )

    with gr.Row():
        image = gr.Image(type="pil", label="Upload an image")
        result = gr.Label(num_top_classes=6, label="Prediction")

    button = gr.Button("Predict")
    button.click(fn=predict_animal, inputs=image, outputs=result)


if __name__ == "__main__":
    demo.launch()
