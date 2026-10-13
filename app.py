import gradio as gr

def preview_image(image):
    if image is None:
        raise gr.Error("No image is provided. Please upload an image.")
    return image

preview = gr.Interface(
    fn = preview_image,
    inputs = gr.Image(),
    outputs = gr.Image(),
    api_name = "image preview"
)

preview.launch()