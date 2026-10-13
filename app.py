import gradio as gr


def preview(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    return image


with gr.Blocks() as demo:
    gr.Markdown("# CAMouflage\nUpload an image to check the dashboard.")

    with gr.Row():
        image = gr.Image(type="pil", label="Upload an image")
        result = gr.Image(label="Preview", interactive=False)

    button = gr.Button("Preview")
    button.click(fn=preview, inputs=image, outputs=result)


if __name__ == "__main__":
    demo.launch()