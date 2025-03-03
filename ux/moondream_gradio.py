import gradio as gr
import moondream as md
import os

moondream_api_key = os.getenv("MOONDREAM_API_KEY")
model = md.vl(api_key=moondream_api_key)

def answer_question(img, prompt):
    buffer = ""
    for chunk in model.query(img, prompt, stream=True)["answer"]:
        buffer += chunk
        yield buffer

def process_answer(img, answer):
    return gr.update(visible=False, value=None)

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🌔 moondream2
        A tiny vision language model. Check out other capabilities (object detection, pointing etc.) in the [Moondream Playground](https://moondream.ai/playground).
        """
    )
    with gr.Row():
        prompt = gr.Textbox(label="Input", value="Describe this image.", scale=4)
        submit = gr.Button("Submit")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image")
        with gr.Column():
            output = gr.Markdown(label="Response")
            ann = gr.Image(visible=False, label="Annotated Image")

    submit.click(answer_question, [img, prompt], output)
    prompt.submit(answer_question, [img, prompt], output)
    output.change(process_answer, [img, output], ann, show_progress=False)

demo.queue().launch()
