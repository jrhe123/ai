import gradio as gr

def greet_1(name):
    return f"Hello, {name} !"


with gr.Blocks() as demo:
    with gr.Row():
        input_1 = gr.Textbox(label="Your name",  placeholder="请输入名字")
        button_1 = gr.Button("Greet")
        output_1 = gr.Textbox(label="Greeting")

    with gr.Row():
        input_2 = gr.Textbox(label="Your name",  placeholder="请输入名字", scale=5)
        button_2 = gr.Button("Greet", scale=2)
        output_2 = gr.Textbox(label="Greeting", scale=1)

    button_1.click(fn=greet_1, inputs=input_1, outputs=output_1)
    button_2.click(fn=greet_1, inputs=input_2, outputs=output_2)

demo.launch()



















