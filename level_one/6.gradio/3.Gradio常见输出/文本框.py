import gradio as gr

def output():
    return "Hello World " * 20

demo = gr.Interface(fn=output,
                    inputs=None,
                    outputs=gr.Textbox(lines=3,
                                       placeholder="默认输出文字"))
demo.launch()










