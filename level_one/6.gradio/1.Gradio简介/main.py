import gradio as gr

def greet(name):
    return "你好, " + name

demo = gr.Interface(fn=greet,
                    inputs="text",
                    outputs="text")

demo.launch(server_port=8000, share=True)
# server_port: 端口



















