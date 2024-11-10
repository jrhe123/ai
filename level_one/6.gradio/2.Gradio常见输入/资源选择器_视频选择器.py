import gradio as gr

def data(file):
    return str(file)

demo = gr.Interface(fn=data,
                    inputs=gr.Video(source="upload", label="Video"),
                    outputs="text") # ffmpeg

demo.launch()

















