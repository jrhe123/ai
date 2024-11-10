import gradio as gr

def greet(number):
    return "!" * number

demo = gr.Interface(fn=greet,
                    inputs=gr.Slider(minimum=10,
                                     maximum=50,
                                     value=20,
                                     step=1,
                                     label="感叹号数量"),
                    outputs="text")

demo.launch()















