import gradio as gr

def greet(number1, number2):
    return str(number1 + number2)

demo = gr.Interface(fn=greet,
                    inputs=["number",
                            gr.Number()],
                    outputs="text")

demo.launch()




