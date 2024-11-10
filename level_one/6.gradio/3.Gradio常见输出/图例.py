# matplotlib 的图片出现在 gradio 中
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt


def fig_output():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    return plt

demo = gr.Interface(fn=fig_output, inputs=None, outputs=gr.Plot())
demo.launch()