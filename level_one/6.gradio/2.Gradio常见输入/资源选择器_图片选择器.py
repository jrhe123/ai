import gradio as gr
import cv2 as cv # pip install opencv-python

def turn_gray(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

demo = gr.Interface(fn=turn_gray,
                    inputs=gr.Image(source="webcam"),
                    outputs="image")  # gr.Image()

demo.launch()


















