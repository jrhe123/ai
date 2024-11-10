import gradio as gr

def button_action(button_name):
    return f"{button_name} was clicked"


with gr.Blocks() as demo:
    normal_button = gr.Button("Normal Button")

    primary_button = gr.Button("Primary Button", variant="primary")
    secondary_button = gr.Button("Secondary Button", variant="secondary")
    stop_button = gr.Button("Stop Button", variant="stop")

    button_1 = gr.Button(icon=r"C:\Users\mooc\Desktop\learn_gradio\2.Gradio常见输入\resources\三明治.png")
    button_2 = gr.Button("蛋挞",icon=r"C:\Users\mooc\Desktop\learn_gradio\2.Gradio常见输入\resources\蛋挞.png")
    button_3 = gr.Button("",icon=r"C:\Users\mooc\Desktop\learn_gradio\2.Gradio常见输入\resources\饭团.png")


    result_label = gr.Label()

    normal_button.click(fn=lambda: button_action("Normal Button"), outputs=result_label)
    primary_button.click(fn=lambda: button_action("Primary Button"), outputs=result_label)
    secondary_button.click(fn=lambda: button_action("Secondary Button"), outputs=result_label)
    stop_button.click(fn=lambda: button_action("Stop Button"), outputs=result_label)


demo.launch()
















