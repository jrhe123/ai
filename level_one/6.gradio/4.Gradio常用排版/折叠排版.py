import gradio as gr

with gr.Blocks() as demo:
    with gr.Accordion("折叠排版"):
        with gr.Row():
            with gr.Column(scale=5):
                text1 = gr.Textbox(label="提示词", placeholder="请输入提示词",lines=5)
                text2 = gr.Textbox(label="反向提示词", placeholder="请输入反向提示词",lines=2)
            with gr.Column(scale=1, min_width=1):
                button1 = gr.Button("",icon=r"C:\Users\mooc\Desktop\learn_gradio\4.Gradio常用排版\resources\长箭头.png", min_width=1)
                button2 = gr.Button("",icon=r"C:\Users\mooc\Desktop\learn_gradio\4.Gradio常用排版\resources\save.png", min_width=1)
                button3 = gr.Button("",icon=r"C:\Users\mooc\Desktop\learn_gradio\4.Gradio常用排版\resources\File.png", min_width=1)
                button4 = gr.Button("",icon=r"C:\Users\mooc\Desktop\learn_gradio\4.Gradio常用排版\resources\删除.png", min_width=1)
            with gr.Column(scale=2):
                button5 = gr.Button("生成",scale=2, min_width=2, variant="primary")
                with gr.Row():
                    gr.Dropdown(["1","2","3"], label="Style 1", scale=2, min_width=2)
                    gr.Dropdown(["1", "2", "3"], label="Style 2", scale=2, min_width=2)

demo.launch()







































