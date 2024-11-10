import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        gr.Dropdown(["模型1", "模型2","模型3"],label="Stable Diffusion模型")
        gr.Dropdown(["模型1", "模型2", "模型3"], label="外挂VAE模型")
        gr.Slider(0,100,value=14,label="CLIP终止层数")

    with gr.Tab("文生图"):
        with gr.Row():
            with gr.Column(scale=5):
                gr.Textbox(label="提示词",lines=3, max_lines=5)
                gr.Textbox(label="反向提示词", lines=2, max_lines=5)
            with gr.Column(scale=2):
                gr.Button("生成", variant="primary")
                with gr.Row():
                    gr.Button("", icon=r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\arrow.png")
                    gr.Button("", icon=r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\file.png")
                    gr.Button("", icon=r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\delete.png")
                gr.Dropdown(["模型1","模型2"],label="Stable Diffusion模型")
        with gr.Tab("生成"):
            with gr.Row():
                with gr.Column():
                    gr.Slider(label="采样步数")
                    gr.Radio(["DPM++2M Karras","DPM++ SDE Karras","DPM++2M SDE Exponential",
                              "DPM++2M SDE Karras", "Euler a", "Eular", "LMS", "Heun", "DPM2",
                              "DPM2 a","DPM++ 2M","DPM++ SDE", "DPM++ 2M SDE","DPM++ 2M SDE Heun",
                              "DPM Fast", "DPM Adaptive", "DPM Adaptive Karras", "LMS Karras", "DPM2 Karras",
                              "DPM2 a Karras", "DPM++ 2M Karras", "DPM++ 2M SDE Karras", "DPM++ 2M SDE Karras Heun"],
                             label="采样方法"
                             )
                    with gr.Accordion("高分辨率修复"):
                        with gr.Row():
                            gr.Dropdown(["模型1","模型2"], label="放大算法")
                            gr.Slider(label="高分迭代步数")
                            gr.Slider(label="重绘幅度")
                        with gr.Row():
                            gr.Slider(label="放大背书")
                            gr.Slider(label="将宽度调整为")
                            gr.Slider(label="将高度调整为")

                    with gr.Accordion("Refiner"):
                        with gr.Row():
                            gr.Dropdown(["模型1", "模型2"], label="模型")
                            gr.Slider(label="切换时机")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Slider(label="宽度")
                            gr.Slider(label="高度")
                        with gr.Column(scale=1, min_width=1):
                            gr.Slider(label="总批次", min_width=1)
                            gr.Slider(label="单批数量", min_width=1)
                    gr.Slider(label="提示词引导系数")



                with gr.Column():
                    gr.Gallery([
                        r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\1.jpg",
                        r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\2.jpg",
                        r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\3.jpg",
                        r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\4.jpg",
                        r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\5.jpg",
                        r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\6.jpg",
                        r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\7.jpg",
                        r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\8.jpg",
                        r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\9.jpg",
                        r"C:\Users\mooc\Desktop\learn_gradio\demo\imgs\10.jpg",
                    ])
                    with gr.Row():
                        gr.Button("发送到重绘")
                        gr.Button("发送到后期处理")
                        gr.Button("下载")
                    gr.Textbox(label="图像信息",lines=5)






        with gr.Tab("嵌入式"):
            pass
        with gr.Tab("超网络"):
            pass
        with gr.Tab("模型"):
            pass
        with gr.Tab("LORA"):
            pass




    with gr.Tab("图生图"):
        pass
    with gr.Tab("后期处理"):
        pass
    with gr.Tab("PNG图片信息"):
        pass
    with gr.Tab("模型融合"):
        pass
    with gr.Tab("训练"):
        pass
    with gr.Tab("无边图像浏览"):
        pass




demo.launch()












