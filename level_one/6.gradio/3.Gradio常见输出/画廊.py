import gradio as gr

def process():
    imgpaths = [
        r"C:\Users\mooc\Desktop\learn_gradio\3.Gradio常见输出\imgs\1.jpg",
        r"C:\Users\mooc\Desktop\learn_gradio\3.Gradio常见输出\imgs\2.jpg",
        r"C:\Users\mooc\Desktop\learn_gradio\3.Gradio常见输出\imgs\3.jpg",
        r"C:\Users\mooc\Desktop\learn_gradio\3.Gradio常见输出\imgs\4.jpg",
        r"C:\Users\mooc\Desktop\learn_gradio\3.Gradio常见输出\imgs\5.jpg",
        r"C:\Users\mooc\Desktop\learn_gradio\3.Gradio常见输出\imgs\6.jpg",
        r"C:\Users\mooc\Desktop\learn_gradio\3.Gradio常见输出\imgs\7.jpg",
        r"C:\Users\mooc\Desktop\learn_gradio\3.Gradio常见输出\imgs\8.jpg",
        r"C:\Users\mooc\Desktop\learn_gradio\3.Gradio常见输出\imgs\9.jpg",
        r"C:\Users\mooc\Desktop\learn_gradio\3.Gradio常见输出\imgs\10.jpg",
    ]

    # results = imgpaths
    results = [(img_i, f"cat {i+1}") for i, img_i in enumerate(imgpaths)]
    return results

demo = gr.Interface(fn=process, inputs=None, outputs=gr.Gallery(columns=4))
demo.launch()