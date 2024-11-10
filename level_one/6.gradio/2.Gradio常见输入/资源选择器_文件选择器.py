import gradio as gr

def data(file):
    filename_list = []
    for file_i in file:
        filename_list.append(file_i.name)
    return str(filename_list)


demo = gr.Interface(fn=data,
                    inputs=gr.File(label="文件选择框",
                                   file_count="directory",
                                   file_types=None,
                                   type="file"),
                    outputs="text")

demo.launch()















