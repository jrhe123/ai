# 勾选框、单选框、多选框、下拉框
import gradio as gr

def function(choose1,choose2, choose3, choose4):
    return str(choose1)+str(choose2)+str(choose3) + str(choose4)
    # pass
demo = gr.Interface(fn=function,
                    inputs=[
                        gr.Checkbox(label="勾选框"), # 勾选框
                        gr.Radio(["选项A","选项B","选项C"],label="单选框"), # 单选框
                        gr.Checkboxgroup(["选项A","选项B","选项C"],label="多选框"), # 多选框
                        gr.Dropdown(["选项A","选项B","选项C"],label="下拉框")  # 下拉框
                    ],
                    outputs="text")
demo.launch()























