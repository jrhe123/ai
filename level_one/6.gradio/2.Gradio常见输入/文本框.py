import gradio as gr

def demo():
    pass

demo = gr.Interface(fn=demo,
                    inputs=[gr.Text(label="账号",placeholder="请输入账号"),
                            gr.Text(label="邮箱",placeholder="请输入邮箱", type="email"),
                            gr.Text(label="密码",placeholder="请输入密码", type="password"),
                            gr.Textbox(label="Textbox",lines=3, max_lines=5, placeholder="请填写信息"),
                            gr.TextArea(label="TextArea",lines=3, max_lines=5, placeholder="请填写信息")],
                    outputs="text")

demo.launch()





























