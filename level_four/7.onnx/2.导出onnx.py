import torch
from torch import nn
from torchvision.models import resnet18
# pip install onnx
# pip install onnxruntime

"""
模型可视化
https://netron.app/
"""

if __name__ == '__main__':

    # 指定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # 指定模型
    model = resnet18(pretrained=False)

    num_ftrs = model.fc.in_features    # 获取全连接层的输入
    model.fc = nn.Linear(num_ftrs, 10)  # 全连接层改为不同的输出

    # 模型加载权重
    torch_data = torch.load('logs_resnet18_pretrain/best.pth',
                            map_location=torch.device(device))

    model.load_state_dict(torch_data)
    model.to(device)

    # 创建一个示例输入
    dummy_input = torch.randn(1,3,224,224, device=device)
    # 指定输出文件路径
    onnx_file_path = "logs_resnet18_pretrain/model.onnx"

    # 导出onnx
    torch.onnx.export(model, dummy_input, onnx_file_path,
                      verbose=True,  # 屏幕中打印日志信息
                      input_names=['input'],
                      output_names=['output'])

    print("Model Exported Success")