'''
    按着路径，导入单张图片做预测
'''
from torchvision.models import resnet18
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import cv2 as cv
import os
import numpy as np
'''
    加载图片与格式转化
'''

# 图片标准化
transform_BZ = transforms.Normalize(
    mean = [0.5062653, 0.46558657, 0.37899864],  # 取决于数据集
    std = [0.22566116, 0.20558165, 0.21950442]
)

img_size = 224
val_tf = transforms.Compose([##简单把图片压缩了变成Tensor模式
                transforms.ToPILImage(),  # 将numpy数组转换为PIL图像
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transform_BZ#标准化操作
            ])


def cv_imread(file_path):
    cv_img = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)
    return cv_img


def predict(img_path):
    '''
        获取标签名字
    '''
    # # 增加类别标签
    # dir_names = []
    # for root, dirs, files in os.walk("dataset"):
    #     if dirs:
    #         dir_names = dirs
    # 将输出保存到exel中，方便后续分析
    label_names = ['ChuJu', 'GuiHua', 'HeHua', 'MeiGui', 'PuGongYing',
                   'QianNiuHua', 'TaoHua', 'XiangRiKui', 'YuJinXiang',
                   'ZiLuoLan']


    # 指定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    """
        加载模型
    """
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features    # 获取全连接层的输入
    model.fc = nn.Linear(num_ftrs, 10)  # 全连接层改为不同的输出
    torch_data = torch.load('best.pth',
                            map_location=torch.device(device))
    model.load_state_dict(torch_data)
    model.to(device)

    '''
        读取图片
    '''
    img = cv_imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_tensor = val_tf(img)

    # 增加batch_size维度
    img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(),
                          requires_grad=False).to(device)

    '''
        数据输入与模型输出转换
    '''
    model.eval()
    with torch.no_grad():
        output_tensor = model(img_tensor)

        # 将输出通过softmax变为概率值
        output = torch.softmax(output_tensor, dim=1)

        # 输出可能性最大的那位
        pred_value, pred_index = torch.max(output, 1)

        # 将数据从cuda转回cpu
        if torch.cuda.is_available() == False:
            pred_value = pred_value.detach().cpu().numpy()
            pred_index = pred_index.detach().cpu().numpy()

        result = "预测类别为： "+str(label_names[pred_index[0]])+ " 可能性为: "+str(pred_value[0].item() * 100)[:5]+ "%"
        return result

if __name__ == "__main__":
    img_path = r'D:/慕课网上课计划/慕课网课程/pytorch/code/训练函数与测试函数/PyTorch模型训练_训练自己的数据集/dataset/MeiGui/12240303_80d87f77a3_n.jpg'
    result = predict(img_path)
    print(result)


