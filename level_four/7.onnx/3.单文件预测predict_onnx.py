'''
    按着路径，导入单张图片做预测
'''
import onnxruntime as ort # pip install onnxruntime onnx
import numpy as np
import torchvision.transforms as transforms
import cv2 as cv
import os


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


def cv_imread(file_path):
    cv_img = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)
    return cv_img


def predict(img_path):
    '''
        获取标签名字
    '''
    # dir_names = []
    # for root, dirs, files in os.walk("dataset"):
    #     if dirs:
    #         dir_names = dirs
    # label_names = dir_names

    label_names = ['ChuJu', 'GuiHua', 'HeHua', 'MeiGui', 'PuGongYing',
                   'QianNiuHua', 'TaoHua', 'XiangRiKui', 'YuJinXiang',
                   'ZiLuoLan']

    '''
        加载图片与格式转化
    '''

    # 图片标准化
    transform_BZ = transforms.Normalize(
        mean=[0.5062653, 0.46558657, 0.37899864],  # 取决于数据集
        std=[0.22566116, 0.20558165, 0.21950442]
        )

    img_size = 224
    val_tf = transforms.Compose([   # 简单把图片压缩了变成Tensor模式
                transforms.ToPILImage(),  # 将numpy数组转换为PIL图像
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transform_BZ # 标准化操作
            ])

    # 读取图片
    img = cv_imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_tensor = val_tf(img)

    # 将图片转换为ONNX运行时所需的格式
    img_numpy = img_tensor.numpy()
    img_numpy = np.expand_dims(img_numpy, axis=0)  # 增加batch_size维度

    # 加载ONNX模型
    onnx_model_path = r'model.onnx'  # 替换为ONNX模型的路径
    ort_session = ort.InferenceSession(onnx_model_path)

    # 运行ONNX模型
    outputs = ort_session.run(None, {'input': img_numpy})
    output = outputs[0]

    # 应用softmax
    probabilities = softmax(output)

    # 获得预测结果
    pred_index = np.argmax(probabilities, axis=1)
    pred_value = probabilities[0][pred_index[0]]

    result = "预测类别为： " + str(label_names[pred_index[0]]) + " 可能性为: " + str(pred_value * 100)[:5] + "%"
    return result

if __name__ == "__main__":
    img_path = r'D:/慕课网上课计划/慕课网课程/pytorch/code/训练函数与测试函数/PyTorch模型训练_训练自己的数据集/dataset/MeiGui/12240303_80d87f77a3_n.jpg'
    result = predict(img_path)
    print(result)


