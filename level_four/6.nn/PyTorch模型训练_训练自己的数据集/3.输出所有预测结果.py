'''
    1.单幅图片验证
    2.多幅图片验证
'''
import torch
from torch.utils.data import DataLoader
from MYDataset import MyDataset
from MyModel import SimpleCNN
import pandas as pd
from tqdm import tqdm
import os


def eval(dataloader, model):
    pred_list = []
    model.eval()
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in tqdm(dataloader, desc="Model is predicting, please wait"):
            # 将数据转到GPU
            X = X.to(device)
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)

            pred_softmax = torch.softmax(pred,1).cpu().numpy()
            pred_list.append(pred_softmax.tolist()[0])

        return pred_list


if __name__ == "__main__":
    '''
        加载预训练模型
    '''
    # 1. 导入模型结构
    device = "mps" if torch.cuda.is_available() else "cpu"

    # 2. 加载模型参数
    model = SimpleCNN()
    model_state_loc = r"logs/best.pth"

    torch_data = torch.load(model_state_loc,  map_location=torch.device(device))
    model.load_state_dict(torch_data)
    model = model.to(device)

    '''
       加载需要预测的图片
    '''
    valid_data = MyDataset("test.txt", train_flag=False)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=4,
                                 pin_memory=True, batch_size=1)


    '''
      获取结果
    '''
    # 获取模型输出
    pred =  eval(test_dataloader, model)


    # 获取标签名字 e.g., ChuJu, GuiHua, etc.
    dir_names = []
    for root,dirs,files in os.walk("dataset"):
        if dirs:
            dir_names = dirs
    # 将输出保存到exel中，方便后续分析
    label_names = dir_names     # 可以把标签写在这里
    print(label_names)


    df_pred = pd.DataFrame(data=pred, columns=label_names)
    df_pred.to_csv('pred_result.csv', encoding='gbk', index=False)
    print("Done!")

