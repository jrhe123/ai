import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

class MyDataset(Dataset):
    def __init__(self, txt_path, img_size=224, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        # 图片标准化
        transform_BZ = transforms.Normalize(
            mean=[0.5062653, 0.46558657, 0.37899864],  # 取决于数据集
            std=[0.22566116, 0.20558165, 0.21950442]
        )

        self.train_tf = transforms.Compose([
                transforms.ToPILImage(),  # 将numpy数组转换为PIL图像
                transforms.Resize((img_size,img_size)),#将图片压缩成224*224的大小
                transforms.RandomHorizontalFlip(),#对图片进行随机的水平翻转
                transforms.RandomVerticalFlip(),#随机的垂直翻转
                transforms.ToTensor(),#把图片改为Tensor格式
                transform_BZ#图片标准化的步骤
            ])

        self.val_tf = transforms.Compose([##简单把图片压缩了变成Tensor模式
                transforms.ToPILImage(),  # 将numpy数组转换为PIL图像
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transform_BZ#标准化操作
            ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info # 返回图片信息

    def __getitem__(self, index): #返回真正想返回的东西
        img_path, label = self.imgs_info[index]
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将图片从BGR转换为RGB格式
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs_info)


if __name__ == "__main__":
    my_dataset_train = MyDataset("train.txt",train_flag=True)
    my_dataloader_train = DataLoader(my_dataset_train, batch_size=10, shuffle=True)
    # 尝试读取训练集数据
    print("读取训练集数据")
    for x, y in my_dataloader_train:
        print(x.type(), x.shape, y)


    my_dataset_test = MyDataset("test.txt",train_flag=False)
    my_dataloader_test = DataLoader(my_dataset_test, batch_size=10, shuffle=False)
    # 尝试读取训练集数据
    print("读取测试集数据")
    for x, y in my_dataloader_test:
        print(x.shape, y)


