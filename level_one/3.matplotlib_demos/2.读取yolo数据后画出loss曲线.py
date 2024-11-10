import pandas as pd
import matplotlib.pyplot as plt

data_loc = r'resources/yolov5s.csv'

data = pd.read_csv(data_loc, index_col=0)

train_bbox_loss = data['      train/box_loss']

x_list = [i for i in range(len(train_bbox_loss))]
plt.plot(x_list, train_bbox_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('YOLOv5s')
plt.show()














