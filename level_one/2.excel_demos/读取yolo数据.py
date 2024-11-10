import pandas as pd
import matplotlib.pyplot as plt


file_1_loc = 'resources/yolov5l.csv'
file_2_loc = 'resources/yolov5m.csv'
file_3_loc = 'resources/yolov5s.csv'

file_1 = pd.read_csv(file_1_loc)
file_2 = pd.read_csv(file_2_loc)
file_3 = pd.read_csv(file_3_loc)

file_1_train_loss = file_1['      train/box_loss']
print(file_1_train_loss)

x_list = [i for i in range(len(file_1_train_loss))]

plt.plot(x_list, file_1_train_loss)
plt.show()
