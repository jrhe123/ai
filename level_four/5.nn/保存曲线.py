from torch.utils.tensorboard import SummaryWriter
import random

# 数据都保存在 "data" 文件夹中
writer = SummaryWriter(log_dir="data")

offset = random.random() / 2
epochs = 10

for epoch_i in range(2, epochs):
    acc = 1 - 2 ** -epoch_i - random.random() / epoch_i - offset
    loss = 2 ** -epoch_i - random.random() / epoch_i - offset
    
    # 记录指标
    writer.add_scalar(tag='Accuracy/acc', scalar_value=acc, global_step=epoch_i)
    writer.add_scalar(tag='Accuracy/loss', scalar_value=loss, global_step=epoch_i)

writer.close()

# """
# tensorboard --logdir .
# """
