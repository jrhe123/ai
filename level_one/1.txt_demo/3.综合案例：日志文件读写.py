# 写一段代码，模拟生成accuracy逐步上升、loss逐步下降的训练日志，并将日志信息记录到 training_log.txt中

import random

epoch = 100
accuracy = 0.5
loss = 0.9

with open('training_log.txt', 'w') as f:
    f.write('Epoch\tAccuracy\tLoss\n')

    for epoch_i in range(1, epoch+1):
        accuracy += random.uniform(0, 0.005)
        loss -= random.uniform(0, 0.005)

        accuracy = min(1, accuracy)
        loss = max(0, loss)

        f.write(f'{epoch_i}\t{accuracy:.3f}\t{loss:.3f}\n')

        print(f'Epoch:{epoch_i}, Accuracy:{accuracy}, Loss:{loss}')


















