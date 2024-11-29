import wandb
import random
import datetime

run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(
    project='wandb_demo',
    name=f"run-{run_time}",
    config={
        "learning_rate": 0.1,
        "model": "CNN",
        "dataset": "MNIST",
        "epochs": 10,
    }
)

offset = random.random() / 2
epochs = 10
for epoch_i in range(2, epochs):
    acc = 1 - 2 ** -epoch_i - random.random() / epoch_i - offset
    loss = 2 ** -epoch_i - random.random() / epoch_i - offset
    # Log metrics
    wandb.log({"acc": acc, "loss": loss})

wandb.finish()

# https://wandb.ai/