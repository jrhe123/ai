# 形参是函数定义时的参数，实参是函数调用时的参数

def create_model(layers, units):  # layers和units是形参
    print(f"Creating a model with {layers} layers and {units} units in each layer.")

# 调用函数
create_model(3, 128)  # 3和128是实参




# 位置参数的顺序很重要

def create_model(layers, units):
    print(f"Creating a model with {layers} layers and {units} units in each layer.")

# 调用函数
create_model(3, 128)
create_model(128, 3)



# 使用关键字参数调用函数

def create_model(layers, units):
    print(f"Creating a model with {layers} layers and {units} units in each layer.")

# 调用函数
create_model(units=128, layers=3)  # 使用关键字参数，顺序不重要
create_model(layers=3, units=128)



# 使用默认参数值

def create_model(layers=3, units=128):
    print(f"Creating a model with {layers} layers and {units} units in each layer.")

# 调用函数
create_model()  # 使用默认值




# 使用可变参数接收多个参数值

def add_layers(model, *layers):
    for layer in layers:
        print(f"Adding layer {layer} to model {model}.")

# 调用函数
add_layers("Model1", "conv", "relu", "softmax")




# 函数返回模型的信息

def create_model(layers, units):
    info = f"Creating a model with {layers} layers and {units} units in each layer."
    return info

# 调用函数
model_info = create_model(3, 128)
print(model_info)




# 全局变量

MODEL_NAME = "CNN"

def print_model_name():
    print(f"The model name is {MODEL_NAME}.")

# 调用函数
print_model_name()


# 局部变量

def create_model():
    model_name = "RNN"  # 局部变量
    print(f"Creating a model named {model_name}.")

# 调用函数
create_model()

print(model_name) # 此行代码会报错




# 使用lambda创建匿名函数

calculate_units = lambda layers: layers * 128

# 调用函数
units = calculate_units(3)
print(f"Total units: {units}")




def create_cnn(input_size, kernel=3, padding=0, stride=1):
    """
    创建一个卷积神经网络层

    参数：
    - input_size: 输入图像的尺寸，形式为 (channels, height, width)
    - kernel: 卷积核的大小，默认为 3
    - padding: 填充大小，默认为 0
    - stride: 步长，默认为 1

    返回：
    - output_size: 卷积操作后输出图像的尺寸，形式为 (channels, height, width)
    """

    # 从输入尺寸中获取通道数和图像大小
    channels, height, width = input_size

    # 计算卷积操作后的输出图像尺寸
    new_height = ((height + 2 * padding - kernel) // stride) + 1
    new_width = ((width + 2 * padding - kernel) // stride) + 1

    # 输出图像的尺寸
    output_size = (channels, new_height, new_width)

    return output_size

# 示例用法：
input_size = (3, 64, 64)  # 3通道，64x64的图像
output_size = create_cnn(input_size, kernel=3, padding=1, stride=2)
print(output_size)



# 函数实战：创建并配置模型

def create_model(layers=3, units=128, activation="relu"):
    print(f"Creating a model with {layers} layers, each with {units} units and {activation} activation.")

# 调用函数
create_model()
create_model(4, 256, "sigmoid")
