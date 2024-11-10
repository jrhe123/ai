x_1 = 40.0
x_2 = 80.9
expected_output = 60.0
learning_rate = 1e-5

# 定义参数初始化函数
def parm_init():
    # 初始化
    # 定义权重
    w_1_11 = 0.5
    w_1_12 = 0.5
    w_1_13 = 0.5
    w_1_21 = 0.5
    w_1_22 = 0.5
    w_1_23 = 0.5

    w_2_11 = 1.0
    w_2_21 = 1.0
    w_2_31 = 1.0

    layer_1_list = [w_1_11, w_1_12, w_1_13, w_1_21, w_1_22, w_1_23]
    layer_2_list = [w_2_11, w_2_21, w_2_31]
    return layer_1_list, layer_2_list

# 定义前向传播函数
def foward_porpagation(layer_1_list, layer_2_list):
    w_1_11, w_1_12, w_1_13, w_1_21, w_1_22, w_1_23 = layer_1_list
    w_2_11, w_2_21, w_2_31 = layer_2_list

    # 前向传播
    z_1 = x_1 * w_1_11 + x_2 * w_1_21
    z_2 = x_1 * w_1_12 + x_2 * w_1_22
    z_3 = x_1 * w_1_13 + x_2 * w_1_23
    y_pred = z_1 * w_2_11 + z_2 * w_2_21 + z_3 * w_2_31
    return y_pred

# 定义损失计算函数
def compute_loss(y_true, y_pred):
    # 计算损失值（L2 损失）
    loss = 0.5 * (y_true - y_pred) ** 2
    return loss

# 定义反向传播函数
def backward_propagation(layer_1_list, layer_2_list, learning_rate):
    w_1_11, w_1_12, w_1_13, w_1_21, w_1_22, w_1_23 = layer_1_list
    w_2_11, w_2_21, w_2_31 = layer_2_list

    z_1 = x_1 * w_1_11 + x_2 * w_1_21
    z_2 = x_1 * w_1_12 + x_2 * w_1_22
    z_3 = x_1 * w_1_13 + x_2 * w_1_23

    # 计算输出层关于损失函数的梯度
    d_loss_predictied_output = -(expected_output - y_pred)

    # 计算权重关于损失函数的梯度
    d_loss_w_2_11 = d_loss_predictied_output * z_1
    d_loss_w_2_21 = d_loss_predictied_output * z_2
    d_loss_w_2_31 = d_loss_predictied_output * z_3
    d_loss_w_1_11 = d_loss_predictied_output * w_2_11 * x_1
    d_loss_w_1_21 = d_loss_predictied_output * w_2_11 * x_2
    d_loss_w_1_12 = d_loss_predictied_output * w_2_21 * x_1
    d_loss_w_1_22 = d_loss_predictied_output * w_2_21 * x_2
    d_loss_w_1_13 = d_loss_predictied_output * w_2_31 * x_1
    d_loss_w_1_23 = d_loss_predictied_output * w_2_31 * x_2

    # 更新权重
    w_2_11 -= learning_rate * d_loss_w_2_11
    w_2_21 -= learning_rate * d_loss_w_2_21
    w_2_31 -= learning_rate * d_loss_w_2_31
    w_1_11 -= learning_rate * d_loss_w_1_11
    w_1_12 -= learning_rate * d_loss_w_1_12
    w_1_13 -= learning_rate * d_loss_w_1_13
    w_1_21 -= learning_rate * d_loss_w_1_21
    w_1_22 -= learning_rate * d_loss_w_1_22
    w_1_23 -= learning_rate * d_loss_w_1_23

    layer_1_list = [w_1_11, w_1_12, w_1_13, w_1_21, w_1_22, w_1_23]
    layer_2_list = [w_2_11, w_2_21, w_2_31]
    return layer_1_list, layer_2_list


if __name__ == "__main__":
    # 设置训练的迭代次数
    epoch = 100

    # 初始化定义权重
    layer_1_list, layer_2_list = parm_init()

    # 开始训练循环
    for i in range(epoch):
        # 前向传播，计算预测值
        y_pred = foward_porpagation(layer_1_list, layer_2_list)
        print(f"前向传播预测值为: {y_pred}")

        # 计算当前损失值
        loss = compute_loss(expected_output, y_pred)
        print(f"当前Loss值为: {loss}")

        # 反向传播，更新权重
        layer_1_list, layer_2_list = backward_propagation(layer_1_list, layer_2_list, learning_rate)