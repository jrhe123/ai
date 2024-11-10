class Cake:
    pass

cake_1 = Cake() # 用 Cake 模板制作成 cake_1
cake_2 = Cake() # 用 Cake 模板制作成 cake_2
print(cake_1)    # 输出cake_1 内存地址
print(cake_2)   # cake_2内存地址



class Car:
    def __init__(self):
        self.color = 'red'
        self.band = 'BYD'
        self.model = 'A1'
        print("正在定义一辆车")


car_1 = Car() # 实例化
print(car_1.color)
print(car_1.band)
print(car_1.model)


class Car:
    def __init__(self):
        self.color="red"
        self.band='BYD'
        self.model = 'A1'
        print("正在定义一辆车")
    def start(self):
        print("启动")
    def forward(self):
        print("向前")
    def stop(self):
        print("停下")

car_1 = Car()
print(car_1.color)
print(car_1.band)
print(car_1.model)

car_1.start()
car_1.forward()
car_1.stop()




class Cake:
    pass

cake_1 = Cake() # 用 Cake 模板制作成 cake_1
cake_2 = Cake() # 用 Cake 模板制作成 cake_2
print(cake_1)    # 输出cake_1 内存地址
print(cake_2)   # cake_2内存地址


class NeuralNetwork:
    def __init__(self, weights, bias):
        print("我是神经网络")
        self.w = weights
        self.b = bias

    def forward(self, x):
        print(f"已经接收到输入{x}")
        print("我在前向传播")
        y = self.w * x + self.b
        return y

    def show_parameters(self):
        print("我的网络参数如下：")
        print(f"self.w = {self.w}")
        print(f"self.b = {self.b}")


# network_1
network_1 = NeuralNetwork(2, 3)  # 创建实例
network_1.show_parameters()
result_1 = network_1.forward(2)
print(f"计算结果为：{result_1}")

# network_2
network_2 = NeuralNetwork(4, 5)  # 创建实例
network_2.show_parameters()
result_2 = network_2.forward(2)
print(f"计算结果为：{result_2}")


class BankAccount:
    def __init__(self, initial_balance=0):
        self.balance = initial_balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return True
        return False

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False

    def get_balance(self):
        return self.balance

# 创建一个银行账户实例
account = BankAccount(1000)

# 模拟存款和取款操作
success_deposit = account.deposit(50)
print("success_deposit = ",success_deposit)
success_withdraw = account.withdraw(2000)
print("success_withdraw = ",success_withdraw)
current_balance = account.get_balance()

print(current_balance)  # 显示当前余额


class NeuralNetwork:
    def __init__(self, input_layer, hidden_layer, output_layer):
        print("我是神经网络")
        print(f"输入层有{input_layer}个神经元")
        print(f"隐藏层有{hidden_layer}个神经元")
        print(f"输出层有{output_layer}个神经元")

    def forward(self, x):
        print(f"已经接收到输入{x}")
        print("我在前向传播")


input_layer = 256
hidden_layer = 128
output_layer = 10

network = NeuralNetwork(input_layer, hidden_layer, output_layer)  # 创建实例
network.forward(10)


class BankAccount:
    def __init__(self, initial_balance=0):
        self._balance = initial_balance  # 私有属性

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            return True
        return False

    def withdraw(self, amount):
        if 0 < amount <= self._balance:
            self._balance -= amount
            return True
        return False

    def get_balance(self):
        return self._balance


class SavingsAccount(BankAccount):
    def __init__(self, initial_balance, interest_rate):
        super().__init__(initial_balance)  # 调用父类的构造函数
        self.interest_rate = interest_rate  # 新增的属性：利率

    def add_interest(self):
        interest = self._balance * self.interest_rate / 100
        self._balance += interest
        return interest


# 创建一个储蓄账户实例
savings_account = SavingsAccount(1000, 5)  # 初始余额1000，利率5%

# 模拟计算利息
interest = savings_account.add_interest()
new_balance = savings_account.get_balance()

print(interest, new_balance)  # 显示计算的利息和新余额





class Shape:
    def area(self):
        return 0

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius * self.radius

class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

def print_area(shape):
    print("The area of the shape is:", shape.area())

# 创建不同形状的实例
circle = Circle(5)
rectangle = Rectangle(10, 5)

# 使用多态性质打印面积
print_area(circle)
print_area(rectangle)



class NeuralNetwork:
    def __init__(self, input_layer, hidden_layer, output_layer):
        print("我是神经网络 ")
        print(f"输入层有{input_layer}个神经元")
        print(f"隐藏层有{hidden_layer}个神经元")
        print(f"输出层有{output_layer}个神经元")

    def forward(self, x):
        print(f"已经接收到输入{x}")
        print("我在前向传播")


class CNN(NeuralNetwork):
    def __init__(self, input_layer, hidden_layer, output_layer, filters):
        super().__init__(input_layer, hidden_layer, output_layer)
        self.filters = filters
        print(f"我是卷积神经网络，我有{self.filters}个卷积核")

    def convolution(self, x):
        print(f"对{x}进行卷积操作")

class RNN(NeuralNetwork):
    def __init__(self, input_layer, hidden_layer, output_layer, time_steps):
        super().__init__(input_layer, hidden_layer, output_layer)
        self.time_steps = time_steps
        print(f"我是循环神经网络，我有{self.time_steps}个时间步")

    def recurrent(self, x):
        print(f"对{x}进行循环操作")


# 使用示例
input_layer = 256
hidden_layer = 128
output_layer = 10

cnn_network = CNN(input_layer, hidden_layer, output_layer, filters=32)  # 创建CNN实例
cnn_network.convolution("图像数据")

rnn_network = RNN(input_layer, hidden_layer, output_layer, time_steps=5)  # 创建RNN实例
rnn_network.recurrent("序列数据")


