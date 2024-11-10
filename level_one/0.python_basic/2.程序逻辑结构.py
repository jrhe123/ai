a = 1
b = 3.14
c = '123'
d = "这也是字符串"
e = '''这也是字符串'''

print("a的数据类型是",type(a))
print("b的数据类型是",type(b))
print("c的数据类型是",type(c))
print("d的数据类型是",type(d))
print("e的数据类型是",type(e))


num_str = input("请输入小数:")
print("num_str = ",num_str," 格式是：",type(num_str))
num_float = float(num_str)
print("num_float = ",num_str," 格式是：",type(num_float))


name = "Alice"
age = 30
print("My name is %s and I'm %d years old."%(name, age))
print("My name is {} and I'm {} years old.".format(name, age))
print(f"My name is {name} and I'm {age} years old.")



number = 12.3456
print("%.2f" % number)
print("{:.2f}".format(number))
print(f"{number:.2f}")

'''
选择结构
'''

x = 10
if x > 5:
    print("x大于5")


x = 10
if x > 5:
    print("x>5")
else:
    print("x<=5")



x = 5
if x > 10:
    print("x大于10")
elif x == 5:
    print("x是5")
else:
    print("x小于10，但不是 5")




a = 10
b = 20
result = a + b
answer = int(input(f"请输入{a}+{b}的结果："))
if result == answer:
    print("回答正确！")
else:
    print("回答错误")



'''
循环结构
'''

epoch = 5
for epoch_i in range(epoch):
    print("--------------")
    print(f"正在处理第{epoch_i}个epoch的数据")
    print(f"第{epoch_i}个数据处理完毕")



optimizers = ["SGD", "Adam", "Momentum","Adagrad"]
for optimizer_i in optimizers:
    print("正在使用 ",optimizer_i," 进行优化")


img_list = ["img_1.png", "img_2.png", "img_3.png"]
for index, img_i in enumerate(img_list):
    print(f"索引 {index} 对应的数据是 {img_i}")

command = ""
while command != "end":
    command = input("请输入命令：")
    print("正在执行命令：", command)




# 这是一个数字列表，机器人将在这个列表中搜索数字“5”
numbers = [1, 3, 4, 2, 5, 6, 8, 7, 9]

# 这是一个标志，用来表示机器人是否找到了数字“5”
found = False

# 机器人开始搜索数字“5”
for number in numbers:
    print(f"正在查看数字{number}")
    if number == 5:
        found = True
        print(f"机器人找到了数字{number}！")
        break  # 一旦找到数字“5”，就退出循环

# 检查机器人是否找到了数字“5”
if not found:
    print("机器人没有找到数字5。")




# 这是一个数字列表，机器人将在这个列表中搜索不是“5”的数字
numbers = [1, 3, 4, 2, 5, 6, 8, 7, 9]

# 机器人开始搜索不是“5”的数字
for number in numbers:
    print(f"正在查看数字{number}")
    if number == 5:
        continue  # 如果数字是“5”，跳过当前迭代，继续下一次循环
    print(f"机器人找到了数字{number}！")





a = 10
b = 20
result = a + b
while True:
    answer = int(input(f"请输入{a}+{b}的结果："))
    if result == answer:
        print("回答成功！")
        break
    else:
        print("回答错误！")



















