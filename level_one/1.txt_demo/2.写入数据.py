# 使用 'write' 方法写入文件
# with open('resources/example_1.txt', 'w') as file:
#     file.write("Hello, World!")


# 使用 'writelines' 方法写入文件
lines = ["Hello, World!", "Welcome to Python programming."]
with open('resources/example_2.txt', 'w') as file:
    file.writelines(line + '\n' for line in lines)
