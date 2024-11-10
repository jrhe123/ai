# # 使用 'read' 方法读取文件的所有内容
# with open('resources/training_log.txt', 'r') as file:
#     content = file.read()
#     print(content)


# # 使用 'readline' 方法逐行读取文件
# with open('resources/training_log.txt', 'r') as file:
#     line = file.readline()
#     # print(line)
#     while line:
#         print(line, end='')
#         line = file.readline()


# 使用 'readlines' 方法读取文件的所有行
with open('resources/training_log.txt', 'r') as file:
    lines = file.readlines()
    # print(lines)
    for line in lines:
        print(line, end='')










