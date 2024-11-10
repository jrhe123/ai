import pickle

# # 示例数据
# data = {
#     'name': 'John',
#     'age': 30,
#     'is_student': False,
#     'grades': [85, 90, 78, 92]
# }
#
# # 使用 pickle 保存数据
# with open('data.pkl', 'wb') as file:
#     pickle.dump(data, file)

# 使用 pickle 加载数据
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)