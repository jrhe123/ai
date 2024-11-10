import shutil

file_1_loc = './resources/保存目录1/fire.jpg'
file_1_save_loc = './resources/保存目录2/fire_copy.jpg'
shutil.copyfile(file_1_loc, file_1_save_loc)




import shutil

# 源目录和目标目录
src = 'resources/fire_yolo_format'
dst = 'resources/fire_yolo_format_new'

# 使用copytree复制目录
shutil.copytree(src, dst)

print(f"Directory copied from {src} to {dst}")




import shutil

# 源目录和目标目录
src = 'resources/fire_yolo_format'
dst = 'resources/fire_yolo_format_new_2'

# 使用copytree复制目录
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.txt"))

print(f"Directory copied from {src} to {dst}")


import shutil

file_1_loc = './resources/保存目录1/fire_label.txt'
file_1_save_loc = './resources/保存目录2/fire_label.txt'
shutil.move(file_1_loc, file_1_save_loc)




import os

file_loc = r'./resources/保存目录1/fire.jpg'

os.remove(file_loc)





import os

dir_name = "my_dir"
if os.path.exists(dir_name):
    print("文件夹已经存在！")
else:
    os.mkdir(dir_name)
    print("文件夹已经创建完毕！")




import os
os.makedirs("my_dir_1\\my_dir_2\\my_dir_3")

import os

root_dir = "dir_loc"

file_full_path_list = []
for root, dirs, files in os.walk(root_dir):
    for file_i in files:
        file_i_full_path = os.path.join(root, file_i)
        file_full_path_list.append(file_i_full_path)

print(file_full_path_list)




import os

dir_path = 'my_dir'

if os.path.exists(dir_path):
    print("删除文件夹"+dir_path)
    os.rmdir('my_dir')
    print("删除完成")
else:
    print("文件夹"+dir_path+"不存在")







import os
import shutil

dir_name = "my_dir"
if os.path.exists(dir_name):
    shutil.rmtree(dir_name) # 文件夹里有东西也一并删除
    print("文件夹已经删除！")
else:
    os.mkdir(dir_name)
    print("文件夹不存在！")






# 文件清洗
# 将 'resources/fire_yolo_format' 当中标签名和图片名匹配的文件，按原排放顺序保存到新的文件夹 'resources/clean_data'
# 需要用代码创建文件夹

import os
import shutil


def get_files(root_path):
    file_full_path_list = []
    file_name_list = []
    for root, dirs, files in os.walk(root_path):
        for file_i in files:
            file_i_full_path = os.path.join(root, file_i)
            file_full_path_list.append(file_i_full_path)
            file_name_list.append(os.path.splitext(file_i)[0])  # 使用os.path.splitext获取文件名

    return file_full_path_list, file_name_list


root_path_from = r'resources/fire_yolo_format'
root_path_save = r'resources/clean_data'

root_images_from = os.path.join(root_path_from, 'images')
root_labels_from = os.path.join(root_path_from, 'labels')

root_images_save = os.path.join(root_path_save, 'images')
root_labels_save = os.path.join(root_path_save, 'labels')

# 创建保存的文件夹
for dir_path in [root_images_save, root_labels_save]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

image_file_full_path_list, image_file_name_list = get_files(root_images_from)
labels_file_full_path_list, labels_file_name_list = get_files(root_labels_from)

intersection_set = set(image_file_name_list) & set(labels_file_name_list)

# 复制交集中的文件到新的文件夹
for file_name in intersection_set:
    image_index = image_file_name_list.index(file_name)
    label_index = labels_file_name_list.index(file_name)

    shutil.copy(image_file_full_path_list[image_index], root_images_save)
    shutil.copy(labels_file_full_path_list[label_index], root_labels_save)




import os
import shutil

root_path_from = r'resources/fire_yolo_format'
root_path_save = r'resources/clean_data'

root_images_from = os.path.join(root_path_from, 'images')
root_labels_from = os.path.join(root_path_from, 'labels')
root_images_save = os.path.join(root_path_save, 'images')
root_labels_save = os.path.join(root_path_save, 'labels')

dir_list_1 = ['images', 'labels']
dir_name_list = ['train','test', 'val']
for dir_1_i in dir_list_1:
    for dir_2_i in dir_name_list:
        dir_i_full_path = os.path.join(root_path_save, dir_1_i,dir_2_i)
        if not os.path.exists(dir_i_full_path):
            os.makedirs(dir_i_full_path)



def get_info(root_from):
    file_full_path_list = []
    file_name_list = []
    for root, dirs, files in os.walk(root_from):
        for file_i in files:
            file_i_full_path = os.path.join(root,file_i)
            file_full_path_list.append(file_i_full_path)
            # file_name_list.append(file_i_full_path[:-4])
            file_i_split_path = file_i_full_path.split('\\')
            # print(file_i_split_path)
            file_i_short_path = os.path.join(file_i_split_path[-2],file_i_split_path[-1])
            # print(file_i_short_path)
            # file_name_list = file_i_short_path[:-4]
            # print(file_name_list)
            file_name_list.append(file_i_short_path[:-4])

    return file_full_path_list, file_name_list

image_full_path_list, image_name_list = get_info(root_images_from)
label_full_path_list, label_name_list = get_info(root_labels_from)



image_name_set = set(image_name_list)
print(len(image_name_list))
print(len(image_name_set))

label_name_set = set(label_name_list)
print(len(label_name_set))

intersection_set = image_name_set & label_name_set
print(len(intersection_set))

print(intersection_set)



for intersection_i in intersection_set:
    intersection_i_image_full_path_from = os.path.join(root_images_from, intersection_i) + '.jpg'
    # print(intersection_i_image_full_path)
    intersection_i_label_full_path_from = os.path.join(root_labels_from, intersection_i) + '.txt'
    # print(intersection_i_label_full_path)

    intersection_i_image_full_path_save = os.path.join(root_images_save, intersection_i) + '.jpg'

    intersection_i_label_full_path_save = os.path.join(root_labels_save, intersection_i) + '.txt'

    shutil.copy(intersection_i_image_full_path_from, intersection_i_image_full_path_save)
    shutil.copy(intersection_i_label_full_path_from, intersection_i_label_full_path_save)













