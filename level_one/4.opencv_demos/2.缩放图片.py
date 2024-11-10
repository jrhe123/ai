import cv2

# 读取图片
image = cv2.imread('resources/food.png')

# 检查图片是否正确加载
if image is None:
    print("Error: Could not load image.")
    exit()


print(image.shape)

# 获取图片的原始尺寸
original_height, original_width = image.shape[:2]
#
# 计算新的尺寸
new_width = int(original_width / 2)
new_height = int(original_height / 2)
#
# 使用cv2.resize进行图片缩放
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
#
# 显示原始图片和缩放后的图片
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image', resized_image)
#
# 等待用户按键，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
