import cv2

# 读取图像
image = cv2.imread('resources/food.png')

# 如果图像不为空，则保存图像
if image is not None:
    cv2.imwrite('output_image.png', image)
else:
    print("无法读取图像")