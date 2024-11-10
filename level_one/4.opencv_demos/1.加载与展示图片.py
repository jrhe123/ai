import cv2

img_path = r'resources/food.png'

# 以彩色模式读取图片
image_color = cv2.imread(img_path)

# 以灰度模式读取图片
image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


# 显示图片
cv2.imshow('Color Image', image_color)
# cv2.imshow('Grayscale Image', image_gray)

# 等待用户按键，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()



