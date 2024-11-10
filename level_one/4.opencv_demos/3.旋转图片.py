import cv2

# 读取图片
image = cv2.imread('resources/food.png')

# 使用cv2.rotate()函数旋转图片
rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90度
rotated_180 = cv2.rotate(image, cv2.ROTATE_180)  # 顺时针旋转180度
rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 顺时针旋转270度

cv2.imshow('original', image)
cv2.imshow('90 degree', rotated_90)
cv2.imshow('180 degree', rotated_180)
cv2.imshow('270 degree', rotated_270)
cv2.waitKey(0)