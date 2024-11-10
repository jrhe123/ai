import numpy as np
import cv2

# 黑白
two_d_matrix = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
# 彩色
three_d_matrix = np.random.randint(0, 256, (3, 512, 512), dtype=np.uint8)

# 图片通道顺序为BGR
three_d_matrix = three_d_matrix.transpose(1, 2, 0)

cv2.imwrite('two_d_matrix.png', two_d_matrix)
cv2.imwrite('three_d_matrix.png', three_d_matrix)

cv2.waitKey(0)