import cv2

# 创建一个 VideoCapture 对象，参数 0 表示使用默认的摄像头, 也可以传入一个视频文件的路径
cap = cv2.VideoCapture(0)   # resources/piano.mp4
# cap = cv2.VideoCapture("resources/piano.mp4")   # resources/piano.mp4

while True:
    # 读取一帧
    ret, frame = cap.read()

    # 如果读取成功，显示这一帧
    if ret:
        cv2.imshow('Frame', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
