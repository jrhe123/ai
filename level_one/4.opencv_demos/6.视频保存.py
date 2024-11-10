import cv2

# 定义视频捕获对象
cap = cv2.VideoCapture(0)

# 检查是否成功打开摄像头
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 获取摄像头的帧宽度和帧高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义视频编码器和输出文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # 将当前帧写入输出视频文件
    out.write(frame)

    # 显示当前帧
    cv2.imshow('frame', frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
