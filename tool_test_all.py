import cv2
import os
import numpy as np
import time
import threading
import math
# import Serial_Servo_Running as SSR
# import signal
# import PWMServo
# import pandas as pd
# from hiwonder.PID import PID
# import hiwonder.Misc as Misc
# import hiwonder.Board as Board
# import hiwonder.Camera as Camera
# import hiwonder.ActionGroupControl as AGC
# import hiwonder.yaml_handle as yaml_handle
# from CameraCalibration.CalibrationConfig import *
# from main_subfunction import *


# 视频图像采集工具——————start
def get_video():
    """
    视频图像采集工具
    """
    cap = cv2.VideoCapture(0)
    f = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter("all.mp4", f, 30, (640, 480))
    while (True):
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        out_video.write(frame)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
# 视频图像采集工具——————end


# 图像抽帧工具——————start
def videos_chouzhen():
    """
    图像抽帧工具
    """
    import cv2
    import math
    import time
    import os
    video_Path = '2023_v/'   # 视频路径
    img_save_path = 'key frame/images/'  # 帧保存路径
    videos = os.listdir(video_Path)
    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        os.mkdir(img_save_path + each_video_name)
        each_video_save_full_path = os.path.join(img_save_path, each_video_name) + '/'
        each_video_full_path = os.path.join(video_Path, each_video)
        cap = cv2.VideoCapture(each_video_full_path)
        print(each_video_full_path)
        frame_count = 1
        success = True
        while (success):
            success, frame = cap.read()
            if success == True:
                cv2.imwrite(each_video_save_full_path + "%04d.jpg" % frame_count, frame)
                frame_count += 1

# 图像抽帧工具——————end
videos_chouzhen()

# 求目标hsv阈值的工具——————start
import cv2
import numpy as np
def callback(object):
    pass
def Choose_Color():
    """
    求目标hsv阈值的工具
    """
    filename = "key frame/images/all_2023_first_part/0002.jpg"
    image0 = cv2.imread(filename, 1)
    img = cv2.cvtColor(image0, cv2.COLOR_BGR2HSV)
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))

    cv2.imshow("image", img)

    cv2.createTrackbar("H_min", "image", 50, 255, callback)
    cv2.createTrackbar("H_max", "image", 150, 255, callback)

    cv2.createTrackbar("S_min", "image", 0, 255, callback)
    cv2.createTrackbar("S_max", "image", 255, 255, callback)

    cv2.createTrackbar("V_min", "image", 0, 255, callback)
    cv2.createTrackbar("V_max", "image", 255, 255, callback)

    while (True):

        H_min = cv2.getTrackbarPos("H_min", "image", )
        S_min = cv2.getTrackbarPos("S_min", "image", )
        V_min = cv2.getTrackbarPos("V_min", "image", )

        H_max = cv2.getTrackbarPos("H_max", "image", )
        S_max = cv2.getTrackbarPos("S_max", "image", )
        V_max = cv2.getTrackbarPos("V_max", "image", )

        lower_hsv = np.array([H_min, S_min, V_min])
        upper_hsv = np.array([H_max, S_max, V_max])

        mask = cv2.inRange(img, lower_hsv, upper_hsv)
        cv2.imshow("mask", mask)
        if cv2.waitKey(1) & 0XFF == 27:
            break
# 求目标hsv阈值的工具——————end
# videos_chouzhen()
# Choose_Color()
"""
# 雷区摄像需要低头——————start
lab_data = None
servo_data = None
def load_config():
    global lab_data, servo_data

    lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
load_config()
# 初始位置
def initMove():
    Board.setPWMServoPulse(1, servo_data['servo1'], 500)
    Board.setPWMServoPulse(2, servo_data['servo2'], 500)
initMove()


'''
雷区摄像需要低头，再执行录像程序就行，仅用于雷区的识别、采集数据，不用低头就把下面两行注释掉

'''

Board.setPWMServoPulse(1, 1500, 500)
Board.setPWMServoPulse(2, 1400, 500)
# 雷区摄像需要低头——————end
# get_video()
"""
# Board.setPWMServoPulse(1, 1500, 500)
# Board.setPWMServoPulse(2, 1400, 500)
# get_video()