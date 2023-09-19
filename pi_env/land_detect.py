import cv2
import os
import numpy as np
import time
import threading
import math
import Serial_Servo_Running as SSR
import signal
import PWMServo
import pandas as pd
from hiwonder.PID import PID
import hiwonder.Misc as Misc
import hiwonder.Board as Board
import hiwonder.Camera as Camera
import hiwonder.ActionGroupControl as AGC
import hiwonder.yaml_handle as yaml_handle
from CameraCalibration.CalibrationConfig import *
# from main_subfunction import *
from pi_send_img import *
org_img = None
ret = None
cap = Camera.Camera()
cap.camera_open()
def get_img():
    global org_img
    global ret
    global cap
    # global orgFrame
    while True:
        ret, org_img = cap.read()
        if ret:
            cv2.imshow('img', org_img)
            # orgFrame = cv2.resize(org_img, (ori_width, ori_height),
            # interpolation=cv2.INTER_CUBIC)
            key = cv2.waitKey(1)
            if key == 27:
                break
            else:
                time.sleep(0.01)


# 读取图像线程
th1 = threading.Thread(target=get_img)
th1.setDaemon(True)  # 设置为后台线程，这里默认是False，设置为True之后则主线程不用等待子线程
th1.start()
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

def img_land_test():
    land_result = pi_send_img()
    print(land_result)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    while True:
        time.sleep(2)
        start_time = time.time()
        cv2.imwrite("pi_send_img.jpg", org_img)
        img_land_test()
        end_time = time.time()
        solution = end_time-start_time
        print(solution)
        time.sleep(2)