import cv2
import sys

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
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
from main_subfunction import *
from land_detect import *

# ################################################初始化##############################
open_once = yaml_handle.get_yaml_data('/boot/camera_setting.yaml')['open_once']
if open_once:
    cap = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream?dummy=param.mjpg')
else:
    cap = Camera.Camera()
    cap.camera_open()
Board.setPWMServoPulse(1, 980, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
Board.setPWMServoPulse(2, 1400, 500)
SSR.running_action_group('0', 1)  ##回中
ret = False  # 读取图像标志位
org_img = None  # 全局变量，原始图像
# count=0
c = 0
a = 0
stop = 0
frame_floor_blue = None
frame_show = True
frame_ready = True

stageLeft = {'start_door', 'square_hole', 'bridge',  # #########     剩余关卡
             'cross_obs', 'baffler', 'gate',
             'deadball', 'floor', 'end_door'}
# 获取最大轮廓以及它的面积
ori_width = int(4 * 160)  # 原始图像640x480
ori_height = int(3 * 160)
r_width = int(4 * 20)  # 处理图像时缩小为80x60,加快处理速度，谨慎修改！
r_height = int(3 * 20)
color_range1 = {
    "black": [(0, 0, 0), (65, 255, 255)],
    "yellow": [(0, 120, 140), (255, 255, 255)],
    "red": [(0, 156, 88), (255, 255, 255)],
    "green": [(90, 0, 135), (255, 110, 255)],
    "blue_stair": [(35, 0, 0), (255, 130, 120)],
    "blue_door": [(0, 0, 0), (255, 255, 112)],
    "blue_mine": [(0, 125, 0), (255, 255, 100)],
    "white_door": [(160, 0, 0), (255, 255, 255)],
    "white_road": [(160, 0, 0), (255, 255, 255)],
    "white_mine": [(100, 0, 0), (255, 255, 255)],
    "white_stair": [(193, 0, 0), (255, 255, 255)],
    "white_ball": [(193, 0, 0), (255, 255, 255)],

    "corner_land": [(54, 0, 46), (255, 255, 255)],
}


# ###############################################读取图像线程###########################
def get_img():
    global org_img
    global ret
    global frame_cali
    global cap
    # global orgFrame
    calibration_param_path = "calibration_param0.npz"
    param_data = np.load(calibration_param_path)
    dim = tuple(param_data["dim_array"])
    k = np.array(param_data["k_array"].tolist())
    d = np.array(param_data["d_array"].tolist())
    p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, dim, None)
    map = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), p, dim, cv2.CV_16SC2)
    while True:
        ret, org_img = cap.read()
        frame_cali = cv2.remap(
            org_img,
            map[0],
            map[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if ret:
            cv2.imshow('img', frame_cali)
            # orgFrame = cv2.resize(org_img, (ori_width, ori_height),
            # interpolation=cv2.INTER_CUBIC)
            key = cv2.waitKey(1)
            if key == 27:
                break
            else:
                time.sleep(0.01)


# 读取图像线程
th1 = threading.Thread(target=get_img)
th1.setDaemon(True)
th1.start()
"""
org_img = cv2.imread('lei_ground1.png')
cv2.imshow('org_img',org_img)
cv2.waitKey()
cv2.destroyAllWindows()
"""
def bridge_color_detect_top(color, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes

    if defaults is None:
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults

    time.sleep(0.5)
    img = frame_cali.copy()
    close = get_frame_bin(img, color, ksizes_[0], ksizes_[1])
    cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnt_max, area_max = getAreaMaxContour(cnts)
    percent = round(100 * area_max / ((close.shape[0]) * (close.shape[1])))  # 计算目标区域占比

    # 得到上顶点，计算中点及线段角度
    if cnt_max is None:
        cnt_max = np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])
        top_angle, top_center, top_left, top_right = defaults_
    else:
        top_left = cnt_max[0][0]
        top_right = cnt_max[0][0]
        for c in cnt_max:  # 遍历找到四个顶点
            if c[0][0] + grad * c[0][1] < top_left[0] + grad * top_left[1]:
                top_left = c[0]
            if -c[0][0] + grad * c[0][1] < -top_right[0] + grad * top_right[1]:
                top_right = c[0]
        line_top = line(top_left, top_right)
        top_angle = line_top.angle()
        top_center = line_top.mid_point()  # 需要转换成tuple使用
        img_show = img.copy()
        cv2.drawContours(img_show, [cnt_max], -1, (0, 255, 255), 2)
        cv2.circle(img_show, tuple(top_center), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(top_left), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(top_right), 5, (255, 0, 0), 2)
    '''
    cv2.imshow("Result", img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return percent, top_angle, [top_center, top_left, top_right]


def bridge_color_detect_bottom(color, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes

    if defaults is None:
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults

    # 图像处理及roi提取
    # global frame_cali, frame_ready
    # while not frame_ready:
    #     time.sleep(0.01)
    # frame_ready = False
    # img = frame_cali.copy()
    # frame_cali = cv2.imread("key frame/images/pit.jpg")
    img = frame_cali.copy()
    close = get_frame_bin(img, color, ksizes_[0], ksizes_[1])
    cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnt_max, area_max = getAreaMaxContour(cnts)
    percent = round(100 * area_max / ((close.shape[0]) * (close.shape[1])))

    if cnt_max is None:
        cnt_max = np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])
        bottom_angle, bottom_center, bottom_left, bottom_right = defaults_

    else:
        bottom_left = cnt_max[0][0]
        bottom_right = cnt_max[0][0]
        for c in cnt_max:  # 遍历找到四个顶点
            if c[0][0] - grad * c[0][1] < bottom_left[0] - grad * bottom_left[1]:
                bottom_left = c[0]
            if c[0][0] + grad * c[0][1] > bottom_right[0] + grad * bottom_right[1]:
                bottom_right = c[0]
        line_bottom = line(bottom_left, bottom_right)
        bottom_angle = line_bottom.angle()
        bottom_center = line_bottom.mid_point()  # 需要转换成tuple使用

    # img_show = img.copy()
    # cv2.drawContours(img_show, [cnt_max], -1, (0, 255, 255), 2)
    # cv2.line(img_show, tuple(bottom_left), tuple(bottom_right), (255, 255, 0), 3)
    # cv2.circle(img_show, tuple(bottom_center), 5, (255, 0, 0), 2)
    # cv2.circle(img_show, tuple(bottom_left), 5, (255, 0, 0), 2)
    # cv2.circle(img_show, tuple(bottom_right), 5, (255, 0, 0), 2)
    # cv2.imshow("Result", img_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return percent, bottom_angle, [bottom_center, bottom_left, bottom_right]


def bridge_percent():
    global frame_cali
    time.sleep(0.5)
    img = frame_cali.copy()
    border = cv2.copyMakeBorder(img, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255))  # 扩展白边，防止边界无法识别
    org_img_copy = cv2.resize(border, (r_w, r_h), interpolation=cv2.INTER_CUBIC)  # 将图片缩放
    frame_gauss = cv2.GaussianBlur(org_img_copy, (3, 3), 0)  # 高斯模糊
    frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    frame_green = cv2.inRange(frame_hsv, color_range['bridge'][0],
                              color_range['bridge'][1])  # 对原图像和掩模(颜色的字典)进行位运算
    opened = cv2.morphologyEx(frame_green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)  # 找出轮廓cv2.CHAIN_APPROX_NONE
    areaMaxContour, area_max = getAreaMaxContour(contours)  # 找出最大轮廓
    percent = round(area_max * 100 / (r_w * r_h), 2)
    return percent


################################################第三关：独木桥 ##################
def bridge_move():
    global org_img
    global frame_cali
    state = 'bridge'
    step = 0
    SSR.running_action_group('stand', 1)
    Board.setPWMServoPulse(1, 1100, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
    Board.setPWMServoPulse(2, 1450, 500)
    while state == 'bridge':
        # 测数据所用
        if step == -1:
            while step == -1:
                # 获取数据
                time.sleep(0.2)
                grad = 2.5
                ksizes = [11, 7]
                defaults = [-2, (320, 240), (300, 240), (340, 240)]
                _, top_angle, points = bridge_color_detect_top(
                    "green", grad, ksizes, defaults
                )
                top_center_x, top_center_y = points[0]
                print(f"color_detect_top:  top_angle, top_center_x, top_center_y = {top_angle, top_center_x, top_center_y}")

                _, bot_angle, points = bridge_color_detect_bottom(
                    "green", grad, ksizes, defaults
                )
                bot_center_x, bot_center_y = points[0]
                print(f"color_detect_bottom : bot_angle, bot_center_x, bot_center_y = {bot_angle,bot_center_x, bot_center_y}")
                print("------------------------")

        # 靠近独木桥
        if step == 0:
            count = 0
            while step == 0:
                grad = 2.5
                ksizes = [11, 7]
                defaults = [-2, (320, 240), (300, 240), (340, 240)]
                # 计算底部的的角度，用以纠正方向,bot_center_y，用以纠正左右移动
                _, bot_angle, points = bridge_color_detect_bottom(
                    "green", grad, ksizes, defaults
                )
                bot_center_x, bot_center_y = points[0]
                print(f"color_detect_bottom :bot_angle, bot_center_x, bot_center_y = {bot_angle, bot_center_x, bot_center_y}")

                if bot_angle <= -3.5:
                    print("右转")
                    SSR.running_action_group('turn_right_fast', 1)
                    time.sleep(0.1)
                elif bot_angle >= 3:
                    print("左转")
                    SSR.running_action_group('turn_left_fast', 1)
                    time.sleep(0.1)
                else:
                    if bot_center_x <= 20:
                        print("左移")
                        SSR.running_action_group('160_test', 1)
                        time.sleep(0.1)
                    elif bot_center_x >= 350:
                        print("右移")
                        SSR.running_action_group('161_test', 1)
                        time.sleep(0.1)
                    else:
                        print("正对，前进")
                        # 当top_y很小时，直接往前冲几步就过了独木桥
                        if bot_center_y <= 450:
                            SSR.running_action_group('232new', 2)
                            # time.sleep(0.2)
                            # 进入下一步
                        else:
                            count += 1
                            if count >= 2:
                                print("state:bridge:step 0 done, goto step 1")
                                step = 1

        # 机器人过独木桥过程
        if step == 1:
            count = 0
            while step == 1:
                grad = 2.5
                ksizes = [11, 7]
                defaults = [-2, (320, 240), (300, 240), (340, 240)]
                # 计算顶部的角度，用以纠正方向，top_y用以判断是否靠近终点
                _, top_angle, points = bridge_color_detect_top(
                    "green", grad, ksizes, defaults
                )
                top_center_x, top_center_y = points[0]
                print(f"color_detect_top:  top_angle, top_center_x, top_center_y = {top_angle,top_center_x, top_center_y}")
                # 计算底部的cent_x，用以纠正左右移动
                _, bot_angle, points = bridge_color_detect_bottom(
                    "green", grad, ksizes, defaults
                )
                bot_center_x, bot_center_y = points[0]
                print(f"color_detect_bottom : bot_center_x, bot_center_y = {bot_center_x, bot_center_y}")

                if top_angle <= -2.5:
                    print("右转")
                    SSR.running_action_group('turn_right_fast', 1)
                    time.sleep(0.1)
                elif top_angle >= 3.0:
                    print("左转")
                    SSR.running_action_group('turn_left_fast', 1)
                    time.sleep(0.1)
                else:
                    if bot_center_x <= 300:
                        print("左移")
                        SSR.running_action_group('160', 1)
                        time.sleep(0.1)
                    elif bot_center_x >= 340:
                        print("右移")
                        SSR.running_action_group('161', 1)
                        time.sleep(0.1)
                    else:
                        print("正对，前进")
                        # 当bridge_percent很小时，直接往前冲几步就过了独木桥
                        bridge_per = bridge_percent()
                        print(f"bridge_per={bridge_per}")
                        if bridge_per >= 7:
                            SSR.running_action_group('232new', 2)
                            # time.sleep(0.2)
                            # 进入下一步
                        else:
                            count += 1
                            if count >= 2:
                                print(f"接近终点，终点bridge_percent={bridge_per}")
                                for i in range(5):
                                    print("再往前走几步通过独木桥")
                                    SSR.running_action_group('232new', 2)
                                    time.sleep(0.1)
                                print("state:bridge:step 1 done, goto step 2")
                                step = 888888
                                state = None
    print('----------------- state:bridge    out -------------------')


def bridge():  # 桥
    Board.setPWMServoPulse(1, 1100, 500)
    Board.setPWMServoPulse(2, 1450, 500)
    bridge_move()


bridge()