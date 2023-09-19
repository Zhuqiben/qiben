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
th1.setDaemon(True)
th1.start()
"""
org_img = cv2.imread('lei_ground1.png')
cv2.imshow('org_img',org_img)
cv2.waitKey()
cv2.destroyAllWindows()
"""

stageLeft = {'start_door', 'square_hole', 'bridge',  # #########     剩余关卡
             'cross_obs', 'baffler', 'gate',
             'deadball', 'floor', 'end_door'}
# 获取最大轮廓以及它的面积
ori_width = int(4 * 160)  # 原始图像640x480
ori_height = int(3 * 160)
r_width = int(4 * 20)  # 处理图像时缩小为80x60,加快处理速度，谨慎修改！
r_height = int(3 * 20)


# ############################################### 开门 ######################
def greeen_ground_percent():  # 计算草地percent，判断是否走完
    global org_img
    # img = org_img[200:, :]
    r_h = org_img.shape[0]
    r_w = org_img.shape[1]

    gauss = cv2.GaussianBlur(org_img, (3, 3), 0)  # 高斯模糊
    hsv = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    Imask0 = cv2.inRange(hsv, color_range['green_ground'][0],
                         color_range['green_ground'][1])  # 对原图像和掩模(颜色的字典)进行位运算
    # opened = cv2.morphologyEx(Imask0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接
    # erode = cv2.erode(Imask0, None, iterations=1)  # 腐蚀
    dilate = cv2.dilate(Imask0, np.ones((3, 3), np.uint8), iterations=2)  # 膨胀

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
    area = getAreaSumContour(contours)  # 绿色的面积，不是提取最大轮廓的面积
    # percent0 = round(area * 100 / (r_w * r_h), 2)  # 绿色的面积百分比（绿色的面积，不是提取最大轮廓的面积）
    areaMaxContour, area_max = getAreaMaxContour(contours)  # 找出最大轮廓
    # cv2.drawContours(org_img, areaMaxContour, -1, (255, 0, 255), 2)
    # cv2.imshow('o', org_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    percent = round(100 * area_max / (r_w * r_h), 2)  # 最大轮廓最小外接矩形的面积占比
    print(f"green_percent = {percent}")
    return percent


def start_door_color_detect_bottom(color, grad=1.0, ksizes=None, defaults=None):
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


def start_door():
    global org_img
    r_h = 120
    r_w = 160
    state = 'start_door'
    if state == 'start_door':  # 初始化
        print('进入start_door')
        step = 0
        Board.setPWMServoPulse(1, 1200, 500)  # NO3机器人，初始云台角度
        Board.setPWMServoPulse(2, 1400, 500)
        time.sleep(0.2)
    else:
        return
    while state == 'start_door':
        if step == 0:
            border = cv2.copyMakeBorder(org_img, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,
                                        value=(255, 255, 255))  # 扩展白边，防止边界无法识别
            org_img_copy = cv2.resize(border, (r_w, r_h), interpolation=cv2.INTER_CUBIC)  # 将图片缩放
            frame_gauss = cv2.GaussianBlur(org_img_copy, (3, 3), 0)  # 高斯模糊
            frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
            frame_door = cv2.inRange(frame_hsv, color_range['yellow_door'][0],
                                     color_range['yellow_door'][1])  # 对原图像和掩模(颜色的字典)进行位运算
            opened = cv2.morphologyEx(frame_door, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接

            contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
            contour_area_sum = getAreaSumContour(contours)

            if contour_area_sum > 1000:
                print(f"step == 0   yellow_door_area_sum ={contour_area_sum}")
                step = 1
        elif step == 1:
            border = cv2.copyMakeBorder(org_img, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,
                                        value=(255, 255, 255))
            org_img_copy = cv2.resize(border, (r_w, r_h), interpolation=cv2.INTER_CUBIC)
            frame_gauss = cv2.GaussianBlur(org_img_copy, (3, 3), 0)
            frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)
            frame_door = cv2.inRange(frame_hsv, color_range['yellow_door'][0],
                                     color_range['yellow_door'][1])
            opened = cv2.morphologyEx(frame_door, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour_area_sum = getAreaSumContour(contours)

        if contour_area_sum > 1000:  ##############################################################################################################参数值
            print(f"step == 1   yellow_door_area_sum ={contour_area_sum}")
        else:
            print('go')
            print(f"else    yellow_door_area_sum ={contour_area_sum}")
            percent = greeen_ground_percent()
            if percent >= 6:  # 判断草地停止程序########################################################################################################参数值
                SSR.running_action_group('239new', 1)  # 直行
            else:
                SSR.running_action_group('stand', 1)
                # SSR.running_action_group('239new', 3)
                print("state:start_door:exit")
                state = None
                # for i in range(1):
                #     SSR.running_action_group('239new', 1)  # 直行
                count = 0
                while True:
                    grad = 2.5
                    ksizes = [11, 7]
                    defaults = [-2, (320, 240), (300, 240), (340, 240)]
                    _, angle, points = start_door_color_detect_bottom(
                        "white_road", grad, ksizes, defaults
                    )
                    if angle <= -2:
                        print("右转")
                        SSR.running_action_group('turn_right_fast', 1)
                        # time.sleep(0.1)
                    elif angle >= 2.8:
                        print("左转")
                        SSR.running_action_group('turn_left_fast', 1)
                        # time.sleep(0.1)
                    else:
                        count += 1
                        if count >= 2:
                            break


# ############################################### 坑 ######################

def pit_ground_percent(type):
    """
    计算坑绿色地percent，判断是否走完
    """
    global org_img
    # img = org_img[200:, :]
    r_h = org_img.shape[0]
    r_w = org_img.shape[1]

    gauss = cv2.GaussianBlur(org_img, (3, 3), 0)  # 高斯模糊
    hsv = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    Imask0 = cv2.inRange(hsv, color_range[type][0],
                         color_range[type][1])  # 对原图像和掩模(颜色的字典)进行位运算
    # opened = cv2.morphologyEx(Imask0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接
    # erode = cv2.erode(Imask0, None, iterations=1)  # 腐蚀
    dilate = cv2.dilate(Imask0, np.ones((3, 3), np.uint8), iterations=2)  # 膨胀

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
    area = getAreaSumContour(contours)  # 绿色的面积，不是提取最大轮廓的面积
    # percent0 = round(area * 100 / (r_w * r_h), 2)  # 绿色的面积百分比（绿色的面积，不是提取最大轮廓的面积）
    areaMaxContour, area_max = getAreaMaxContour(contours)  # 找出最大轮廓
    # cv2.drawContours(org_img, areaMaxContour, -1, (255, 0, 255), 2)
    # cv2.imshow('o', org_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    percent = round(100 * area_max / (r_w * r_h), 2)  # 最大轮廓最小外接矩形的面积占比
    print(f"green_percent = {percent}")
    return percent


def pit_color_detect_top(color, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes

    if defaults is None:
        defaults_ = [0, (135, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults
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


def pit_color_detect_bottom(color, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes

    if defaults is None:
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults

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


def state_pit():
    print("state:hole:enter")
    global state
    state = 'pit'
    step = 0  # 2
    while state == 'pit':
        if step == -1:

            angle_range = [-3, 3]
            center_x_range = [280, 360]
            center_y_switch = 90
            center_y_stop = 120
            SSR.running_action_group('stand', 1)
            Board.setPWMServoPulse(1, 1100, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
            Board.setPWMServoPulse(2, 1400, 500)
            while step == -1:
                # 获取数据
                time.sleep(0.25)
                grad = 2.5
                ksizes = [11, 7]
                defaults = [-2, (320, 240), (300, 240), (340, 240)]
                _, angle, points = pit_color_detect_top(
                    "green", grad, ksizes, defaults
                )
                center_x, center_y = points[0]
                print(f"angle,top_center_x, top_center_y = {angle, center_x, center_y}")

                # _, angle, points = color_detect_top(
                #     "green", grad, ksizes, defaults
                # )
                left_x, left_y = points[1]
                print(f"step == 1: angle,left_x, left_y = {angle, left_x, left_y}")
                print("------------------------")

        # 0、靠近
        if step == 0:
            grad = 2.5
            ksizes = [11, 7]
            defaults = [-2, (320, 240), (300, 240), (340, 240)]
            angle_range = [-4, 4]
            # center_x_range = [280, 380]
            SSR.running_action_group('stand', 1)
            Board.setPWMServoPulse(1, 1100, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
            Board.setPWMServoPulse(2, 1400, 500)
            count = 0
            while step == 0:
                # 获取数据
                time.sleep(0.1)
                _, angle, points = pit_color_detect_top(
                    "green", grad, ksizes, defaults
                )
                center_x, center_y = points[0]
                print(f"angle,center_x, center_y = {angle, center_x, center_y}")
                # 调整角度
                print(f"angle: {angle:.2f}  ({center_x},{center_y})")
                if angle <= -2.5:
                    print("右转")
                    SSR.running_action_group('turn_right_fast', 1)
                    # time.sleep(0.1)
                elif angle >= 3.2:
                    print("左转")
                    SSR.running_action_group('turn_left_fast', 1)
                    # time.sleep(0.1)
                else:
                    print("正对完成,前进")
                    print("step=0,exit!")
                    # 大步走
                    if center_y < 60:
                        SSR.running_action_group('232new', 2)
                    else:
                        count += 1
                        if count >= 3:
                            print("state:hole:step 1 done, goto step 2")
                            step = 1
                            #break

        # 1、左移
        if step == 1:
            grad = 2.5
            ksizes = [11, 7]
            defaults = [-2, (320, 240), (300, 240), (340, 240)]
            left_flag = 0
            camera_flag = 0
            SSR.running_action_group('stand', 1)
            Board.setPWMServoPulse(1, 1100, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
            Board.setPWMServoPulse(2, 1400, 500)
            count = 0

            while step == 1:
                # 获取数据
                time.sleep(0.1)
                _, angle, _ = pit_color_detect_top(
                    "green", grad, ksizes, defaults
                )

                _, _, points = pit_color_detect_bottom(
                    "green", grad, ksizes, defaults
                )
                left_x, left_y = points[1]

                print(f"step == 1: angle,{left_x},{left_y}): {angle:.2f}  ({left_x},{left_y})")
                if angle <= -2:
                    print("右转")
                    SSR.running_action_group('turn_right_fast', 1)
                    time.sleep(0.1)
                elif angle >= 2.8:
                    print("左转")
                    SSR.running_action_group('turn_left_fast', 1)
                    time.sleep(0.1)
                else:
                    print("正对完成,左移调整")
                    # 大步左平移
                    if left_x <= 110:
                        print("左移")
                        SSR.running_action_group('left_move_fast', 1)
                    elif left_x >= 120:
                        print("右移")
                        SSR.running_action_group('right_move_faster', 1)
                    else:
                        # SSR.running_action_group('232new', 2)
                        count += 1
                        if count >= 2:
                            print("state:hole:step 1 done, goto step 2")
                            for i in range(2):
                                SSR.running_action_group('232new', 1)
                            step = 2
                            #break

        # 2、前进
        if step == 2:
            grad = 2.5
            ksizes = [11, 7]
            defaults = [-2, (320, 361), (300, 361), (340, 361)]
            count = 0
            right_count = 0
            left_count = 0
            SSR.running_action_group('stand', 1)
            Board.setPWMServoPulse(1, 1000, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
            Board.setPWMServoPulse(2, 1400, 500)
            while step == 2:
                # 获取数据
                time.sleep(0.1)
                _, angle, points = pit_color_detect_top(
                    "green", grad, ksizes, defaults
                )
                center_x, center_y = points[0]
                print(f"step == 2: angle, center_y = {angle, center_y}")
                _, _, points = pit_color_detect_bottom(
                    "green", grad, ksizes, defaults
                )
                left_x, left_y = points[1]
                print(f"step == 2: {left_x},{left_y}): ({left_x},{left_y})")
                if angle <= -3:
                    print("右转")
                    SSR.running_action_group('turn_right_fast', 1)
                    time.sleep(0.1)

                elif angle >= 3.5:
                    print("左转")
                    SSR.running_action_group('turn_left_fast', 1)
                    time.sleep(0.1)

                else:
                    if left_x <= 130:
                        print("左移")
                        SSR.running_action_group('left_move_faster', 1)
                        # left_count += 1
                        # if left_count >= 3:
                        #     step = 3

                    elif left_x >= 150:
                        print("右移")
                        # right_count += 1
                        # if right_count >= 5:
                        #     step = 3

                        SSR.running_action_group('right_move_faster', 1)
                    else:
                        left_count = 0
                        right_count = 0
                        print("正对，前进")
                        percent = pit_ground_percent(type='bridge')
                        print(f"percent = {percent}")
                        if percent >= 0.1:
                            SSR.running_action_group('232new', 2)

                            # 进入下一步
                        else:
                            # count += 1
                            print("state:hole:step 2 done, goto step 3")
                            SSR.running_action_group('232new', 3)
                            step = 3
                            #break

        # 3、回到中间
        if step == 3:
            # 进入下一关
            Board.setPWMServoPulse(1, 1350, 100)
            Board.setPWMServoPulse(2, 550, 100)
            SSR.running_action_group('232new', 1)
            time.sleep(0.5)
            while step == 3:
                grad = 2.5
                ksizes = [11, 7]
                defaults = [-2, (320, 240), (300, 240), (340, 240)]
                _, angle, points = pit_color_detect_top(
                    "green", grad, ksizes, defaults
                )
                print(f"angle,points = {angle, points}")
                if angle <= -9:
                    print("右转")
                    SSR.running_action_group('turn_right_fast', 1)
                    time.sleep(0.1)
                elif angle >= -2.5:
                    print("左转")
                    SSR.running_action_group('turn_left_fast', 1)
                    time.sleep(0.1)
                else:
                    print("已经正对右侧绿色，下一步进行右移")
                    break
            # count = 0
            # while pit_ground_percent(type='pit_right_green') >= 0.1 and count <= 10:
            for i in range(8):
                print("需要右移")
                SSR.running_action_group('right_move_30', 1)
                # count += 1

            state = None
            break
    Board.setPWMServoPulse(1, 1000, 500)
    Board.setPWMServoPulse(2, 1450, 500)
    SSR.running_action_group('stand', 1)
    print("state:hole:exit")


# ############################################### 雷区 ######################
def adjust_angle(angle, angle_range):
    dAng = abs((angle_range[0] + angle_range[1]) / 2 - angle)
    print(f"dAng = {dAng:.2f}")
    if angle < angle_range[0]:
        if dAng > 15:
            # AGC.runAction(Actions["turn_right_L"], 1, action_path)
            print("右转2步")
            SSR.running_action_group('turn_right_fast', 2)
        # elif dAng > 8:
        #     # AGC.runAction(Actions["turn_right_m"], 1, action_path)
        #     print("右转一中步")
        #     SSR.running_action_group('203', 1)
        else:
            # AGC.runAction(Actions["turn_right_s"], 1, action_path)
            print("右转一小步")
            SSR.running_action_group('turn_right_fast', 1)
        # time.sleep(0.5)
    elif angle > angle_range[1]:
        if dAng > 15:
            # AGC.runAction(Actions["turn_left_L"], 1, action_path)
            print("左转2步")
            SSR.running_action_group('turn_left_fast', 2)
        # elif dAng > 8:
        #     # AGC.runAction(Actions["turn_left_m"], 1, action_path)
        #     print("左转一中步")
        #     SSR.running_action_group('202', 1)
        else:
            # AGC.runAction(Actions["turn_left_s"], 1, action_path)
            print("左转一小步")
            SSR.running_action_group('turn_left_fast', 1)
        # time.sleep(0.5)


def sort_contours_by_LT(cnts, method="left-to-right"):
    if not cnts:
        return ([], [])

    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if sort against y rather than x of the bounding box
    if method == "bottom-to-top" or method == "top-to-bottom":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )
    return (cnts, boundingBoxes)


def color_detect_sill(color, area_min=100, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes
    if defaults is None:
        defaults_ = [0, (320, 300), (300, 240), (340, 240)]
    else:
        defaults_ = defaults

    global frame_cali
    # while not frame_ready:
    #     time.sleep(0.01)
    # frame_ready = False
    img = frame_cali.copy()
    img_smooth = cv2.GaussianBlur(img, (3, 3), 0)
    # img_transform = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2LAB)
    img_thresh = cv2.inRange(
        img_smooth, color_range1[color][0], color_range1[color][1]
    )
    kernel = np.ones((3, 3), np.uint8)
    open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnts_A = [c for c in cnts if cv2.contourArea(c) > area_min]

    if cnts_A:
        cnts_fl = [
            c for c in cnts_A if (cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]) > 320
        ]
        if cnts_fl:
            cnts_sorted, _ = sort_contours_by_LT(cnts_fl, method="top-to-bottom")
            cnt_top = cnts_sorted[0]

            top_left = cnt_top[0][0]
            top_right = cnt_top[0][0]
            for c in cnt_top:  # 遍历找到四个顶点
                if c[0][0] + grad * c[0][1] < top_left[0] + grad * top_left[1]:
                    top_left = c[0]
                if -c[0][0] + grad * c[0][1] < -top_right[0] + grad * top_right[1]:
                    top_right = c[0]
            line_top = line(top_left, top_right)
            top_angle = line_top.angle()
            top_center = line_top.mid_point()  # 需要转换成tuple使用
        else:
            cnt_top = np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])
            top_angle, top_center, top_left, top_right = defaults_
    else:
        cnt_top = np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])
        top_angle, top_center, top_left, top_right = defaults_

    if frame_show:
        img_show = img.copy()
        put_text(img_show, f"angle:{top_angle:.2f}", (50, 100))
        put_text(img_show, f"center:{tuple(top_center)} ", (50, 150))
        put_text(img_show, f"left:{tuple(top_left)} ", (50, 200))
        put_text(img_show, f"right:{tuple(top_center)} ", (50, 250))

        cv2.drawContours(img_show, [cnt_top], -1, (0, 255, 255), 2)
        cv2.line(img_show, tuple(top_left), tuple(top_right), (255, 255, 0), 3)
        cv2.circle(img_show, tuple(top_center), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(top_left), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(top_right), 5, (255, 0, 0), 2)

        # # 显示图像
        # frames.put(img_show)

    return (top_angle, [top_center, top_left, top_right])


def color_detect_top(color, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes

    if defaults is None:
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults

    # 图像处理及roi提取
    global frame_cali
    # while not frame_ready:
    #     time.sleep(0.01)
    # frame_ready = False
    img = frame_cali.copy()
    close = get_frame_bin(img, color, ksizes_[0], ksizes_[1])
    cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # cnt_max, area_max = get_maxcontour(cnts)
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

    # 显示
    if frame_show:
        img_show = img.copy()
        put_text(img_show, f"area:{percent}%", (50, 50))
        put_text(img_show, f"angle:{top_angle:.2f}", (50, 100))
        put_text(img_show, f"center:{tuple(top_center)} ", (50, 150))
        put_text(img_show, f"left:{tuple(top_left)} ", (50, 200))
        put_text(img_show, f"right:{tuple(top_right)} ", (50, 250))

        cv2.drawContours(img_show, [cnt_max], -1, (0, 255, 255), 2)
        cv2.line(img_show, tuple(top_left), tuple(top_right), (255, 255, 0), 3)
        cv2.circle(img_show, tuple(top_center), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(top_left), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(top_right), 5, (255, 0, 0), 2)

        # # 显示图像
        # frames.put(img_show)

    return percent, top_angle, [top_center, top_left, top_right]


def color_detect_mine_near(color_back, color_obj, ksizes=None, area=None, default=None):
    if ksizes is None:
        ksizes_ = [(3, 3), (3, 3)]
    else:
        ksizes_ = ksizes

    if area is None:
        area_min = 500
        area_max = 8000
    else:
        area_min = area[0]
        area_max = area[1]

    if default is None:
        default_ = (300, 0, 30, 30)
    else:
        default_ = default

    # 提取 roi
    global frame_cali
    # while not frame_ready:
    #     time.sleep(0.01)
    # frame_ready = False
    img = frame_cali.copy()
    img_bin = get_frame_bin(img, color_back, ksizes_[0][0], ksizes_[0][1])
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnt_max, _ = get_maxcontour(cnts)
    cnt_max, _ = getAreaMaxContour(cnts)

    if cnt_max is None:
        cnt_max = np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])
        img_roi = img.copy()
    else:
        cnt_hull = cv2.convexHull(cnt_max)
        img_zero = np.zeros_like(img, np.uint8)
        img_mask = cv2.fillConvexPoly(img_zero, cnt_hull, (255, 255, 255))
        img_mask = cv2.bitwise_not(img_mask, img_mask)
        img_roi = cv2.bitwise_or(img, img_mask)

    img_mine_bin = get_frame_bin(img_roi, color_obj, ksizes_[1][0], ksizes_[1][1])
    cnts, _ = cv2.findContours(img_mine_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if cnts:
        boundingBoxs = [cv2.boundingRect(c) for c in cnts]
        mines = [box for box in boundingBoxs if area_min < box[2] * box[3] < area_max]
        if mines:
            mine_near = min(
                mines, key=lambda m: (1.2 * (m[0] - 320)) ** 2 + (m[1] - 480) ** 2
            )
        else:
            mine_near = default_
            print("default_    1")
    else:
        mine_near = default_
        print("default_    2")

    x, y, w, h = mine_near
    mine_x = x + w // 2
    mine_y = y

    if frame_show:
        img_show = img.copy()
        put_text(img_show, f"mine:({mine_x}, {mine_y})", (50, 50))
        put_text(img_show, f"area:{w * h}", (50, 100))

        cv2.drawContours(img_show, [cnt_max], -1, (0, 255, 255), 2)
        cv2.rectangle(img_show, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.circle(img_show, (mine_x, mine_y), 5, (255, 0, 0), 2)

        # # 显示图像
        # frames.put(img_show)

    return (mine_x, mine_y)


def state_mine():
    print("state:mine:enter")
    state = 'mine'
    self_flag = 0
    camera_flag = 0
    self_x = 320
    go_num = 0
    go_num_max = 3

    step = 0
    SSR.running_action_group('0', 1)  # 初始化动作
    while state == 'mine':
        if step == -1:
            grad = 2.5
            ksizes = [11, 7]
            defaults = [0, (320, 410), (300, 410), (340, 410)]

            Board.setPWMServoPulse(1, 1050, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
            Board.setPWMServoPulse(2, 1450, 500)
            # init_camera_platform(500, 400)
            while step == -1:
                # angle, points = color_detect_sill(
                #     colors["next"], grad, ksizes[0], defaults
                # )
                # mine_x, mine_y = color_detect_mine_near(
                #     colors["this"][1], colors["obj"], ksizes
                # )

                #
                angle, points = color_detect_sill(
                    "blue_mine", 400, grad, ksizes, defaults
                )
                # color_detect_top(colors["this"][0], grad, ksizes, defaults)

        # 0、抬头调整
        if step == 0:
            grad = 2.5
            ksizes = [11, 5]
            defaults = [0, (320, 410), (300, 410), (340, 410)]

            while step == 0:
                # 调整相机高度
                if camera_flag == 0:
                    angle_range = [-2, 2]
                    center_y_switch = 220
                    area_min = 400
                    color = "blue_mine"
                    print("state:mine:camera:A")

                    Board.setPWMServoPulse(1, 1150, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
                    Board.setPWMServoPulse(2, 1450, 500)
                    # init_camera_platform(500, 500)  # 头部左右、上下
                    time.sleep(0.5)

                if camera_flag == 1:
                    angle_range = [-2, 2]
                    center_y_switch = 250
                    area_min = 1000
                    color = "blue_mine"
                    print("state:mine:camera:B")

                    Board.setPWMServoPulse(1, 1150, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
                    Board.setPWMServoPulse(2, 1450, 500)
                    # init_camera_platform(500, 500)  # 头部左右、上下
                    time.sleep(0.5)

                if camera_flag == 2:
                    angle_range = [-2, 2]
                    center_y_switch = 210
                    area_min = 1000
                    color = "blue_mine"
                    print("state:mine:camera:C")

                    Board.setPWMServoPulse(1, 1050, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
                    Board.setPWMServoPulse(2, 1450, 500)
                    # init_camera_platform(500, 400)  # 头部左右、上下
                    time.sleep(0.5)

                print("--------------- mine:step 0 ---------------")

                # 获取数据
                if camera_flag == 0:
                    # color_detect_top() 返回目标区域占比, 顶部角度, 顶部中点[x, y], 顶部左点[x, y], 顶部右点[x, y]
                    _, angle, points = color_detect_top(
                        "white_road", grad, ksizes, defaults
                    )
                elif camera_flag == 1 or camera_flag == 2:
                    angle, points = color_detect_sill(
                        color, area_min, grad, ksizes, defaults
                    )

                center_x, center_y = points[0]
                if center_x < 300:
                    self_flag = 0
                else:
                    self_flag = 1
                self_x = 320 - center_x

                # 调整角度
                if not angle_range[0] < angle < angle_range[1]:
                    adjust_angle(angle, angle_range)
                    continue

                # 判断是否低头
                if camera_flag == 0 and center_y > center_y_switch:
                    camera_flag = 1
                    continue

                # 进入下一步 2
                if camera_flag == 1 and center_y > center_y_switch:
                    camera_flag = 2
                    continue

                if camera_flag == 2 and center_y > center_y_switch:
                    print("state:mine:step 0 done, goto step 2")
                    step = 2

                # 进入下一步 1
                print("state:mine:step 0 done, goto step 1")
                step = 1
                break

        # 1、低头前进
        if step == 1:
            grad = 2.5
            ksizes = [(11, 7), (11, 7)]
            defaults = [-2, (320, 0), (300, 0), (340, 0)]
            mine_y_bigbig = 100
            mine_y_big = 215
            mine_y_little = 265
            mine_y_stop = 365
            mine_x_range = [105, 500]
            mine_x_switch = 70  # 65#70
            m_x_b_range = [-80, 80]
            left_flag = False

            # 低头看地雷
            Board.setPWMServoPulse(1, 1050, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
            Board.setPWMServoPulse(2, 1450, 500)
            # init_camera_platform(500, 350)  # 头部左右、上下
            time.sleep(0.25)
            while step == 1:
                if go_num >= go_num_max:
                    step = 0
                    go_num = 0
                    break

                print("--------------- mine:step 1 ---------------")
                # 识别地雷
                time.sleep(0.5)
                mine_x, mine_y = color_detect_mine_near(
                    "white_mine", "black", ksizes
                )
                print(f"mine: {mine_x}, {mine_y}")
                if mine_y == 0:
                    step = 2
                    # AGC.runAction(Actions["go_LL"], 1, action_path)

                    SSR.running_action_group('239new', 1)
                    print("往前走一步")

                    break

                mine_x_self = mine_x - 315
                mine_x_base = 1.8 * self_x + 0.5 * mine_x_self
                print(f"mine_x_self: {mine_x_self:.2f}")
                print(f"mine_x_base: {mine_x_base:.2f}")

                if mine_y < mine_y_bigbig:
                    print("state:mine:1.1")
                    # AGC.runAction(Actions["go_LL"], 1, action_path)

                    SSR.running_action_group('239new', 1)
                    print("往前走一步")

                    go_num += 1
                    continue

                if mine_y < mine_y_big:
                    print("state:mine:1.2")
                    # AGC.runAction(Actions["go_L"], 1, action_path)

                    SSR.running_action_group('232new', 1)
                    print("往前走一步")

                    go_num += 1
                    continue

                if mine_y < mine_y_little:
                    print("state:mine:2.1")
                    # AGC.runAction(Actions["go_s"], 2, action_path)

                    SSR.running_action_group('239new', 2)
                    print("往前走2步")

                    go_num += 1
                    continue

                if mine_y < mine_y_stop:
                    print("state:mine:2.2")
                    # AGC.runAction(Actions["go_s"], 1, action_path)

                    SSR.running_action_group('232new', 1)
                    print("往前走1步")

                    go_num += 1
                    continue

                if not m_x_b_range[0] < mine_x_base < m_x_b_range[1]:
                    if mine_x_range[0] < mine_x < mine_x_range[1] and self_flag == 0:
                        print("state:mine:3.1")
                        left_flag = True
                        if abs(mine_x_self) < mine_x_switch:
                            print("state:mine:3.1.1")
                            # AGC.runAction(Actions["left_go_L"], 1, action_path)

                            SSR.running_action_group('left_move_fast', 1)
                            print("往左移2步")
                            time.sleep(0.2)

                        else:
                            print("state:mine:3.1.2")
                            # AGC.runAction(Actions["left_go_m"], 1, action_path)

                            SSR.running_action_group('left_move_fast', 1)
                            print("往左移1小步")
                            time.sleep(0.2)

                        continue

                    if mine_x_range[0] < mine_x < mine_x_range[1] and self_flag == 1:
                        print("state:mine:4.1")
                        left_flag = True
                        if abs(mine_x_self) < mine_x_switch:
                            print("state:mine:4.1.1")
                            # AGC.runAction(Actions["right_go_L"], 1, action_path)

                            SSR.running_action_group('161_test', 1)
                            print("往右移2步")

                        else:
                            print("state:mine:4.1.2")
                            # AGC.runAction(Actions["right_go_m"], 1, action_path)

                            SSR.running_action_group('161_test', 1)
                            print("往右移1小步")

                        continue
                else:
                    if mine_x_range[0] < mine_x < mine_x_range[1] and mine_x > 315:
                        print("state:mine:3.2")
                        left_flag = True
                        if abs(mine_x_self) < mine_x_switch:
                            print("state:mine:3.2.1")
                            # AGC.runAction(Actions["left_go_L"], 1, action_path)

                            SSR.running_action_group('left_move_fast', 1)
                            time.sleep(0.2)
                            print("往左移2步")

                        else:
                            print("state:mine:3.2.2")
                            # AGC.runAction(Actions["left_go_m"], 1, action_path)

                            SSR.running_action_group('left_move_fast', 1)
                            time.sleep(0.2)
                            print("往左移1小步")

                        continue

                    if mine_x_range[0] < mine_x < mine_x_range[1] and mine_x < 316:
                        print("state:mine:4.2")
                        left_flag = True
                        if abs(mine_x_self) < mine_x_switch:
                            print("state:mine:4.2.1")
                            # AGC.runAction(Actions["right_go_L"], 1, action_path)

                            SSR.running_action_group('161old', 1)
                            print("往右移2步")

                        else:
                            print("state:mine:4.2.2")
                            # AGC.runAction(Actions["right_go_m"], 1, action_path)

                            SSR.running_action_group('161old', 1)
                            print("往右移1小步")

                        continue

                if not mine_x_range[0] < mine_x < mine_x_range[1] and left_flag:
                    step = 0
                    left_flag = False
                    continue

                print("state:mine:5")
                print("state:mine:step 1 done, goto step 0")
                # AGC.runAction(Actions["go_L"], 1, action_path)

                SSR.running_action_group('239new', 1)
                print("前进一大步")

        # 2、衔接
        if step == 2:
            grad = 2.5
            ksizes = [11, 7]
            defaults = [-2, (320, 240), (300, 240), (340, 240)]
            area_min = 1000
            angle_range = [-6, 2]
            center_y_stop = 300

            # 低头
            # init_camera_platform(500, 300)  # 头部左右、上下

            Board.setPWMServoPulse(1, 1050, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
            Board.setPWMServoPulse(2, 1450, 500)

            time.sleep(0.25)
            while step == 2:
                print("--------------- mine:step 2 ---------------")
                time.sleep(0.5)
                angle, points = color_detect_sill(
                    "blue_mine", area_min, grad, ksizes, defaults
                )
                _, center_y = points[0]

                # 调整角度
                if not angle_range[0] < angle < angle_range[1]:
                    adjust_angle(angle, angle_range)
                    continue

                if center_y < center_y_stop:
                    # AGC.runAction(Actions["go_s"], 1, action_path)

                    SSR.running_action_group('239new', 1)
                    print("前进一小步")

                    continue

                print("state:mine:exit")
                # state = StateTrans[State.mine]
                state = 'other'
                SSR.running_action_group('stand', 1)
                break


# ############################################### 挡板 ######################
def baffler():
    global state
    Board.setPWMServoPulse(1, 1000, 500)
    Board.setPWMServoPulse(2, 1450, 500)
    SSR.running_action_group('stand', 1)
    # SSR.running_action_group('turn_left', 1)
    SSR.running_action_group('232new', 3)
    time.sleep(0.2)
    # SSR.running_action_group('turn_right', 2)
    SSR.running_action_group('stand', 1)
    # 放进翻越动作
    SSR.running_action_group('kuayueplus3', 1)
    SSR.running_action_group('stand', 1)
    # 左转身一定的次数
    # turn_flag = False
    turn_flag = False
    count = 0
    if turn_flag:
        while True:
            print("左转")
            SSR.running_action_group('turn_left_fast', 1)
            count += 1
            time.sleep(0.5)
            if count >= 15:
                break
    SSR.running_action_group("stand", 1)
    print("state:baffler:exit")


# ############################################### 门框 ######################
def door_color_detect_sill(color, area_min=1000, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes
    if defaults is None:
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults

    global frame_cali
    # while not frame_ready:
    #     time.sleep(0.01)
    # frame_ready = False
    img = frame_cali.copy()
    img_bin = get_frame_bin(img, color, ksizes_[0], ksizes_[1])
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnts_A = [c for c in cnts if cv2.contourArea(c) > area_min]

    if cnts_A:
        cnts_fl = [
            c for c in cnts_A if (cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]) > 320
        ]
        if cnts_fl:
            cnts_sorted, _ = sort_contours_by_LT(cnts_fl, method="top-to-bottom")
            cnt_top = cnts_sorted[0]

            top_left = cnt_top[0][0]
            top_right = cnt_top[0][0]
            for c in cnt_top:  # 遍历找到四个顶点
                if c[0][0] + grad * c[0][1] < top_left[0] + grad * top_left[1]:
                    top_left = c[0]
                if -c[0][0] + grad * c[0][1] < -top_right[0] + grad * top_right[1]:
                    top_right = c[0]
            line_top = line(top_left, top_right)
            top_angle = line_top.angle()
            top_center = line_top.mid_point()  # 需要转换成tuple使用
        else:
            cnt_top = np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])
            top_angle, top_center, top_left, top_right = defaults_
    else:
        cnt_top = np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])
        top_angle, top_center, top_left, top_right = defaults_

    if frame_show:
        img_show = img.copy()
        put_text(img_show, f"angle:{top_angle:.2f}", (50, 100))
        put_text(img_show, f"center:{tuple(top_center)} ", (50, 150))
        put_text(img_show, f"left:{tuple(top_left)} ", (50, 200))
        put_text(img_show, f"right:{tuple(top_center)} ", (50, 250))

        cv2.drawContours(img_show, [cnt_top], -1, (0, 255, 255), 2)
        cv2.line(img_show, tuple(top_left), tuple(top_right), (255, 255, 0), 3)
        cv2.circle(img_show, tuple(top_center), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(top_left), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(top_right), 5, (255, 0, 0), 2)

    return (top_angle, [top_center, top_left, top_right])


def door_color_detect_door(color_back, color_obj, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = [(3, 3), (3, 3)]
    else:
        ksizes_ = ksizes
    if defaults is None:
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults

    global frame_cali
    # while not frame_ready:
    #     time.sleep(0.01)
    # frame_ready = False
    img = frame_cali.copy()
    img_bin = get_frame_bin(img, color_back, ksizes_[0][0], ksizes_[0][1])
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnt_max, _ = get_maxcontour(cnts)
    cnt_max, _ = getAreaMaxContour(cnts)

    if cnt_max is None:
        cnt_hull = np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])
        cnts_max_two = [np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])]
        angle, center, left, right = defaults_
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

        cnt_hull = cv2.convexHull(cnt_max)
        img_zero = np.zeros_like(img, np.uint8)
        img_mask = cv2.fillConvexPoly(img_zero, cnt_hull, (255, 255, 255))
        img_roi = cv2.bitwise_and(img, img_mask)

        img_roi_bin = get_frame_bin(img_roi, color_obj, ksizes_[1][0], ksizes_[1][1])
        cnts_roi, _ = cv2.findContours(
            img_roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if len(cnts_roi) < 2:
            cnts_max_two = [np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])]
            angle, center, left, right = defaults_
        else:
            cnts_fl_area = [c for c in cnts_roi if cv2.contourArea(c) > 1200]
            cnts_fl_y = [
                c
                for c in cnts_fl_area
                # if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] < 480
                if cv2.boundingRect(c)[1] < 400
            ]
            if len(cnts_fl_y) < 2:
                cnts_max_two = [
                    np.array([[[0, 0]], [[639, 0]], [[639, 479]], [[0, 479]]])
                ]
                angle, center, left, right = defaults_
            else:
                cnts_max_two, _ = sort_contours_by_A(cnts_fl_y, True)
                cnts_sorted, _ = sort_contours_by_LT(cnts_max_two[:2])
                cnt_left, cnt_right = cnts_sorted

                # 左侧轨迹右下点
                left = cnt_left[0][0]
                for p in cnt_left:
                    if p[0][0] + grad * p[0][1] > left[0] + grad * left[1]:
                        left = p[0]

                # 右侧轨迹左下点
                right = cnt_right[0][0]
                for p in cnt_right:
                    if -p[0][0] + grad * p[0][1] > -right[0] + grad * right[1]:
                        right = p[0]

                line_door = line(left, right)
                angle = line_door.angle()
                center = line_door.mid_point()

    if frame_show:
        img_show = img.copy()
        put_text(img_show, f"angle:{angle:.2f}", (50, 100))
        put_text(img_show, f"center:{tuple(center)} ", (50, 150))
        put_text(img_show, f"left:{tuple(left)} ", (50, 200))
        put_text(img_show, f"right:{tuple(center)} ", (50, 250))

        cv2.polylines(img_show, [cnt_hull], True, (0, 255, 255), 2)
        cv2.drawContours(img_show, cnts_max_two, -1, (255, 0, 255), 2)
        cv2.line(img_show, tuple(left), tuple(right), (255, 255, 0), 3)
        cv2.circle(img_show, tuple(center), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(left), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(right), 5, (255, 0, 0), 2)

    return ([angle, top_angle], [center, left, right, top_center])


def door_color_detect_top(color, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes

    if defaults is None:
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults

    # 图像处理及roi提取
    global frame_cali
    # while not frame_ready:
    #     time.sleep(0.01)
    # frame_ready = False
    # frame_cali = cv2.imread("key frame/images/4858.jpg")
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

    # 显示
    if frame_show:
        img_show = img.copy()
        put_text(img_show, f"area:{percent}%", (50, 50))
        put_text(img_show, f"angle:{top_angle:.2f}", (50, 100))
        put_text(img_show, f"center:{tuple(top_center)} ", (50, 150))
        put_text(img_show, f"left:{tuple(top_left)} ", (50, 200))
        put_text(img_show, f"right:{tuple(top_right)} ", (50, 250))

        cv2.drawContours(img_show, [cnt_max], -1, (0, 255, 255), 2)
        cv2.line(img_show, tuple(top_left), tuple(top_right), (255, 255, 0), 3)
        cv2.circle(img_show, tuple(top_center), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(top_left), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(top_right), 5, (255, 0, 0), 2)

        # 显示图像
        # frames.put(img_show)
        # cv2.drawContours(img_show, [box], -1, (0, 255, 0), 2)
        # cv2.imshow("Result", img_show)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return percent, top_angle, [top_center, top_left, top_right]


def door_color_detect_bottom(color, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes

    if defaults is None:
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults

    # 图像处理及roi提取
    global frame_cali
    # while not frame_ready:
    #     time.sleep(0.01)
    # frame_ready = False
    # frame_cali = cv2.imread("key frame/images/4858.jpg")
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

    if frame_show:
        img_show = img.copy()
        put_text(img_show, f"area:{percent}%", (50, 50))
        put_text(img_show, f"angle:{bottom_angle:.2f}", (50, 100))
        put_text(img_show, f"center:{tuple(bottom_center)} ", (50, 150))
        put_text(img_show, f"left:{tuple(bottom_left)} ", (50, 200))
        put_text(img_show, f"right:{tuple(bottom_center)} ", (50, 250))

        cv2.drawContours(img_show, [cnt_max], -1, (0, 255, 255), 2)
        cv2.line(img_show, tuple(bottom_left), tuple(bottom_right), (255, 255, 0), 3)
        cv2.circle(img_show, tuple(bottom_center), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(bottom_left), 5, (255, 0, 0), 2)
        cv2.circle(img_show, tuple(bottom_right), 5, (255, 0, 0), 2)

        # 显示图像
        # frames.put(img_show)
        # cv2.imshow("Result", img_show)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return percent, bottom_angle, [bottom_center, bottom_left, bottom_right]


def door_mid_center(type):
    # global org_img
    global frame_cali
    org_img = frame_cali.copy()
    time.sleep(0.1)
    img = org_img[:350, :]
    # org_img_copy = cv2.resize(img, (r_w, r_h), interpolation=cv2.INTER_CUBIC)
    frame_gauss = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊
    frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    frame_floor_blue = cv2.inRange(frame_hsv, color_range[type][0],
                                   color_range[type][1])  # 对原图像和掩模(颜色的字典)进行位运算
    # cv2.imshow('test', frame_floor_blue)
    contours_pit_hole, hierarchy = cv2.findContours(frame_floor_blue, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)
    # areaMaxContour = getAreaMaxContour(contours_pit_hole)[-2]  # 找出最大轮廓
    areaMaxContour, area_max = getAreaMaxContour(contours_pit_hole)
    if areaMaxContour is not None:
        rect = cv2.minAreaRect(areaMaxContour)
        box = np.int0(cv2.boxPoints(rect))  # 点的坐标
        # cv2.drawContours(org_img, [box], -1, (0, 255, 0), 2)
        # cv2.imshow("Result", org_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        center_x, center_y = rect[0]

    else:
        center_x = 0
        center_y = 0
    print(f"center_x{center_x, center_y}")
    return center_x, center_y, area_max


def door_percent(type):
    """
    计算坑绿色地percent，判断是否走完
    """
    global frame_cali
    org_img = frame_cali.copy()
    # org_img = cv2.imread("key frame/images/2477.jpg")
    # img = org_img[200:, :]
    r_h = org_img.shape[0]
    r_w = org_img.shape[1]

    gauss = cv2.GaussianBlur(org_img, (3, 3), 0)  # 高斯模糊
    hsv = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    Imask0 = cv2.inRange(hsv, color_range[type][0],
                         color_range[type][1])  # 对原图像和掩模(颜色的字典)进行位运算
    # opened = cv2.morphologyEx(Imask0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接
    # erode = cv2.erode(Imask0, None, iterations=1)  # 腐蚀
    # dilate = cv2.dilate(Imask0, np.ones((3, 3), np.uint8), iterations=2)  # 膨胀

    contours, hierarchy = cv2.findContours(Imask0, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
    area = getAreaSumContour(contours)  # 绿色的面积，不是提取最大轮廓的面积
    # percent0 = round(area * 100 / (r_w * r_h), 2)  # 绿色的面积百分比（绿色的面积，不是提取最大轮廓的面积）
    # areaMaxContour, area_max = getAreaMaxContour(contours)  # 找出最大轮廓
    # cv2.drawContours(org_img, contours, -1, (255, 0, 255), 2)
    # cv2.imshow('o', org_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    percent = round(100 * area / (r_w * r_h), 2)  # 最大轮廓最小外接矩形的面积占比
    print(f"green_percent = {percent}")
    return percent


def door():
    count = 0
    turn_flag = False
    # turn_flag = True
    if turn_flag:
        while True:
            print("左转")
            SSR.running_action_group('turn_left_fast', 1)
            count += 1
            if count >= 12:
                break
    print("state:door:enter")
    SSR.running_action_group('stand', 1)
    # global state
    state = 'door'
    step = -2
    peh_turn_left_flag0 = False
    peh_turn_right_flag1 = False
    turn_right_flag0 = False
    turn_left_flag1 = False
    count_ = 0
    while state == 'door':
        # 根据赛道边缘，使机器人处于赛道中间
        if step == -2:
            Board.setPWMServoPulse(1, 1050, 500)
            Board.setPWMServoPulse(2, 1450, 500)
            time.sleep(0.5)
            while step == -2:
                grad = 2.5
                ksizes = [11, 7]
                defaults = [-2, (320, 240), (300, 240), (340, 240)]
                _, top_angle, top_points = door_color_detect_top(
                    "white_road", grad, ksizes, defaults
                )
                top_cen_x, top_cen_y = top_points[0]
                print(f"top_angle, top_cen_x, top_cen_y = {top_angle, top_cen_x, top_cen_y}")
                # 根据该直线的倾斜角度进行纠正
                if top_angle <= -3:
                    print("右转")
                    SSR.running_action_group('turn_right_fast', 1)
                    time.sleep(0.1)
                elif top_angle >= 3.5:
                    print("左转")
                    SSR.running_action_group('turn_left_fast', 1)
                    time.sleep(0.1)
                else:
                    print("正对边缘直线，下一步根据top_cen_y，进行前后预调整")
                    if top_cen_y <= 130:
                        print("往走一点")
                        SSR.running_action_group('232new', 1)
                    elif top_cen_y >= 170:
                        for i in range(4):
                            SSR.running_action_group('back_fast', 1)
                        time.sleep(0.5)

                        print("后退")
                    else:
                        print("step=-1 exit!")
                        step = -1
        # 左转90°
        if step == -1:
            SSR.running_action_group('stand', 1)
            while True:
                print("左转90°")
                SSR.running_action_group('turn_left_fast', 1)
                count += 1
                if count >= 10:
                    step = 0
                    break

        if step == 0:
            Board.setPWMServoPulse(1, 1200, 500)  # 头部抬高,居中
            Board.setPWMServoPulse(2, 1400, 500)
            time.sleep(0.5)

            while step == 0:
                time.sleep(0.5)
                grad = 1.5
                ksizes = [(11, 7), (11, 7)]
                defaults = [-2, [320, 385], [300, 385], [340, 385]]
                _, points = door_color_detect_door(
                    "white_door", "blue_door", grad, ksizes, defaults
                )
                grad = 2.5
                ksizes = [11, 7]
                defaults = [-2, (320, 240), (300, 240), (340, 240)]
                _, angle, top_points = door_color_detect_top(
                    "white_road", grad, ksizes, defaults
                )
                # angle, top_angle = angles
                center_x, _ = points[0]
                # top_center_x, top_center_y = points[-1]
                # center_x, center_y = points[0]
                print(f"top_angle: {angle:.2f}  ({center_x})")
                if angle <= -3:
                    print("右转")
                    SSR.running_action_group('turn_right_fast', 1)
                    time.sleep(0.1)
                elif angle >= 3.5:
                    print("左转")
                    SSR.running_action_group('turn_left_fast', 1)
                    time.sleep(0.1)
                else:
                    print("正对完成,下一步根据门框中心坐标center_x进行调整,使其居中门框")
                    print("step=0,exit!")
                    if center_x <= 270:
                        print("左移")
                        SSR.running_action_group('left_move_fast', 1)
                        time.sleep(0.1)
                    elif center_x >= 310:
                        print("右移")
                        SSR.running_action_group('161_test', 1)
                        time.sleep(0.1)
                    else:
                        step = 1  ################################################################

        if step == 1:
            '''
            根据门框最大面积,判断
            '''
            count = 0
            Board.setPWMServoPulse(1, 950, 500)  # 头部抬头,居中
            Board.setPWMServoPulse(2, 1400, 500)
            time.sleep(0.5)
            while True:
                center_x, center_y, area_max = door_mid_center('door')
                print(f"area_max = {area_max}")
                if area_max <= 13000:
                    print("前进")
                    SSR.running_action_group('232new', 1)
                    time.sleep(0.1)
                else:
                    count += 1
                if count >= 2:
                    print("靠近完成,下一步,在门框近点正前方进行正对角度调整")
                    print("step=2,exit!")
                    step = 2
                    break

        if step == 2:
            Board.setPWMServoPulse(1, 1050, 500)  # 头部抬高,居中
            Board.setPWMServoPulse(2, 1400, 500)
            time.sleep(0.5)
            # grad = 1.5
            # ksizes = [(15, 11), (11, 7)]
            # defaults = [-2, [320, 385], [300, 385], [340, 385]]
            while step == 2:
                time.sleep(0.5)
                grad = 2.5
                ksizes = [11, 7]
                defaults = [-2, (320, 240), (300, 240), (340, 240)]
                _, angle, top_points = door_color_detect_top(
                    "white_road", grad, ksizes, defaults
                )
                # angle, top_angle = angles
                # center_x, _ = points[0]
                # top_center_x, top_center_y = points[-1]
                # center_x, center_y = points[0]
                print(f"angle: {angle:.2f} ")
                if angle <= -3.5:
                    print("右转")
                    SSR.running_action_group('turn_right_fast', 1)
                    time.sleep(0.1)
                elif angle >= 3:
                    print("左转")
                    SSR.running_action_group('turn_left_fast', 1)
                    time.sleep(0.1)
                else:
                    print("正对完成,下一步根据门框中心坐标center_x进行调整,使其居中门框")
                    print("step=3,exit!")
                    step = 3  ##############################################################

        if step == 3:
            count__ = 0
            while step == 3:
                percent = door_percent(type='bridge')
                print(f'percent ={percent}')
                if percent <= 18:
                    print("前进")
                    SSR.running_action_group('239new_door', 1)
                    time.sleep(0.1)
                    count__ = 0
                else:
                    count__ += 1
                    if count__ >= 2:
                        print("前进完成,已通过门框")
                        print("step=3,exit!")
                        state = None
                        break
    SSR.running_action_group('stand', 1)


# ############################################### 独木桥 ######################
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
                print(
                    f"color_detect_top:  top_angle, top_center_x, top_center_y = {top_angle, top_center_x, top_center_y}")

                _, bot_angle, points = bridge_color_detect_bottom(
                    "green", grad, ksizes, defaults
                )
                bot_center_x, bot_center_y = points[0]
                print(
                    f"color_detect_bottom : bot_angle, bot_center_x, bot_center_y = {bot_angle, bot_center_x, bot_center_y}")
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
                print(
                    f"color_detect_bottom :bot_angle, bot_center_x, bot_center_y = {bot_angle, bot_center_x, bot_center_y}")

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
                print(
                    f"color_detect_top:  top_angle, top_center_x, top_center_y = {top_angle, top_center_x, top_center_y}")
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


# ############################################### 球转角 ######################
def ball_land_ground_percent(type):
    global org_img
    # img = org_img[200:, :]
    r_h = org_img.shape[0]
    r_w = org_img.shape[1]

    gauss = cv2.GaussianBlur(org_img, (3, 3), 0)  # 高斯模糊
    hsv = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    Imask0 = cv2.inRange(hsv, color_range[type][0],
                         color_range[type][1])  # 对原图像和掩模(颜色的字典)进行位运算
    # opened = cv2.morphologyEx(Imask0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接
    # erode = cv2.erode(Imask0, None, iterations=1)  # 腐蚀
    dilate = cv2.dilate(Imask0, np.ones((3, 3), np.uint8), iterations=2)  # 膨胀

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
    area = getAreaSumContour(contours)  # 绿色的面积，不是提取最大轮廓的面积
    # percent0 = round(area * 100 / (r_w * r_h), 2)  # 绿色的面积百分比（绿色的面积，不是提取最大轮廓的面积）
    areaMaxContour, area_max = getAreaMaxContour(contours)  # 找出最大轮廓
    # cv2.drawContours(org_img, areaMaxContour, -1, (255, 0, 255), 2)
    # cv2.imshow('o', org_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    percent = round(100 * area_max / (r_w * r_h), 2)  # 最大轮廓最小外接矩形的面积占比
    print(f"green_percent = {percent}")
    return percent


def ball_land_color_detect_top(color, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes

    if defaults is None:
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults
    # frame_cali = cv2.imread("key frame/images/0667.jpg")
    # time.sleep(0.5)
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


def ball_land_color_detect_bottom(color, grad=1.0, ksizes=None, defaults=None):
    if ksizes is None:
        ksizes_ = (3, 3)
    else:
        ksizes_ = ksizes

    if defaults is None:
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults
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


def ball():
    print("state:kick_ball:enter")
    global org_img
    global next_state
    global state
    next_state = 'pit'
    state = 'pit'
    if next_state == 'pit':
        step = 0
        while state == 'pit':

            # 获取数据
            if step == -1:
                SSR.running_action_group('stand', 1)
                Board.setPWMServoPulse(1, 1100, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
                Board.setPWMServoPulse(2, 1400, 500)

                while step == -1:
                    time.sleep(0.25)
                    grad = 2.5
                    ksizes = [11, 7]
                    defaults = [-2, (320, 240), (300, 240), (340, 240)]
                    _, angle, points = ball_land_color_detect_top(
                        "green", grad, ksizes, defaults
                    )
                    center_x, center_y = points[0]
                    print(f"angle,top_center_x, top_center_y = {angle, center_x, center_y}")
                    print("------------------------")

            # 抬头,从远处正对赛道边缘
            if step == 0:
                Board.setPWMServoPulse(1, 1200, 500)  # 头部抬高
                Board.setPWMServoPulse(2, 1400, 500)
                time.sleep(0.5)
                count = 0
                print("开始远处正对赛道边缘")
                while step == 0:
                    grad = 2.5
                    ksizes = [11, 5]
                    defaults = [0, (320, 410), (300, 410), (340, 410)]
                    # _, angle, points = ball_land_color_detect_top(
                    # "red_brick", grad, ksizes, defaults
                    # )
                    # 红砖赛道最顶端的直线
                    _, angle, points = ball_land_color_detect_top(
                        "red", grad, ksizes, defaults
                    )
                    _, top_center_y = points[0]
                    print(f"angle,top_center_y = {angle, top_center_y}")
                    # 根据该直线的倾斜角度进行纠正
                    if angle <= -4:
                        print("右转")
                        SSR.running_action_group('turn_right_fast', 1)
                        time.sleep(0.1)
                    elif angle >= 3.5:
                        print("左转")
                        SSR.running_action_group('turn_left_fast', 1)
                        time.sleep(0.1)
                    else:
                        if top_center_y <= 260:
                            print("前进,直到进入那个格子")
                            SSR.running_action_group('239new', 2)
                            # time.sleep(0.1)
                        else:
                            count += 1
                            if count >= 2:
                                print("已经基本正对赛道边缘,进行下一步检测球坐标,半径")
                                step = 1
                                break

            # 低头,从近处正对赛道边缘
            if step == 1:
                Board.setPWMServoPulse(1, 1000, 500)
                Board.setPWMServoPulse(2, 1400, 500)
                time.sleep(0.5)
                count = 0
                print("开始近处正对赛道边缘")
                while step == 1:
                    grad = 2.5
                    ksizes = [11, 5]
                    defaults = [0, (320, 410), (300, 410), (340, 410)]
                    # _, angle, points = color_detect_top(
                    # "red_brick", grad, ksizes, defaults
                    # )
                    _, angle, points = ball_land_color_detect_top(
                        "red", grad, ksizes, defaults
                    )
                    _, top_center_y = points[0]
                    print(f"angle,top_center_y = {angle, top_center_y}")
                    if angle <= -3:
                        print("右转")
                        SSR.running_action_group('turn_right_fast', 1)
                        time.sleep(0.1)
                    elif angle >= 2.8:
                        print("左转")
                        SSR.running_action_group('turn_left_fast', 1)
                        time.sleep(0.1)
                    else:
                        if top_center_y <= 100:
                            print("前进,kao")
                            SSR.running_action_group('232new', 1)
                            time.sleep(0.1)
                        else:
                            count += 1
                            if count >= 2:
                                print("已经基本正对赛道边缘,进行下一步检测球坐标,半径")
                                step = 2
            # 3、左转
            if step == 2:
                # 进入下一关
                print("左转一定角度")
                for i in range(12):
                    SSR.running_action_group('turn_left_fast', 1)
                    time.sleep(0.1)
                step = 3

            # 头部右转90度，根据右边赛道边缘纠正
            if step == 3:
                Board.setPWMServoPulse(1, 1300, 500)
                Board.setPWMServoPulse(2, 2300, 500)
                time.sleep(0.5)
                count = 0
                while step == 3:
                    grad = 2.5
                    ksizes = [11, 7]
                    defaults = [-2, (320, 240), (300, 240), (340, 240)]
                    _, angle, points = ball_land_color_detect_top(
                        "red", grad, ksizes, defaults
                    )
                    grad = 2.5
                    ksizes = [11, 5]
                    defaults = [0, (320, 410), (300, 410), (340, 410)]
                    # _, angle, points = ball_land_color_detect_top(
                    # "red_brick", grad, ksizes, defaults
                    # )
                    _, angle, points = ball_land_color_detect_top(
                        "red", grad, ksizes, defaults
                    )
                    top_center_x, top_center_y = points[0]
                    print(f"angle,top_center_x, top_center_y = {angle, top_center_x, top_center_y}")
                    if angle <= -4:
                        print("右转")
                        SSR.running_action_group('turn_right_fast', 1)
                        time.sleep(0.1)
                    elif angle >= 3.5:
                        print("左转")
                        SSR.running_action_group('turn_left_fast', 1)
                        time.sleep(0.1)
                    else:
                        print("已经正对右侧红色地面，根据左侧的y值，纠正左右移动")
                        if top_center_y <= 130:
                            print("左移")
                            SSR.running_action_group('left_move_fast', 1)
                            time.sleep(0.1)
                        elif top_center_y >= 200:
                            print("右移")
                            SSR.running_action_group('right_move', 1)
                            time.sleep(0.1)
                        else:
                            if top_center_x >= 500:
                                print("前进")
                                SSR.running_action_group('232new', 1)
                            else:
                                count += 1
                                if count >= 1:
                                    print("已经基本正对赛道边缘,进行下一关卡")
                                    step = 2
                                    state = None
                                    break

        Board.setPWMServoPulse(1, 1050, 500)
        Board.setPWMServoPulse(2, 1450, 500)
        SSR.running_action_group('stand', 1)
        print("state:hole:exit")
    else:
        print("next_state 不是 pit，有错")
        # 再写一个，当next_state不对的，但执行该程序的东西


# ############################################### 台阶 ######################

def floor_main_subfunction(type):
    global orgFrame
    global debug
    global org_img
    time.sleep(1)
    skip = 0
    img = org_img
    r_h = img.shape[0]
    r_w = img.shape[1]

    time.sleep(0.1)
    org_img_copy = cv2.resize(org_img, (r_w, r_h), interpolation=cv2.INTER_CUBIC)
    frame_gauss = cv2.GaussianBlur(org_img_copy, (3, 3), 0)  # 高斯模糊
    frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    frame_floor_blue = cv2.inRange(frame_hsv, color_range[type][0],
                                   color_range[type][1])  # 对原图像和掩模(颜色的字典)进行位运算
    # cv2.imshow('test', frame_floor_blue)
    contours_pit_hole, hierarchy = cv2.findContours(frame_floor_blue, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)
    areaMaxContour = getAreaMaxContour(contours_pit_hole)[-2]  # 找出最大轮廓
    rect = cv2.minAreaRect(areaMaxContour)
    box = np.int0(cv2.boxPoints(rect))  # 点的坐标
    w = math.sqrt(pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2))
    h = math.sqrt(pow(box[0][0] - box[3][0], 2) + pow(box[0][1] - box[3][1], 2))
    area_max = w * h  # 蓝色台阶的面积
    percent = round(area_max * 100 / (r_w * r_h), 2)  # 最大轮廓百分比
    # center, w_h, angle = rect  # 中心点 宽高 旋转角度
    # box = np.int0(cv2.boxPoints(rect))  # 点的坐标
    top_right = areaMaxContour[0][0]  # 右上角
    top_left = areaMaxContour[0][0]  # 左上角
    if w < h:
        center_x = (box[0][0] + box[3][0]) / 2
        return center_x, box[0], box[3], h
    else:
        center_x = (box[0][0] + box[1][0]) / 2
        return center_x, box[0], box[1], w


def floor_zhengdui(type):
    global orgFrame
    global debug
    global org_img
    angle = 90
    while abs(angle - 0) >= 5:
        print(f"abs(angle-0) = {abs(angle - 0)}")
        img_h, img_w = org_img.shape[:2]
        bace_x, bace_y = img_w, 0
        _, dian0, dian1, _ = floor_main_subfunction(type)
        dian0_x, dian0_y = dian0[0], dian0[1]
        dian1_x, dian1_y = dian1[0], dian1[1]
        if float(dian0_x - dian1_x) != 0.:
            # 参考lane的斜率
            bace_k = (0 - 0) / (float(bace_x - 0))
            # 球—洞圆心直线的斜率
            line_k = (dian1_y - dian0_y) / (float(dian1_x - dian0_x))
            # 参考lane的方向向量
            bace_n = np.array([1, bace_k])
            # 球—洞圆心直线的方向向量
            line_n = np.array([1, line_k])
            # 参考lane的模长
            bace_L = np.sqrt(bace_n.dot(bace_n))
            # 球—洞圆心直线的模长
            line_L = np.sqrt(line_n.dot(line_n))
            # 参考line与球-洞圆心直线的夹角
            angle = int((np.arccos(bace_n.dot(line_n) / (float(bace_L * line_L))) * 180 / np.pi) + 0.5)
        else:
            angle = 90
        if angle is not 90:
            print(angle)
        if line_k >= 0:
            print("右转一小步")
            SSR.running_action_group('turn_right_fast', 1)
            # 需要右转
        else:
            print("左转一小步")
            SSR.running_action_group('turn_left_fast', 1)
            # 需要左转
    # return angle
    # while abs(angle - 0) >= 10:
    #     if dian1_y > dian0_y:
    #         print("右转一小步")
    #         # SSR.running_action_group('203', 1)
    #     else:
    #         print("左转一小步")
    #         # SSR.running_action_group('202', 1)


def floor_juzhong_pre(type):
    global orgFrame
    global debug
    global org_img
    center_x, _, _, _ = floor_main_subfunction(type)
    # print("box坐标",box)
    # cv2.imshow('org_Frame', org_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    print("center_x:", center_x)

    if center_x >= 270 and center_x <= 340:  # #######################################################################################################################参数
        skip = 1
        print("stand")
        # SSR.running_action_group("stand", 1)
    elif center_x < 270:  # #######################################################################################################################参数
        print("左移")
        skip = 0
        # SSR.running_action_group("160", 1)
    else:  # #######################################################################################################################参数
        print("右移")
        skip = 2
        # # SSR.running_action_group("161", 1)
    # print(sub_img.shape)

    # cv2.imshow('org_Frame',org_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return skip


def floor_juzhong(type):
    global bridge_c
    global orgFrame
    global debug
    global org_img
    bridge_c = 0
    floor_zhengdui(type)
    floor_juzhong_flag = True
    while floor_juzhong_flag:
        if floor_juzhong_pre(type=type) == 1:
            SSR.running_action_group("stand", 1)
            floor_juzhong_flag = False
            break
        elif floor_juzhong_pre(type=type) == 0:
            SSR.running_action_group("left_move_fast", 1)
            time.sleep(0.2)
        else:
            SSR.running_action_group("161", 1)
            # time.sleep(0.2)
        bridge_c = bridge_c + 1
        print("-------------" + str(bridge_c) + "----------")
    return floor_juzhong_flag


def closetofloor(type):
    global org_img
    count = 0
    while True:
        org_img_copy = cv2.resize(org_img, (r_w, r_h), interpolation=cv2.INTER_CUBIC)
        frame_gauss = cv2.GaussianBlur(org_img_copy, (3, 3), 0)  # 高斯模糊
        frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
        frame_floor_blue = cv2.inRange(frame_hsv, color_range[type][0],
                                       color_range[type][1])  # 对原图像和掩模(颜色的字典)进行位运算
        # cv2.imshow('test', frame_floor_blue)
        contours_floor_blue, hierarchy = cv2.findContours(frame_floor_blue, cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_NONE)
        floor_blue_area = getAreaSumContour(contours_floor_blue)
        print("floor_blue_area:", floor_blue_area)
        print("count:", count)
        if floor_blue_area < 15000:
            SSR.running_action_group('232new', 2)

            # SSR.running_action_group('232new', 1)

        else:
            count = count + 1
            SSR.running_action_group('stand', 1)
            time.sleep(1)
            break


def floor_():
    global org_img

    Board.setPWMServoPulse(1, 1100, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
    Board.setPWMServoPulse(2, 1450, 500)

    type = 'floor_blue'
    floor_juzhong(type)
    print("0")
    closetofloor(type)
    Board.setPWMServoPulse(1, 1050, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
    Board.setPWMServoPulse(2, 1450, 500)
    SSR.running_action_group('232new', 4)
    floor_juzhong(type)
    SSR.running_action_group('232new', 4)

    print('进入floor关卡')
    SSR.running_action_group('up-stairs', 1)
    SSR.running_action_group('232new', 2)
    time.sleep(0.1)
    SSR.running_action_group('up-stairs', 1)
    time.sleep(0.1)
    SSR.running_action_group('232new', 2)

    # SSR.running_action_group('stand', 1)
    SSR.running_action_group('up-stairs', 1)
    SSR.running_action_group('m_downstairs', 1)
    SSR.running_action_group("m_back_onestep", 1)
    # SSR.running_action_group("m_back", 1)
    time.sleep(0.1)
    SSR.running_action_group('m_downstairs', 1)
    SSR.running_action_group("m_back_onestep", 1)
    time.sleep(0.5)
    SSR.running_action_group('stand', 1)


def floor_to_slope():
    SSR.running_action_group("m_back_onestep", 1)
    slope_juzhong_pre()
    #  floor_juzhong('red_ramp')

    SSR.running_action_group("m_back_onestep", 1)
    SSR.running_action_group('stand', 1)
    SSR.running_action_group("m_downhill1_new", 1)
    time.sleep(0.2)

    c = 0
    while c < 7:
        SSR.running_action_group("232_su", 1)
        time.sleep(0.1)
        c = c + 1

    SSR.running_action_group("m_downhill3", 1)
    SSR.running_action_group("m_back_onestep", 2)




def slope_zhengdui(type):
    global orgFrame
    global debug
    global org_img
    angle = 90
    while abs(angle - 0) >= 5:
        print(f"abs(angle-0) = {abs(angle - 0)}")
        img_h, img_w = org_img.shape[:2]
        bace_x, bace_y = img_w, 0
        _, dian0, dian1, _ = floor_main_subfunction(type)
        dian0_x, dian0_y = dian0[0], dian0[1]
        dian1_x, dian1_y = dian1[0], dian1[1]
        if float(dian0_x - dian1_x) != 0.:
            # 参考lane的斜率
            bace_k = (0 - 0) / (float(bace_x - 0))
            # 球—洞圆心直线的斜率
            line_k = (dian1_y - dian0_y) / (float(dian1_x - dian0_x))
            # 参考lane的方向向量
            bace_n = np.array([1, bace_k])
            # 球—洞圆心直线的方向向量
            line_n = np.array([1, line_k])
            # 参考lane的模长
            bace_L = np.sqrt(bace_n.dot(bace_n))
            # 球—洞圆心直线的模长
            line_L = np.sqrt(line_n.dot(line_n))
            # 参考line与球-洞圆心直线的夹角
            angle = int((np.arccos(bace_n.dot(line_n) / (float(bace_L * line_L))) * 180 / np.pi) + 0.5)
            if line_k >= 0:
                print("右转一小步")
                SSR.running_action_group('turn_right_fast', 1)
                # 需要右转
            else:
                print("左转一小步")
                SSR.running_action_group('turn_left_fast', 1)
        else:
            angle = 90
        if angle is not 90:
            print(angle)


def slope_juzhong_pre():
    type = 'slope'
    global orgFrame
    global debug
    global org_img
    slope_zhengdui(type)
    center_x, _, _, _ = floor_main_subfunction(type)
    # print("box坐标",box)
    # cv2.imshow('org_Frame', org_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    print("center_x:", center_x)

    if center_x >= 270 and center_x <= 340:  # #######################################################################################################################参数
        skip = 1
        print("stand")
        SSR.running_action_group("stand", 1)
    elif center_x < 270:  # #######################################################################################################################参数
        print("左移")
        skip = 0
        SSR.running_action_group("left_move_faster", 1)
    else:  # #######################################################################################################################参数
        print("右移")
        skip = 2
        SSR.running_action_group("right_move_faster", 1)
    # print(sub_img.shape)

    # cv2.imshow('org_Frame',org_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return skip


def floor():
    floor_()
    floor_to_slope()


# ############################################### 终点站 ######################
def end_door_land_percent(type):  # 计算end_door_percent，判断是否走完
    global org_img
    # img = org_img[200:, :]
    r_h = org_img.shape[0]
    r_w = org_img.shape[1]

    gauss = cv2.GaussianBlur(org_img, (3, 3), 0)
    hsv = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)
    Imask0 = cv2.inRange(hsv, color_range[type][0],
                         color_range[type][1])
    # opened = cv2.morphologyEx(Imask0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接
    # erode = cv2.erode(Imask0, None, iterations=1)  # 腐蚀
    dilate = cv2.dilate(Imask0, np.ones((3, 3), np.uint8), iterations=2)  # 膨胀
    try:
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        area = getAreaSumContour(contours)
        # percent0 = round(area * 100 / (r_w * r_h), 2)
        areaMaxContour, area_max = getAreaMaxContour(contours)  # 找出最大轮廓
        # cv2.drawContours(org_img, areaMaxContour, -1, (255, 0, 255), 2)
        # cv2.imshow('o', org_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        percent = round(100 * area_max / (r_w * r_h), 2)  # 最大轮廓最小外接矩形的面积占比
        print(f"green_percent = {percent}")
        return percent, 'ok'
    except:
        return 0, 'error'


def end_door_color_detect_top(color, grad=1.0, ksizes=None, defaults=None):
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
    # frame_cali = cv2.imread("key frame/images/0667.jpg")
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


def end_door_area_sum(type):
    global org_img
    r_w = 160
    r_h = 120
    border = cv2.copyMakeBorder(org_img, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255))  # 扩展白边，防止边界无法识别
    org_img_copy = cv2.resize(border, (r_w, r_h), interpolation=cv2.INTER_CUBIC)  # 将图片缩放
    frame_gauss = cv2.GaussianBlur(org_img_copy, (3, 3), 0)  # 高斯模糊
    frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间

    frame_door = cv2.inRange(frame_hsv, color_range[type][0],
                             color_range[type][1])  # 对原图像和掩模(颜色的字典)进行位运算

    opened = cv2.morphologyEx(frame_door, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
    contour_area_sum = getAreaSumContour(contours)
    print("contour_area_sum:", contour_area_sum)
    return contour_area_sum


def floor_to_end():
    global org_img
    global land_result
    land_result = None
    SSR.running_action_group('stand', 1)
    time.sleep(0.5)
    img = org_img.copy()
    Board.setPWMServoPulse(1, 1200, 500)
    Board.setPWMServoPulse(2, 1450, 500)
    time.sleep(0.5)
    # cv2.imwrite("pi_send_img.jpg", img)
    # land_result = img_land_test()
    land_result = 'end_land'
    print(land_result)
    if land_result == 'end_land':
        cou_ = 6
        # 先前进几步，直到靠近最后的门那里（保持一定的距离，便于再次检测到门打开时，进行冲刺）
        while cou_ is not 0:
            print(f"{cou_}")
            SSR.running_action_group('239new', 1)
            cou_ -= 1
        # print("强行右转2次")
        # SSR.running_action_group('203', 2)
        # for i in range(3):
        #     SSR.running_action_group('203',1)
        #     time.sleep(0.5)
        # 正对且居中
        count = 0
        while True:
            grad = 2.5
            ksizes = [11, 7]
            defaults = [-2, (320, 240), (300, 240), (340, 240)]
            _, angle, points = end_door_color_detect_top(
                "white_road", grad, ksizes, defaults
            )
            center_x, center_y = points[0]
            print(f"angle,center_x, center_y = {angle, center_x, center_y}")
            # 调整角度
            print(f"angle: {angle:.2f}  ({center_x},{center_y})")
            if angle <= -3.5:
                print("右转")
                SSR.running_action_group('turn_right_fast', 1)
                time.sleep(0.2)
            elif angle >= 3:
                print("左转")
                SSR.running_action_group('turn_left_fast', 1)
                time.sleep(0.2)
            else:
                print("正对完成,左移调整")
                # 大步左平移
                if center_x <= 300:
                    print("左移")
                    SSR.running_action_group('left_move_fast', 1)
                elif center_x >= 370:
                    print("右移")
                    SSR.running_action_group('right_move_fast', 1)
                else:
                    SSR.running_action_group('stand', 1)
                    # time.sleep(0.1)
                    count += 1
                    if count >= 2:
                        print("移动到与正前方赛道边缘平行且居中的位置")
                        # for i in range(3):
                        # SSR.running_action_group('232new', 1)
                        # time.sleep(0.2)
                        break

        # print("land_result")
        # 抬头，便于检测地面
        # Board.setPWMServoPulse(1, 1100, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
        # Board.setPWMServoPulse(2, 1400, 500)
        # 先前进几步
        # 检测最后的门，如果门还没重新闭合，就等待，直到门重新打开时，立马往前冲过去


def end_door_():
    """
    思路：
    站立，抬头，确定处于end_door这一关卡（利用land_detect+door_area面积，双重判断），
    识别门是否闭合（情况1：闭合；情况2：未闭合）（计算面积）
        当闭合时，识别door面积，判断开启，等待开启
        当未闭合时，等待闭合，进入情况1
    最后在开启时，往前冲，根据地面图纸面积percent，判断何时停下来
    """
    global orgFrame
    global debug
    global org_img
    # global land_result
    type = 'yellow_door'
    end_door_flag = 0
    # 抬头
    Board.setPWMServoPulse(1, 1050, 500)
    Board.setPWMServoPulse(2, 1400, 500)
    time.sleep(0.5)
    end_door_percent_ = end_door_area_sum(type=type)
    # 判断是否处于end_door
    # if end_door_percent_ is not 'error' and land_result == 'end_land':
    if end_door_percent_ is not 'error':
        end_door_flag = 1
        # while True:
        # if end_door_area_sum(type=type) >= 350:
        # 直行
        ## SSR.running_action_group('239new', 1)
        # else:
        #  SSR.running_action_group('stand', 1)

        # while end_door_flag <= 3:

        while end_door_flag == 1:
            # 判断第一次门是否闭合
            print("close or open")
            a = end_door_area_sum(type=type)
            print(f"a = {a}")
            time.sleep(1)
            b = end_door_area_sum(type=type)
            print(f"b = {b}")
            deta = a - b
            print(f"deta = {deta}")
            if deta >= 100:  # 正闭合
                time.sleep(0.5)
                end_door_flag = 4


            else:  # 没闭合,原地等待

                SSR.running_action_group('stand', 1)
                # end_door_flag = 3

        # while end_door_flag == 2:
        #     # 判断门是否闭合
        #     print("close")
        #     if end_door_area_sum(type=type) == 0:  # 开
        #         end_door_flag = 4
        #         break
        #     else:  # 闭合,原地等待
        #         SSR.running_action_group('stand', 1)
        #
        # while end_door_flag == 3:
        #     print('open')
        #     SSR.running_action_group('stand', 1)
        #     if end_door_area_sum(type=type) >= 200:
        #         end_door_flag = 2

    count = 0
    while end_door_flag == 4:
        # 低头
        Board.setPWMServoPulse(1, 950, 500)
        Board.setPWMServoPulse(2, 1400, 500)
        time.sleep(0.5)
        percent, result = end_door_land_percent(type='end_land')
        print(f"percent, result = {percent, result}")
        if result is not 'error':
            if percent >= 20:
                while percent >= 20:
                    # 直行
                    SSR.running_action_group('239new', 1)
                    # end_door_flag = 3

            else:
                count += 1
            if count >= 2:
                SSR.running_action_group('stand', 1)
                print("state:end_door:exit")


def end_door():
    floor_to_end()
    end_door_()


# ############################################### main ######################

if __name__ == "__main__":
    start_door()
    state_pit()
    state_mine()
    baffler()
    door()
    bridge()
    ball()
    floor()
    end_door()
