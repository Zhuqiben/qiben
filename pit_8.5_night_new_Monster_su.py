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
        defaults_ = [0, (320, 240), (300, 240), (340, 240)]
    else:
        defaults_ = defaults

    # 图像处理及roi提取
    # global frame_cali, frame_ready
    # while not frame_ready:
    #     time.sleep(0.01)
    # frame_ready = False
    # frame_cali = cv2.imread("key frame/images/0667.jpg")
    #time.sleep(0.5)
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
    """返回目标区域占比, 底部角度, 底部中点[x,y], 底部左点[x,y], 底部右点[x,y]

    Args:
        color (str): 颜色
        grad (float, optional): 斜率. Defaults to 1.0.
        ksizes (list, optional): 高斯核与形态学运算核. Defaults to (3, 3).
        defaults (list, optional): 缺省[底部角度, 底部中点[x, y], 底部左点[x, y], 底部右点[x, y]]. Defaults to [0, (320, 240), (300, 240), (340, 240)].

    Returns:
        float, float, list: 目标区域占比, 底部角度, [底部中点[x, y], 底部左点[x, y], 底部右点[x, y]]
    """
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


def state_pit():
    print("state:hole:enter")
    global state
    step = 0# 2
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
                if angle <= -2:
                    print("右转")
                    SSR.running_action_group('turn_right_fast', 1)
                    #time.sleep(0.1)
                elif angle >= 2.8:
                    print("左转")
                    SSR.running_action_group('turn_left_fast', 1)
                    #time.sleep(0.1)
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
                            break

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
                        #SSR.running_action_group('232new', 2)
                        count += 1
                        if count >= 2:
                            print("state:hole:step 1 done, goto step 2")
                            step = 2
                            break

        # 2、前进
        if step == 2:
            grad = 2.5
            ksizes = [11, 7]
            defaults = [-2, (320, 361), (300, 361), (340, 361)]
            count = 0
            SSR.running_action_group('stand', 1)
            Board.setPWMServoPulse(1, 1000, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
            Board.setPWMServoPulse(2, 1400, 500)
            while step == 2:
                # 获取数据
                time.sleep(0.1)
                _, angle, _ = pit_color_detect_top(
                    "green", grad, ksizes, defaults
                )
                _, _, points = pit_color_detect_bottom(
                    "white_road", grad, ksizes, defaults
                )
                _, center_y = points[0]
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
                    elif left_x >= 150:
                        print("右移")
                        SSR.running_action_group('right_move_faster', 1)
                    else:
                        print("正对，前进")
                        if center_y <= 290:
                            SSR.running_action_group('232new', 2)
                            # 进入下一步
                        else:
                            #count += 1
                            print("state:hole:step 2 done, goto step 3")
                            SSR.running_action_group('232new', 3)
                            step = 3
                            # break

        # 3、回到中间
        if step == 3:
            # 进入下一关
            Board.setPWMServoPulse(1, 1350, 100)
            Board.setPWMServoPulse(2, 550, 100)
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
            #count = 0
            # while pit_ground_percent(type='pit_right_green') >= 0.1 and count <= 10:
            for i in range(8):
                print("需要右移")
                SSR.running_action_group('right_move_30', 1)
                #count += 1
                

            state = None
            break
    Board.setPWMServoPulse(1, 1000, 500)
    Board.setPWMServoPulse(2, 1450, 500)
    SSR.running_action_group('stand', 1)
    print("state:hole:exit")


if __name__ == "__main__":
    state = 'pit'
    state_pit()
