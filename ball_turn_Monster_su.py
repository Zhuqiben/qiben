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

    # 图像处理及roi提取
    # global frame_cali, frame_ready
    # while not frame_ready:
    #     time.sleep(0.01)
    # frame_ready = False
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
                        SSR.running_action_group('turn_left_small_step', 1)
                        time.sleep(0.1)
                    else:
                        if top_center_y <= 240:
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
                        if top_center_y <= 90:
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
                Board.setPWMServoPulse(2, 2350, 500)
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
                    elif angle >= 4:
                        print("左转")
                        SSR.running_action_group('turn_left_fast', 1)
                        time.sleep(0.1)
                    else:
                        print("已经正对右侧红色地面，根据左侧的y值，纠正左右移动")
                        if top_center_y <= 110:
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


if __name__ == "__main__":
    ball()