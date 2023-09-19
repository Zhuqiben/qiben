import cv2
import time
import numpy as np
# from config import *
import math
import queue
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

frame_show = True  # 是否显示
frame_cali = None  # 校正帧
frame_ready = True
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
    return center_x, center_y,area_max
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
    #turn_flag = True
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
                        SSR.running_action_group('160_test', 1)
                        time.sleep(0.1)
                    elif center_x >= 310:
                        print("右移")
                        SSR.running_action_group('161_test', 1)
                        time.sleep(0.1)
                    else:
                        step = 1################################################################

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
                    step = 3##############################################################

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

if __name__ == "__main__":
    door()