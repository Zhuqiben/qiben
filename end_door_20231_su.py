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
        #cv2.imshow('o', org_img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
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
        cou_ = 5
        # 先前进几步，直到靠近最后的门那里（保持一定的距离，便于再次检测到门打开时，进行冲刺）
        while cou_ is not 0:
            print(f"{cou_}")
            SSR.running_action_group('239new', 1)
            cou_ -= 1
        #print("强行右转2次")
        #SSR.running_action_group('203', 2)
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

if __name__ == '__main__':
    floor_to_end()
    end_door_()
    # Board.setPWMServoPulse(1, 1200, 500)
    # Board.setPWMServoPulse(2, 1450, 500)
"""
最后的双开合门
还没测试
"""