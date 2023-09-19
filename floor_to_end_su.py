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


# ###############################################读取图像线程###########################
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
            SSR.running_action_group("160", 1)
            time.sleep(0.2)
        else:
            SSR.running_action_group("161", 1)
            time.sleep(0.2)
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
            SSR.running_action_group('232new',2)

            #SSR.running_action_group('232new', 1)

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
    SSR.running_action_group('232new', 1)
    time.sleep(0.1)
    SSR.running_action_group('up-stairs', 1)
    time.sleep(0.1)
    SSR.running_action_group('232new', 1)

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


def floor_to_end():
    global org_img
    global land_result
    land_result = None
    time.sleep(0.5)
    img = org_img.copy()

    # cv2.imwrite("pi_send_img.jpg", img)
    # land_result = img_land_test()
    # print(land_result)
    land_result = 'end_land'
    if land_result == 'end_land':
        cou_ = 0
        # 先前进几步，直到靠近最后的门那里（保持一定的距离，便于再次检测到门打开时，进行冲刺）
        while cou_ is not 0:
            print(f"{cou_}")
            SSR.running_action_group('239new', 1)
            cou_ -= 1
        # print("land_result")
        # 抬头，便于检测地面
        Board.setPWMServoPulse(1, 1100, 500)  # NO3机器人，初始云台角度（1号代表上下，2号代表左右）
        Board.setPWMServoPulse(2, 1400, 500)
        # 先前进几步
        # 检测最后的门，如果门还没重新闭合，就等待，直到门重新打开时，立马往前冲过去


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

     #cv2.imshow('org_Frame',org_img)
     #cv2.waitKey()
     #cv2.destroyAllWindows()
    return skip


if __name__ == '__main__':
    floor_()
    floor_to_slope()
    #floor_to_end()
