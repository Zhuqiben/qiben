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

class line:
    """线段, 可计算中点、长度、角度"""

    def __init__(self, p0, p1):
        self.point0 = p0
        self.point1 = p1

    def mid_point(self):
        x = (self.point0[0] + self.point1[0]) // 2
        y = (self.point0[1] + self.point1[1]) // 2
        return (x, y)

    def length(self):
        dx_pow_2 = math.pow(self.point1[0] - self.point0[0], 2)
        dy_pow_2 = math.pow(self.point1[1] - self.point0[1], 2)
        return math.sqrt(dx_pow_2 + dy_pow_2)

    def angle(self):
        if self.point0[0] == self.point1[0]:
            return 90
        ang_tan = (self.point1[1] - self.point0[1]) / (self.point1[0] - self.point0[0])
        return -math.atan(ang_tan) * 180 / math.pi


def put_text(img, text, pos):
    cv2.putText(
        img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )


def get_frame_bin(img, color, ksize_smooth=3, ksize_morph=3):

    img_smooth = cv2.GaussianBlur(img, (ksize_smooth, ksize_smooth), 0)
    img_transform = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2LAB)
    img_thresh = cv2.inRange(
        img_transform, color_range1[color][0], color_range1[color][1]
    )
    kernel = np.ones((ksize_morph, ksize_morph), np.uint8)
    open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)
    return close


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


def mine_sort_contours_by_LT(cnts, method="left-to-right"):
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


def mine_color_detect_sill(color, area_min=1000, grad=1.0, ksizes=None, defaults=None):
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
        """
        cv2.imshow("Result", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    return (top_angle, [top_center, top_left, top_right])


def mine_color_detect_top(color, grad=1.0, ksizes=None, defaults=None):
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
        """
        cv2.imshow("Result", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """


    return percent, top_angle, [top_center, top_left, top_right]


def mine_color_detect_mine_near(color_back, color_obj, ksizes=None, area=None, default=None):
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
    else:
        mine_near = default_

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
        """
        cv2.imshow("Result", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    return (mine_x, mine_y)


def state_mine():
    print("state:mine:enter")
    global state
    time.sleep(0.5)
    self_flag = 0
    camera_flag = 0
    self_x = 320
    go_num = 0
    go_num_max = 3
    state = 'mine'
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
                # angle, points = mine_color_detect_sill(
                #     colors["next"], grad, ksizes[0], defaults
                # )
                # mine_x, mine_y = mine_color_detect_mine_near(
                #     colors["this"][1], colors["obj"], ksizes
                # )

                #
                angle, points = mine_color_detect_sill(
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
                    _, angle, points = mine_color_detect_top(
                        "white_road", grad, ksizes, defaults
                    )
                elif camera_flag == 1 or camera_flag == 2:
                    angle, points = mine_color_detect_sill(
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
            mine_x_switch = 75  # 65#70
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
                mine_x, mine_y = mine_color_detect_mine_near(
                    "white_mine", "black", ksizes
                )
                print(f"mine: {mine_x}, {mine_y}")
                if mine_y == 0:
                    step = 2
                    # AGC.runAction(Actions["go_LL"], 1, action_path)
                    SSR.running_action_group('239new', 1)
                    print("往前走一步")
                    break

                mine_x_self = mine_x - 320
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
                            print("往左移1步")

                        else:
                            print("state:mine:3.1.2")
                            # AGC.runAction(Actions["left_go_m"], 1, action_path)

                            SSR.running_action_group('160old', 1)
                            print("往左移2小步")
                        continue

                    if mine_x_range[0] < mine_x < mine_x_range[1] and self_flag == 1:
                        print("state:mine:4.1")
                        left_flag = True
                        if abs(mine_x_self) < mine_x_switch:
                            print("state:mine:4.1.1")
                            # AGC.runAction(Actions["right_go_L"], 1, action_path)

                            SSR.running_action_group('right_move_fast', 1)
                            print("往右移1步")

                        else:
                            print("state:mine:4.1.2")
                            # AGC.runAction(Actions["right_go_m"], 1, action_path)

                            SSR.running_action_group('right_move', 2)
                            print("往右移2小步")


                        continue
                else:
                    if mine_x_range[0] < mine_x < mine_x_range[1] and mine_x > 320:
                        print("state:mine:3.2")
                        left_flag = True
                        if abs(mine_x_self) < mine_x_switch:
                            print("state:mine:3.2.1")
                            # AGC.runAction(Actions["left_go_L"], 1, action_path)

                            SSR.running_action_group('left_move_fast', 1)
                            print("往左移1步")

                        else:
                            print("state:mine:3.2.2")
                            # AGC.runAction(Actions["left_go_m"], 1, action_path)

                            SSR.running_action_group('160old', 1)
                            print("往左移2小步")

                        continue

                    if mine_x_range[0] < mine_x < mine_x_range[1] and mine_x < 321:
                        print("state:mine:4.2")
                        left_flag = True
                        if abs(mine_x_self) < mine_x_switch:
                            print("state:mine:4.2.1")
                            # AGC.runAction(Actions["right_go_L"], 1, action_path)

                            SSR.running_action_group('right_move_fast', 1)
                            print("往右移1步")

                        else:
                            print("state:mine:4.2.2")
                            # AGC.runAction(Actions["right_go_m"], 1, action_path)

                            SSR.running_action_group('right_move', 2)
                            print("往右移2小步")

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
                angle, points = mine_color_detect_sill(
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


if __name__ == "__main__":
    state_mine()