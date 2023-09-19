import cv2
import sys

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
import numpy as np
import math


def leMap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def getAreaMaxContour(contours, area_min=100):
    """
    获取最大轮廓以及它的面积
    """
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None
    for c in contours:  # 历遍所有轮廓
        contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > area_min:  # 最大面积的轮廓才是有效的，以过滤干扰
                area_max_contour = c
    return area_max_contour, contour_area_max  # 返回最大的轮廓与最大面积


def getAreaSumContour(contours):
    """
    求所有轮廓总面积
    """
    contour_area_sum = 0
    for c in contours:  # 历遍所有轮廓
        contour_area_sum += math.fabs(cv2.contourArea(c))  # 计算轮廓面积
    return contour_area_sum  # 返回总面积


def getAreaMaxContour2(contours, area=100):
    """
    提取最大轮廓
    """
    contour_area_max = 0
    area_max_contour = None
    for c in contours:
        contour_area_temp = math.fabs(cv2.contourArea(c))
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > area:  # 面积大于1
                area_max_contour = c
    return area_max_contour


def getAreaMax_contour_2(contours, area_min=100):
    """
    用于门的双框检测，单框难以检测
    """
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None
    contour_area_max_2 = 0
    area_max_contour_2 = None

    for c in contours:  # 历遍所有轮廓
        contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max_2 = contour_area_max
            area_max_contour_2 = area_max_contour
            contour_area_max = contour_area_temp
            if contour_area_temp > area_min:  # 最大面积的轮廓才是有效的，以过滤干扰
                area_max_contour = c
        elif contour_area_temp > contour_area_max_2:
            contour_area_max_2 = contour_area_temp
            if contour_area_temp > area_min:  # 最大面积的轮廓才是有效的，以过滤干扰
                area_max_contour_2 = c

    return area_max_contour, contour_area_max, area_max_contour_2, contour_area_max_2


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


def sort_contours_by_A(cnts, reverse=False):
    if not cnts:
        return ([], [])
    contourAreas = [cv2.contourArea(c) for c in cnts]
    (cnts, contourAreas) = zip(
        *sorted(zip(cnts, contourAreas), key=lambda b: b[1], reverse=reverse)
    )
    return (cnts, contourAreas)


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


color_dict = {
    'green': {  # hsv
        'Lower': np.array([40, 43, 46]),
        'Upper': np.array([82, 255, 255])
    },
    'ground': {
        'Lower': np.array([0, 123, 0]),
        'Upper': np.array([150, 255, 150])
    },
    'black': {
        'Lower': np.array([0, 25, 0]),
        'Upper': np.array([60, 255, 30])
    },
    'blue': {
        'Lower': np.array([88, 124, 79]),
        'Upper': np.array([108, 255, 255])  # ---------------------------------------------------------

    },
    'red': {
        'Lower': np.array([156, 43, 46]),
        'Upper': np.array([180, 255, 255])
    },
    'white': {
        'Lower': np.array([0, 0, 221]),
        'Upper': np.array([180, 30, 255])
    },
    'yellow': {
        'Lower': np.array([15, 43, 46]),
        'Upper': np.array([24, 255, 255])
    },
    'road_white': {
        'Lower': np.array([30, 2, 140]),
        'Upper': np.array([130, 16, 230])
    }
}
# 以下数据均为为实验室场景所测
color_range_old_lab = {
    'yellow_door': [(5, 169, 135), (28, 255, 255)],  # 开门的颜色阈值  ok
    'green_ground': [(0, 21, 107), (99, 77, 255)],  # 开门之后的草地颜色 ok
    'pit_hole': [(25, 23, 0), (61, 255, 141)],  # 坑颜色   ok
    # 'pit_hole2': [(0, 26, 0), (56, 255, 255)],  # 侧面坑颜色
    'pit_green': [(58, 23, 0), (255, 255, 255)],  # 坑边缘绿色   ok
    'pit_right_green': [(54, 63, 49), (66, 255, 255)],  # 右边坑边缘绿色
    'mine_land': [(39, 0, 118), (100, 44, 230)],  # 雷区地面    ok
    'floor_blue': [(97, 63, 0), (140, 255, 234)],  # 台阶蓝色   ok
    'floor_green': [(48, 48, 82), (86, 255, 255)],  # 台阶绿色
    'slope': [(153, 0, 0), (255, 255, 255)],  # 红色斜坡        ok
    'end_land': [(43, 0, 0), (128, 255, 255)],  # 最后的赛道图纸
    'bridge': [(51, 138, 54), (84, 255, 255)],  # 绿桥        ok
    'baffler': [(74, 118, 35), (237, 255, 255)],  # 挡板  ok
    'corner_land': [(54, 0, 46), (255, 255, 255)],  # 转角地面
    'door': [(99, 143, 49), (255, 255, 151)],  # 门框     ok
    "corner_test": [(58, 0, 189), (253, 30, 255)],  # 在靠近门之后,根据该阈值进行percent判断
    'red_brick': [(166, 20, 0), (255, 255, 151)],  # 红砖(踢球部分)   ok
    'white_ball': [(193, 0, 0), (255, 255, 255)],  # 白球     ok
}

color_range = {
    'yellow_door': [(0, 92, 0), (48, 255, 255)],  # 开门的颜色阈值       # ok
    'slope': [(151, 151, 70), (179, 255, 255)],  # 红色斜坡             ok
    'door': [(88, 124, 79), (108, 255, 255)],  # 门的                     ok
    'baffler': [(97, 152, 97), (123, 255, 255)],  # 挡板             #  ok
    'bridge': [(63, 29, 174), (81, 255, 238)],  # 桥，             # ok
    'pit_hole': [(0, 0, 0), (167, 255, 126)],  # 回字黑色洞                 # ok
    'red_brick': [(0, 0, 197), (75, 219, 255)],  # ball前的红砖           # ok
    'lei': [(0, 0, 0), (179, 255, 46)],  # 雷的阈值                          ok
    'floor_blue': [(83, 96, 80), (109, 255, 255)],  # 蓝色阶梯           #  ok
    'green_ground': [(29, 17, 133), (77, 143, 198)],  # 开门之后的草地颜色   # ok
    'pit_green': [(60, 105, 1119), (77, 255, 246)],  # 回字绿色         # ok
    'pit_right_green': [(71, 146, 48), (86, 255, 255)],  # 回字右边绿色         # ok
    'ball_ground': [(24, 9, 28), (156, 255, 160)],  # 球处的地面           ok
    'mine_land': [(40, 47, 41), (85, 255, 255)],  # 雷的地面              # ok
    'end_door': [(7, 119, 87), (70, 251, 199)],  # 最后的门的阈值           ok
    'lei_huizi': [(58, 59, 56), (75, 255, 169)],  # 雷后面回字             ok
    'lei_dangban': [(101, 131, 88), (135, 254, 176)],  # 雷后面挡板        ok
    'lei_left_ground': [(40, 47, 41), (85, 255, 255)],  # 雷/回字的左地面        ok
    'lei_right_ground': [(40, 47, 41), (85, 255, 255)],  # 雷/回字的右地面       ok
    'white_ball': [(193, 0, 0), (255, 255, 255)],  # 白球         # ok
    "end_land": [(54, 0, 46), (255, 255, 255)],  # 终点地面      未调整
    # 'green_hole': [(35, 43, 20), (100, 255, 255)],
    # 'yellow_hole': [(10, 70, 46), (34, 255, 255)],
    # 'black_hole': [(0, 0, 0), (180, 255, 80)],
    # 'black_gap': [(0, 0, 0), (180, 255, 100)],
    # 'black_dir': [(0, 0, 0), (180, 255, 46)],
    # 'blue': [(110, 43, 46), (124, 255, 255)],
    # 'black_door': [(0, 0, 0), (180, 255, 46)],
    # 'black': [(0, 0, 0), (180, 255, 46)],
    # 'red_floor': [(0, 27, 0), (5, 255, 255)],  # 红色阶梯
    # 'green_floor': [(31, 46, 172), (179, 255, 229)],  # 绿色阶梯
    # 'ball': [(0, 0, 202), (179, 65, 255)],  # 球的颜色阈值
    # 'ball_hole': [(78, 6, 216), (108, 95, 255)],  # 球洞的颜色阈值
    # 'lei_zhengdui_huizi': [(58, 59, 56), (75, 255, 169)],  # 雷区后面为绿色回字时的阈值（绿色的，在现场要测）
    # 'ground_left': [(0, 0, 0), (179, 255, 150)],
    # 'ground_right': [(0, 0, 0), (179, 255, 150)],
    # 'ground': [(0, 5, 71), (153, 255, 120)],
    # 'huizi_black': [(0, 0, 0), (180, 255, 46)],  ###################回字hsv特殊值

}

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
r_w = 320
r_h = 240
if __name__ == '__main__':
    print("比赛相关的计算轮廓函数  以及  某些颜色阈值")
