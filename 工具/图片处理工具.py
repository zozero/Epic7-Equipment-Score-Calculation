import os

import cv2
import numpy as np
from PIL import Image


def 读取图片并转为灰度图(图片路径):
    图片 = Image.open(图片路径)
    图片 = np.asarray(图片)
    图片 = cv2.cvtColor(图片, cv2.COLOR_RGB2BGR)
    灰度_图片 = cv2.cvtColor(图片, cv2.COLOR_BGR2GRAY)
    return 灰度_图片


def 起始位置(图片):
    """
    是用开源计算机可视化库进行模版匹配，来确定一个起始位置。这样在位置多变的情况也可以找到正确的位置
    :param 图片: 一张二维的灰度图
    :return:
    """
    该文件路径 = os.path.dirname(__file__)
    图片目录 = 该文件路径 + '/../图片/其他/'
    传说_图片 = Image.open(图片目录 + '传说横条.jpg')
    传说_图片 = np.asarray(传说_图片)
    传说_图片 = cv2.cvtColor(传说_图片, cv2.COLOR_RGB2BGR)
    灰度_传说_模板 = cv2.cvtColor(传说_图片, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('1', 灰度_传说_模板)

    匹配结果 = cv2.matchTemplate(图片, 灰度_传说_模板, cv2.TM_CCOEFF_NORMED)
    最小值, 最大值, 最小值位置, 最大值位置 = cv2.minMaxLoc(匹配结果)
    print('匹配结果最大值位置：', 最大值位置)
    # (左，上，右，下）
    return (最大值位置[0], 最大值位置[1] + 127, 最大值位置[0] + 135, 最大值位置[1] + 127 + 35), \
           (最大值位置[0] + 322, 最大值位置[1] + 127, 最大值位置[0] + 322 + 80, 最大值位置[1] + 127 + 35)

    # 转为灰度图后二者是一样的。
    # 英雄_图片 = Image.open(图片目录 + '英雄横条.jpg')
    # 英雄_图片 = np.asarray(英雄_图片)
    # 英雄_图片 = cv2.cvtColor(英雄_图片, cv2.COLOR_RGB2BGR)
    # 灰度_英雄_模板 = cv2.cvtColor(英雄_图片, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('2', 灰度_英雄_模板)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
