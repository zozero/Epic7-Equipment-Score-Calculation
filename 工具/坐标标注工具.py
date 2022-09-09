import os.path

import cv2
import numpy as np
from PIL import Image

from 工具.图片处理工具 import 读取图片并转为灰度图, 起始位置


def 打印坐标(位置, 类型=''):
    增量y = 位置[3] - 位置[1]
    坐标列表 = []
    起始数值 = 1
    位置 = list(位置)
    for i in range(4):
        起始数值 += 1
        坐标列表.append(f'{str(位置[0])},{str(位置[1])},{str(位置[2])},{str(位置[3])},{类型}')

        位置[1] = 位置[1] + 增量y
        位置[3] = 位置[3] + 增量y
    print(坐标列表)
    return 坐标列表


def 显示截取效果(坐标列表1, 坐标列表2):
    图片 = Image.open('../图片/训练_提取/' + 图片名)
    图片 = np.asarray(图片)
    图片 = cv2.cvtColor(图片, cv2.COLOR_RGB2BGR)
    for 坐标1, 坐标2 in zip(坐标列表1, 坐标列表2):
        坐标1 = 坐标1.split(',')
        坐标2 = 坐标2.split(',')
        cv2.rectangle(图片, (int(坐标1[0]), int(坐标1[1])), (int(坐标1[2]), int(坐标1[3])), (0, 255, 0), 2)
        cv2.rectangle(图片, (int(坐标2[0]), int(坐标2[1])), (int(坐标2[2]), int(坐标2[3])), (0, 255, 0), 2)
    cv2.imshow('1', 图片)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 无英雄有转换石截取的位置 =906 579
    图片名 = 'test5.png'
    灰度_图片 = 读取图片并转为灰度图('../图片/训练_提取/' + 图片名)
    属性截取位置, 数值截取位置 = 起始位置(灰度_图片)

    属性坐标列表 = 打印坐标(属性截取位置, 类型='属性')
    数值坐标列表 = 打印坐标(数值截取位置, 类型='数值')
    显示截取效果(属性坐标列表, 数值坐标列表)

    文件名 = 图片名.split('.')[0] + '.txt'
    with open('../标签/坐标/' + 文件名, 'w', encoding='utf8') as 文件:
        文件.writelines('\n'.join(属性坐标列表))
        文件.writelines('\n')
        文件.writelines('\n'.join(数值坐标列表))
        文件.close()
