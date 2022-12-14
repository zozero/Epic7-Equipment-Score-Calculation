import os
from PIL import Image

from 工具.图片处理工具 import 起始位置, 读取图片并转为灰度图


def 保存图片(图片, 保存图片名):
    # 图片.save('图片/测试_属性/'+保存图片名 + '.jpg')
    # 图片.save('未标注的图片/' + 保存图片名 + '.jpg')
    图片.save(保存图片名 + '.jpg')
    print(保存图片名 + '：已保存')


def 读取图片(图片路径):
    if os.path.isfile(图片路径):
        图片 = Image.open(图片路径)
        # self.显示图片(图片数组)
        return 图片
    else:
        raise ValueError('开源计算机可视化库类-读取图片 %s' % 图片路径, '图片载入失败')


def 截取并保存(图片名='test.png', 截取位置=(0, 0, 0, 0), 图片存储路径='../图片/测试_数值/', 前缀=''):
    增量y = 截取位置[3] - 截取位置[1]
    图片1 = 读取图片(图片名)
    if os.path.exists(图片存储路径) is False:
        os.mkdir(图片存储路径)
    起始数值 = len(os.listdir(图片存储路径))
    截取位置 = list(截取位置)
    for i in range(4):
        图片2 = 图片1.crop(截取位置)
        # 显示图片(图片2)
        起始数值 += 1
        保存图片(图片2, 图片存储路径 + 前缀 + str(起始数值))
        截取位置[1] = 截取位置[1] + 增量y
        截取位置[3] = 截取位置[3] + 增量y


def 属性和数值图片截取():
    """
    仅在使用.py文件中使用
    :return:
    """
    该文件路径 = os.path.dirname(__file__)
    图片文件名 = 该文件路径 + '/../未标注的图片/'

    for 文件名 in os.listdir(图片文件名):
        if 'Screenshot' in 文件名:
            图片名 = 图片文件名 + 文件名
            灰度_图片 = 读取图片并转为灰度图(图片名)
            属性截取位置, 数值截取位置 = 起始位置(灰度_图片)
            截取并保存(图片名=图片名, 截取位置=属性截取位置, 图片存储路径=该文件路径 + '/../图片/测试_属性/', 前缀='ceshi')
            截取并保存(图片名=图片名, 截取位置=数值截取位置, 图片存储路径=该文件路径 + '/../图片/测试_数值/', 前缀='ceshi')


if __name__ == '__main__':
    属性和数值图片截取()
    图片名 = 'test4.png'
    灰度_图片 = 读取图片并转为灰度图('../图片/训练_提取/' + 图片名)
    exit()
    属性截取位置, 数值截取位置 = 起始位置(灰度_图片)

    截取并保存(截取位置=属性截取位置, 图片存储路径='../图片/训练_属性/')
    截取并保存(截取位置=数值截取位置, 图片存储路径='../图片/训练_数值/')
    # 属性和数值图片截取()
    pass
