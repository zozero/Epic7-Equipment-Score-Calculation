import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

import torch.nn.functional as func
from torch.utils.data import Dataset


class 数据处理(Dataset):
    def __init__(self, 图片目录, 标签目录):
        self.图片目录 = 图片目录
        self.标签目录 = 标签目录
        self.图片名列表 = os.listdir(图片目录)

    def __len__(self):
        return len(self.图片名列表)

    def __getitem__(self, 索引):
        图片名 = self.图片名列表[索引]
        图片 = self.读取图片(self.图片目录 + 图片名)
        图片[np.where(图片[:, :] == 0)] = 1
        高, 宽 = 图片.shape

        标签文件路径 = os.path.join(self.标签目录, 图片名.split('.')[0] + '.txt')
        标注的大方框列表 = self.读取文件(标签文件路径)
        标注内的小方框矩阵 = self.生成标注内的小方框(标注的大方框列表)
        # print(小方框矩阵.shape)
        # self.画出方框(图片, 注内的小方框矩阵)

        [标签矩阵, 偏移量], 满图的小方框矩阵 = self.生成标签((高, 宽), 标注内的小方框矩阵)
        # self.画出方框(图片, 满图的小方框矩阵)

        图片均值 = [123.68, 116.779, 103.939]  # 固定的，从别人那获取的固定值，根据一个较为全面的很多很多图片统计得出的结果
        # 减去均值后的图片 = 图片 - 图片均值
        减去均值后的图片 = 图片
        # self.显示图片(减去均值后的图片)
        # exit()

        偏移量 = np.hstack([标签矩阵.reshape(标签矩阵.shape[0], 1), 偏移量])
        标签矩阵 = np.expand_dims(标签矩阵, axis=0)

        减去均值后的图片 = torch.from_numpy(减去均值后的图片.transpose([0, 1])).float()
        标签矩阵 = torch.from_numpy(标签矩阵).float()
        偏移量 = torch.from_numpy(偏移量).float()

        return 图片, 减去均值后的图片, 标签矩阵, 偏移量

    def 保存图片(self, 图片, 图片路径):
        图片 = cv2.cvtColor(图片, cv2.COLOR_GRAY2RGB)
        图片 = Image.fromarray(图片)
        图片.save(图片路径)

    def 读取图片(self, 图片路径):
        图片 = Image.open(图片路径)
        # 把png图片转jpg图片然后转为灰度图，用于减少数据计算量
        if 'png' in 图片路径:
            图片路径2 = str.replace(图片路径, 'png', 'jpg')
            图片 = 图片.convert('L')
            图片.save(图片路径2, format='jpeg')
            图片.close()
            os.remove(图片路径)
            图片 = Image.open(图片路径2)

        图片 = np.asarray(图片).copy()
        return 图片

    def 显示图片(self, 图片):
        cv2.imshow('xstp', 图片)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def 读取文件(self, 文件路径):
        方框列表 = list()
        with open(文件路径, 'r', encoding='utf8') as 文本:
            for 一行 in 文本.readlines():
                一行 = 一行.split('\n')[0]
                一行 = 一行.split(',')
                属性坐标 = list(map(int, 一行[:4]))
                方框列表.append(属性坐标)
            文本.close()
        return 方框列表

    def 生成标注内的小方框(self, 大方框列表):
        """
        通过已经标注好的坐标，在坐标所形成的大方框内生成很多小方框
        :param 大方框列表:
        :return:
        """
        小方框列表 = list()
        for 大方框 in 大方框列表:
            # 大方框数值（左，上，右，下）对应图片的位置
            # 宽度16是个固定值
            左1 = 大方框[0]
            for i in range(大方框[0] // 16 + 1, 大方框[2] // 16 + 1):
                # 左2 = 16 * i - 0.5  # 减0.5是为了与上一个相交？？
                左2 = 16 * i
                小方框列表.append((左1, 大方框[1], 左2, 大方框[3]))
                左1 = 左2
            小方框列表.append((左1, 大方框[1], 大方框[2], 大方框[3]))

        return np.array(小方框列表)

    def 生成满图的小方框(self, 行列数):
        高度 = 35  # 35是事先计算好的值
        宽度 = 16

        行数, 列数 = 行列数
        方框列表 = []
        for 行 in range(行数):
            for 列 in range(列数):
                方框列表.append([宽度 * 列, 高度 * 行, 宽度 * (列 + 1), 高度 * (行 + 1)])
        return np.array(方框列表).reshape((-1, 4))

    def 生成标签(self, 图片尺寸, 标注内的小方框矩阵):
        图片高, 图片宽 = 图片尺寸
        满图的小方框矩阵 = self.生成满图的小方框((int(图片高 / 35), int(图片宽 / 16)))
        重叠部分的交并比矩阵 = self.计算重叠的小方框(满图的小方框矩阵, 标注内的小方框矩阵)

        标签矩阵 = np.empty(满图的小方框矩阵.shape[0])
        标签矩阵.fill(-1)
        # 这里可能固定只有60列，所以有60个描述在哪一行的值，相当于找到最大值的行号
        每一列最大值的行索引 = 重叠部分的交并比矩阵.argmax(axis=0)
        # 这里可能固定只有8040行，所以有8040个描述在哪一列的值，相当于找到最大值的列号
        每一行最大值的列索引 = 重叠部分的交并比矩阵.argmax(axis=1)
        每一行的最大值 = 重叠部分的交并比矩阵[range(重叠部分的交并比矩阵.shape[0]), 每一行最大值的列索引]

        # 好像不存在大于0.7的值。。。。。。，可能需要更改
        # 开始制作标签
        标签矩阵[每一行的最大值 >= 0.2] = 1
        标签矩阵[每一行的最大值 < 0.2] = 0

        # 标签矩阵[每一列最大值的行索引] = 1
        # 越界的小方框索引 = np.where(
        #     (满图的小方框矩阵[:, 0] < 0) |
        #     (满图的小方框矩阵[:, 1] < 0) |
        #     (满图的小方框矩阵[:, 2] > 图片宽) |
        #     (满图的小方框矩阵[:, 3] > 图片高)
        # )[0]
        # 标签矩阵[越界的小方框索引] = -1

        # 积极的小方框索引 = np.where(标签矩阵 == 1)[0]
        偏移量 = self.微调小方框的边界(满图的小方框矩阵, 标注内的小方框矩阵[每一行最大值的列索引, :])
        return [标签矩阵, 偏移量], 满图的小方框矩阵

    def 微调小方框的边界(self, 满图的小方框矩阵, 标注好的小方框矩阵):
        """
        通过计算两个参数各小方框纵轴的中心坐标
        这里为了能够更简单的理解所以变量名很长
        :param 满图的小方框矩阵:
        :param 标注好的小方框矩阵: 大小和满图的小方框矩阵一致，但每一行都是存着标注好的小方框
        :return: 返回两个参数纵坐标的距离与高之间的比率和满图的小方框的高
        """
        标注好的小方框中心的纵坐标 = (标注好的小方框矩阵[:, 3] + 标注好的小方框矩阵[:, 1]) * 0.5
        满图的小方框中心的纵坐标 = (满图的小方框矩阵[:, 3] + 满图的小方框矩阵[:, 1]) * 0.5
        标注好的小方框高度的矩阵 = 标注好的小方框矩阵[:, 3] - 标注好的小方框矩阵[:, 1] + 1.0
        满图的小方框高度的矩阵 = 满图的小方框矩阵[:, 3] - 满图的小方框矩阵[:, 1] + 1.0

        两者纵坐标距离与其中的满图的小方框的高之间比率的矩阵 = (
                                                                       标注好的小方框中心的纵坐标 - 满图的小方框中心的纵坐标) / 满图的小方框高度的矩阵
        # 由于当前高度只有一个值35，所以比率始终是1，加入对数后结果为零......
        两者高之间比率对数的矩阵 = np.log(标注好的小方框高度的矩阵 / 满图的小方框高度的矩阵)
        return np.vstack((两者纵坐标距离与其中的满图的小方框的高之间比率的矩阵, 两者高之间比率对数的矩阵)).transpose()

    def 还原微调的小方框(self, 满图的小方框矩阵, 偏移量):
        满图的小方框中心的纵坐标 = (满图的小方框矩阵[:, 3] + 满图的小方框矩阵[:, 1]) * 0.5
        满图的小方框高度的矩阵 = 满图的小方框矩阵[:, 3] - 满图的小方框矩阵[:, 1] + 1.0

        两者纵坐标距离与其中的满图的小方框的高之间比率的矩阵 = 偏移量[0, :, 0]
        两者高之间比率对数的矩阵 = 偏移量[0, :, 1]

        预测的小方框高度 = np.exp(两者高之间比率对数的矩阵) * 满图的小方框高度的矩阵
        标注好的小方框中心的纵坐标 = 满图的小方框高度的矩阵 * 两者纵坐标距离与其中的满图的小方框的高之间比率的矩阵 + 满图的小方框中心的纵坐标

        # 该步骤在该项目可能不需要
        满图的小方框中心的横坐标 = (满图的小方框矩阵[:, 0] + 满图的小方框矩阵[:, 2]) * 0.5
        x1 = 满图的小方框中心的横坐标 - 16 * 0.5
        x2 = 满图的小方框中心的横坐标 + 16 * 0.5

        y1 = 标注好的小方框中心的纵坐标 - 预测的小方框高度 * 0.5
        y2 = 标注好的小方框中心的纵坐标 + 预测的小方框高度 * 0.5

        微调好的小方框 = np.vstack((x1, y1, x2, y2)).transpose()
        return 微调好的小方框

    def 计算小方框坐标(self, 满图的小方框矩阵):
        pass

    def 处理溢出值(self, 微调好的满图小方框, 尺寸):
        """
        防止小方框的超出图片的范围
        :param 微调好的满图小方框:
        :param 尺寸:
        :return:
        """
        微调好的满图小方框[:, 0] = np.maximum(np.minimum(微调好的满图小方框[:, 0], 尺寸[1] - 1), 0)
        微调好的满图小方框[:, 1] = np.maximum(np.minimum(微调好的满图小方框[:, 1], 尺寸[0] - 1), 0)
        微调好的满图小方框[:, 2] = np.maximum(np.minimum(微调好的满图小方框[:, 2], 尺寸[1] - 1), 0)
        微调好的满图小方框[:, 3] = np.maximum(np.minimum(微调好的满图小方框[:, 3], 尺寸[0] - 1), 0)
        return 微调好的满图小方框

    def 计算大于最小尺寸的索引(self, 许多小方框):
        所有宽 = 许多小方框[:, 2] - 许多小方框[:, 0] + 1
        所有高 = 许多小方框[:, 3] - 许多小方框[:, 1] + 1
        索引 = np.where((所有宽 >= 16) & (所有高 >= 35))[0]
        return 索引

    def 计算重叠的小方框(self, 小方框矩阵1, 小方框矩阵2):
        小方框的面积矩阵1 = (小方框矩阵1[:, 0] - 小方框矩阵1[:, 2]) * (小方框矩阵1[:, 1] - 小方框矩阵1[:, 3])
        小方框的面积矩阵2 = (小方框矩阵2[:, 0] - 小方框矩阵2[:, 2]) * (小方框矩阵2[:, 1] - 小方框矩阵2[:, 3])
        重叠部分的交并比矩阵 = np.zeros((小方框矩阵1.shape[0], 小方框矩阵2.shape[0]))

        for 索引, (小方框, 小方框的面积) in enumerate(zip(小方框矩阵1, 小方框的面积矩阵1)):
            # 生成的都是矩阵
            x1 = np.maximum(小方框[0], 小方框矩阵2[:, 0])
            x2 = np.minimum(小方框[2], 小方框矩阵2[:, 2])
            y1 = np.maximum(小方框[1], 小方框矩阵2[:, 1])
            y2 = np.minimum(小方框[3], 小方框矩阵2[:, 3])

            各重叠部分的面积 = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
            # 重叠部分的面积与两个小方框面积之和减去一次重叠部分的比例
            交并比矩阵 = 各重叠部分的面积 / (小方框的面积 + 小方框的面积矩阵2[:] - 各重叠部分的面积[:])
            重叠部分的交并比矩阵[索引][:] = 交并比矩阵
        return 重叠部分的交并比矩阵

    def 画出方框(self, 图片, 方框):
        """
        用于可视化显示图片处理效果
        :param 方框:
        :return:
        """
        for i in range(len(方框)):
            点1 = int(方框[i][0]), int(方框[i][1])
            点2 = int(方框[i][2]), int(方框[i][3])
            图片 = cv2.rectangle(图片, 点1, 点2, (100, 200, 100))
        cv2.imshow('fangkuang', 图片)
        cv2.waitKey()
        cv2.destroyAllWindows()


class 简化的可视化几何群组网络(nn.Module):
    def __init__(self):
        super(简化的可视化几何群组网络, self).__init__()
        self.卷积1_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.整流线性函数1_1 = nn.ReLU(True)
        self.最大池化1_1 = nn.MaxPool2d(3, 3)
        self.卷积1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.整流线性函数1_2 = nn.ReLU(True)
        self.最大池化1_2 = nn.MaxPool2d(2, 2)

        self.卷积2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.整流线性函数2_1 = nn.ReLU(True)
        self.最大池化2_1 = nn.MaxPool2d(2, 2)
        self.卷积2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.整流线性函数2_2 = nn.ReLU(True)
        self.最大池化2_2 = nn.MaxPool2d(2, 2)

        # 全连接层输入 = 4 * 7
        # self.全连接层1 = nn.Linear(全连接层输入, 128)
        # self.全连接层2 = nn.Linear(128, 64)
        # self.软最大 = nn.Softmax()

    def forward(self, 输入):
        输出 = self.卷积1_1(输入)
        输出 = self.整流线性函数1_1(输出)
        输出 = self.最大池化1_1(输出)
        输出 = self.卷积1_2(输出)
        输出 = self.整流线性函数1_2(输出)
        输出 = self.最大池化1_2(输出)

        输出 = self.卷积2_1(输出)
        输出 = self.整流线性函数2_1(输出)
        输出 = self.最大池化2_1(输出)
        输出 = self.卷积2_2(输出)
        输出 = self.整流线性函数2_2(输出)
        输出 = self.最大池化2_2(输出)
        # 个数, 特征, 高, 宽 = 输出.size()
        # 输出 = 输出.view(个数 * 特征, 高 * 宽)
        #
        # 输出 = self.全连接层1(输出)
        # 输出 = self.全连接层2(输出)
        # 输出 = self.软最大(输出)
        return 输出


class 连接文本提取网络(nn.Module):
    def __init__(self):
        super(连接文本提取网络, self).__init__()
        self.简化的可视化几何群组 = 简化的可视化几何群组网络()
        self.卷积1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.整流线性函数1 = nn.ReLU(True)

        self.门控反复单元 = nn.GRU(128, 64, bidirectional=True, batch_first=True)
        self.卷积2 = nn.Conv2d(128, 128, 1, 1)
        self.整流线性函数2 = nn.ReLU(True)

        # 1是选择的高的数量，乘2是因为把它分为前景和背景，
        self.分类 = nn.Conv2d(128, 1 * 2, 1, 1)
        self.偏移量 = nn.Conv2d(128, 1 * 2, 1, 1)
        # self.分类 = nn.Conv2d(64, 1, 1, 1)
        # self.偏移量 = nn.Conv2d(64, 1, 1, 1)

    def forward(self, 输入):
        # 1*1*1080*1920
        输入 = 输入.view(1, 输入.size(0), 输入.size(1), 输入.size(2))
        输出 = self.简化的可视化几何群组(输入)

        输出 = self.卷积1(输出)
        输出 = self.整流线性函数1(输出)
        尺寸备份 = 输出.size()

        # 置换维度，使之符合门控反复单元函数
        输出 = 输出.permute(0, 2, 3, 1).contiguous()
        尺寸 = 输出.size()
        输出 = 输出.view(尺寸[0] * 尺寸[1], 尺寸[2], 尺寸[3])
        输出, _ = self.门控反复单元(输出)

        输出 = 输出.view(尺寸备份[0], 尺寸备份[2], 尺寸备份[3], 128)
        输出 = 输出.permute(0, 3, 1, 2).contiguous()
        输出 = self.卷积2(输出)
        输出 = self.整流线性函数2(输出)

        分类 = self.分类(输出)
        分类 = 分类.permute(0, 2, 3, 1).contiguous()
        分类 = 分类.view(分类.size(0), 分类.size(1) * 分类.size(2), 2)

        偏移量 = self.偏移量(输出)
        偏移量 = 偏移量.permute(0, 2, 3, 1).contiguous()
        偏移量 = 偏移量.view(偏移量.size(0), 偏移量.size(1) * 偏移量.size(2), 2)
        # print(分类.shape)

        return 分类, 偏移量


class 分类损失值(nn.Module):
    def __init__(self, 设备):
        super(分类损失值, self).__init__()
        self.设备 = 设备
        self.损失函数 = nn.CrossEntropyLoss(reduction='none')

    # def forward(self, 输入, 标签):
    #     标签 = 标签[0][0]
    #     标签的索引 = (标签 == 1).nonzero()[:, 0]
    #     标签 = 标签[标签的索引].long()
    #     预测的分类 = 输入[0][标签的索引]
    #     # print(预测的分类)
    #     a=func.log_softmax(预测的分类, dim=-1)[:,1]
    #     # a=a.cpu().detach().numpy()
    #     # b=np.where(a[:]>0)
    #     # 损失值 = func.nll_loss(func.log_softmax(预测的分类, dim=-1), 标签)
    #     损失值 = func.nll_loss(a, 标签)
    #     损失值 = torch.clamp(torch.mean(损失值), 0, 10) if 损失值.numel() > 0 else torch.tensor(0.0)
    #     # print(损失值)
    #
    #     return 损失值.to(self.设备)

    # def forward(self, 输入, 标签):
    #     标签 = 标签[0][0]
    #     标签 = 标签.long()
    #     预测的分类 = 输入[0][:, 0]
    #     a=预测的分类 - 标签
    #     b=func.softmax(a)
    #     损失值 = abs((预测的分类 - 标签).mean())
    #     return 损失值.to(self.设备)

    def forward(self, 输入, 标签):
        """
        前景是指要找到的文字，背景则与之相反
        :param 输入:
        :param 标签:
        :return:
        """
        总数 = 300
        标签 = 标签[0][0]
        标签为一的索引 = (标签 == 1).nonzero()[:, 0]
        标签的前景 = 标签[标签为一的索引].long()
        预测的前景 = 输入[0][标签为一的索引]
        前景的损失值 = self.损失函数(预测的前景.view(-1, 2), 标签的前景.view(-1))
        前景的损失值之和 = 前景的损失值.sum()
        前景的损失值之数量 = len(预测的前景)

        标签为零的索引 = (标签 == 0).nonzero()[:, 0]
        背景的标签 = 标签[标签为零的索引].long()
        背景的预测 = 输入[0][标签为零的索引]
        背景的损失值 = self.损失函数(背景的预测.view(-1, 2), 背景的标签.view(-1))
        背景的损失值, _ = torch.topk(背景的损失值, 总数 - 前景的损失值之数量)
        背景的损失值之和 = 背景的损失值.sum()

        分类的损失值 = (前景的损失值之和 + 背景的损失值之和) / 总数
        return 分类的损失值.to(self.设备)


class 偏移量损失值(nn.Module):
    def __init__(self, 设备, 西格玛=9.0):
        super(偏移量损失值, self).__init__()
        self.设备 = 设备
        self.西格玛 = 西格玛

    def forward(self, 输入, 标签):
        分类 = 标签[0, :, 0]  # 各个方框的类型是前景1或者背景0
        偏移量 = 标签[0, :, 1:3]  # 由索引1开始到索引2；3是指第三个元素，计数是从零开始的，即索引0，1，2
        分类为一的索引 = (分类 == 1).nonzero()[:, 0]
        偏移量 = 偏移量[分类为一的索引]
        预测的偏移量 = 输入[0][分类为一的索引]
        差距 = torch.abs(偏移量 - 预测的偏移量)
        小于1 = (差距 < 1.0 / self.西格玛).float()
        损失值 = 小于1 * 0.5 * 差距 ** 2 * self.西格玛 + torch.abs(1 - 小于1) * (差距 - 0.5 / self.西格玛)
        损失值 = torch.sum(损失值, 1)
        损失值 = torch.mean(损失值) if 损失值.numel() > 0 else torch.tensor(0.0)
        # print(偏移量)

        return 损失值.to(self.设备)
