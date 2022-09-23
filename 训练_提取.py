import os
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from 神经网络.连接文本提取网络类 import 连接文本提取网络, 分类损失值, 数据处理, 偏移量损失值

模型存储路径 = '已训练的模型/提取_模型11.ckpt'


def 评估(模型, 设备):
    图片目录 = '图片/测试_提取/'
    标签目录 = '标签/测试_坐标/'
    测试用数据集 = 数据处理(图片目录, 标签目录)
    测试用数据加载器 = DataLoader(测试用数据集, batch_size=1)
    模型.load_state_dict(torch.load(模型存储路径))
    模型.eval()
    for 索引, (原图, 图片, 标签, 偏移量) in enumerate(测试用数据加载器):
        原图 = np.asarray(原图[0])
        高, 宽 = 原图.shape[:2]

        图片 = 图片.to(设备)
        标签 = 标签.to(设备)
        偏移量 = 偏移量.to(设备)

        输出的分类, 输出的偏移量 = 模型(图片)
        # 标签概率 = F.log_softmax(输出的分类, dim=-1).cpu().detach().numpy()
        # 标签概率 = 输出的分类.cpu().detach().numpy()
        标签概率 = F.softmax(输出的分类, dim=-1)
        print(torch.topk(标签概率[0, :, 1], 100))
        标签概率 = 标签概率.cpu().detach().numpy()
        输出的偏移量 = 输出的偏移量.cpu().detach().numpy()

        满图的小方框 = 测试用数据集.生成满图的小方框((int(1080 / 35), int(1920 / 16)))
        # 微调好的满图小方框 = 测试用数据集.还原微调的小方框(满图的小方框, 输出的偏移量)
        # 微调好的满图小方框 = 测试用数据集.处理溢出值(微调好的满图小方框, [高, 宽])
        # 原图拷贝1=原图.copy()
        # 测试用数据集.画出方框(原图拷贝1, 微调好的满图小方框.astype(np.int32))

        前景概率大于阈值的索引 = np.where(标签概率[0, :, 1] > 0.1)[0]
        # 前景概率大于阈值的小方框 = 微调好的满图小方框[前景概率大于阈值的索引, :]
        前景概率大于阈值的小方框 = 满图的小方框[前景概率大于阈值的索引, :]
        前景概率大于阈值的小方框 = 前景概率大于阈值的小方框.astype(np.int32)
        预测的索引 = 测试用数据集.计算大于最小尺寸的索引(前景概率大于阈值的小方框)
        预测的小方宽 = 前景概率大于阈值的小方框[预测的索引]
        原图拷贝2 = 原图.copy()
        测试用数据集.画出方框(原图拷贝2, 预测的小方宽)

        标签2 = 标签.cpu().detach().numpy()
        a = np.where(标签2[:, :, :] > 0)[2]
        标签小方框 = 满图的小方框[a]
        原图拷贝3 = 原图.copy()
        测试用数据集.画出方框(原图拷贝3, 标签小方框)
        # exit()


random_seed = 200
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # 避免因为随机性产生出差异


def 初始化权重(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    图片目录 = '图片/训练_提取/'
    标签目录 = '标签/坐标/'
    训练用数据 = 数据处理(图片目录, 标签目录)
    # 训练用数据.__getitem__(2)
    训练用数据加载器 = DataLoader(训练用数据, batch_size=1, shuffle=False)

    设备 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    模型 = 连接文本提取网络()
    模型.to(设备)
    评估(模型, 设备)
    exit()
    # 是否继续训练 = True
    是否继续训练 = False
    if os.path.exists(模型存储路径) and 是否继续训练:
        模型.load_state_dict(torch.load(模型存储路径))
    else:
        模型.apply(初始化权重)
        pass
    分类的损失函数 = 分类损失值(设备)
    偏移量的损失函数 = 偏移量损失值(设备)

    # 优化器 = optim.SGD(模型.parameters(), lr=0.0001, momentum=0.9)
    优化器 = optim.SGD(模型.parameters(), lr=0.001)
    # 调度器 = optim.lr_scheduler.StepLR(优化器, step_size=10, gamma=0.1)

    总次数 = 0
    for 轮回 in range(80):
        模型.train()
        for 索引, (_, 图片, 标签, 偏移量) in enumerate(训练用数据加载器):
            图片 = 图片.to(设备)
            标签 = 标签.to(设备)
            偏移量 = 偏移量.to(设备)

            优化器.zero_grad()

            # 分类：不同的高度是不同的类型，每个高度分为前景和背景
            # 偏移量：相当于每个小方框的偏移量
            输出的分类, 输出的偏移量 = 模型(图片)
            分类损失值 = 分类的损失函数(输出的分类, 标签)
            # 偏移量损失值 = 偏移量的损失函数(输出的偏移量, 偏移量)
            损失值 = 分类损失值
            # 损失值 = 分类损失值 + 偏移量损失值
            损失值.backward()
            优化器.step()
            总次数 += 1
            print(f'总次数：{总次数}，损失值：{损失值.item():.8f}')
        # exit()
        # 调度器.step()
    print(f'总次数：{总次数}')
    torch.save(模型.state_dict(), 模型存储路径)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 评估
    评估(模型, 设备)
