import argparse
import collections
import os
import re

import torch
from PIL import Image
from torch.nn import CTCLoss
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as func

from 神经网络.卷积反复网络类 import 卷积反复网络
from 工具.截图工具 import 读取图片


# 随机种子 = 23
# torch.manual_seed(随机种子)
# torch.backends.cudnn.deterministic = True  # 为了提升计算速度
# torch.backends.cudnn.benchmark = False  # 避免因为随机性产生出差异
# np.random.seed(随机种子)


class 调整尺寸并标准化:
    def __init__(self):
        self.转张量 = transforms.ToTensor()

    def __call__(self, 一批数据):
        标签, 图片列表 = zip(*一批数据)
        结果列表 = []
        for 图片 in 图片列表:
            # 属性训练时使用
            图片 = 图片.resize((280, 32), Image.LANCZOS)
            # 图片.show()
            图片 = self.转张量(图片) * 10
            图片.sub_(0.5).div_(0.5)
            结果列表.append(图片)
            # print(图片)
        图片列表 = torch.cat([img.unsqueeze(0) for img in 结果列表], 0)
        return 标签, 图片列表


class 数据类(Dataset):
    def __init__(self, 图片目录='训练_属性'):
        self.图片列表 = []
        self.无后缀_图片名列表 = []
        self.图片目录 = '图片/' + 图片目录
        图片名列表 = os.listdir(self.图片目录)
        for 图片名 in 图片名列表:
            图片 = 读取图片(self.图片目录 + '/' + 图片名)
            图片 = 图片.convert('L')  # 灰度图
            # 图片.show()
            self.图片列表.append(图片)
            字符串 = 图片名.split('.')[0].strip()
            字符串 = re.findall(r'\D+', 字符串)[0]
            字符串 = ' ' + 字符串 + ' '
            self.无后缀_图片名列表.append(字符串)
        # print(self.图片列表)
        # print(self.无后缀_图片名列表)

        # 末尾也需要一个额外字符，用于适配损失值函数，我选择用“尾”这个字当末尾，
        # 如果没有这个字符在训练是可能会识别不到字符字典中最后一个字，而且计算损失值时会出现负数。
        self.字符字典 = {' ': 1}
        self.索引字典 = {1: ' '}
        with open('标签/字符_属性.txt', 'r', encoding='utf8') as 文件:
            for 索引, 字符 in enumerate(文件):
                self.字符字典[字符.strip()] = 索引 + 2
                self.索引字典[索引 + 2] = 字符.strip()
            文件.close()

    def 编码文字(self, 字符串列表):
        # print(字符串列表)
        长度 = [len(c) for c in 字符串列表]
        字符串 = ''.join(字符串列表)

        编码列表 = [
            self.字符字典[字符] for 字符 in 字符串
        ]
        # print(编码列表, 长度)
        return torch.IntTensor(编码列表), torch.IntTensor(长度)

    def 解码文字(self, 数据, 尺寸):
        # print(数据.numel())
        # print(尺寸.numel())
        字符列表 = []
        if 尺寸.numel() == 1:
            尺寸 = 尺寸[0]
            assert 数据.numel() == 尺寸, "数据和尺寸长度匹配"
            for i in range(尺寸):
                if 数据[i] != 0 and (not (i > 0 and 数据[i - 1] == 数据[i])):
                    字符列表.append(self.索引字典[int(数据[i])])
            print(字符列表)
        return 字符列表

    def __len__(self):
        return len(self.图片列表)

    def __getitem__(self, 索引):
        return self.无后缀_图片名列表[索引], self.图片列表[索引]


def 初始化权重(模型):
    类名 = 模型.__class__.__name__

    if 类名.find('Conv') != -1:
        模型.weight.data.normal_(0.0, 0.02)
    elif 类名.find('BatchNorm') != -1:
        模型.weight.data.normal_(1.0, 0.02)
        模型.bias.data.fill_(0)


def 主要():
    设备 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    模型 = 卷积反复网络(1, 21, 256)
    print(模型)
    模型.to(设备)
    # 评估(模型, 设备)
    # exit()

    是否继续训练 = False
    if os.path.exists('已训练的模型/最新模型wk3.ckpt') and 是否继续训练:
        模型.load_state_dict(torch.load('已训练的模型/最新模型wk3.ckpt'))
    else:
        模型.apply(初始化权重)

    损失值函数 = CTCLoss(reduction='sum', zero_infinity=True)
    损失值函数.to(设备)

    优化器 = optim.Adam(模型.parameters(), lr=0.0001, betas=(0.5, 0.999))

    训练用数据集 = 数据类()
    训练用数据加载器 = DataLoader(训练用数据集, batch_size=1, collate_fn=调整尺寸并标准化(), shuffle=True)
    最低损失值 = float("inf")
    总次数 = 0
    for 轮回 in range(160):
        for 图片名列表, 图片列表 in 训练用数据加载器:
            图片列表 = 图片列表.to(设备)
            for p in 模型.parameters():
                p.requires_grad = True
            模型.train()

            优化器.zero_grad()
            字符列表, 长度列表 = 训练用数据集.编码文字(图片名列表)
            预测 = 模型(图片列表)
            预测_尺寸 = Variable(torch.IntTensor([预测.size(0)]))
            预测 = 预测.log_softmax(2)
            损失 = 损失值函数(预测, 字符列表, 预测_尺寸, 长度列表)
            模型.zero_grad()
            损失.backward()
            优化器.step()
            损失值 = 损失.item()
            print(f"损失值：{损失值:.8f}")
            # if 最低损失值 > 损失.item():
            #     torch.save(已训练的模型.state_dict(), '已训练的模型/最新模型w1.ckpt')
            #     最低损失值 = 损失.item()
            总次数 += 1
    torch.save(模型.state_dict(), '已训练的模型/最新模型wk4.ckpt')
    print('总次数：', 总次数)
    # 评估
    评估(模型, 设备)


def 评估(模型, 设备):
    测试用数据集 = 数据类('测试_属性')
    # 测试用数据集 = 数据类()
    测试用数据加载器 = DataLoader(测试用数据集, batch_size=1, collate_fn=调整尺寸并标准化())
    模型.load_state_dict(torch.load('已训练的模型/当前最佳模型.ckpt'))
    模型.eval()
    for 图片名列表, 图片列表 in 测试用数据加载器:
        图片列表 = 图片列表.to(设备)
        验证 = 模型(图片列表)
        验证 = func.log_softmax(验证, 2)
        _, 验证 = 验证.max(2)
        验证 = 验证.transpose(1, 0).contiguous().view(-1)
        验证_尺寸 = Variable(torch.IntTensor([验证.size(0)]))
        测试用数据集.解码文字(验证.data, 验证_尺寸.data)
        # exit()


if __name__ == '__main__':
    参数 = argparse.ArgumentParser(description="第七史诗 装备分数计算")
    参数.add_argument('--配置', default=None, type=str, help="项目配件文件路径，默认为None")
    参数.add_argument('--恢复', default=None, type=str, help="最后检查点文件路径，默认为None")
    参数.add_argument('--设备', default=None, type=str, help="要启用图形处理单元的索引，默认为全部")

    # 可以更改json文件中的参数直接用命令的方式
    自定义参数 = collections.namedtuple('自定义参数', ['标记', '类型', '目标'])
    选项 = [
        自定义参数(['--学习率'], 类型=float, 目标='优化器;参数;学习率'),
        自定义参数(['--批长度'], 类型=int, 目标='数据加载器;参数;批长度'),
    ]

    # 为了简化暂时不写该代码
    # 配置=
    主要()
