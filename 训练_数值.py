import os

import torch
from torch import optim
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import CTCLoss
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as func

from 神经网络.卷积叠加网络类 import 卷积重叠网络
from 训练_属性 import 初始化权重
from 工具.截图工具 import 读取图片

# 随机种子 = 121
# torch.manual_seed(随机种子)
# torch.backends.cudnn.deterministic = True  # 为了提升计算速度
# torch.backends.cudnn.benchmark = False  # 避免因为随机性产生出差异
# np.random.seed(随机种子)


class 调整尺寸并归一化:
    def __init__(self):
        self.转张量 = transforms.ToTensor()

    def __call__(self, 一批数据):
        标签, 图片列表 = zip(*一批数据)
        结果列表 = []
        for 图片 in 图片列表:
            # 数值训练时使用
            图片 = 图片.resize((180, 32), Image.LANCZOS)
            # 图片.show()
            # exit()
            图片 = self.转张量(图片) * 10
            图片.sub_(0.5).div_(0.5)
            结果列表.append(图片)
            # print(图片)
        图片列表 = torch.cat([img.unsqueeze(0) for img in 结果列表], 0)
        return 标签, 图片列表


class 数值数据类(Dataset):
    def __init__(self, 图片目录='训练_数值', 是否为训练=True):
        self.图片列表 = []
        self.图片名列表 = []
        self.图片目录 = '图片/' + 图片目录

        if 是否为训练:
            self.载入训练数值用图()
        else:
            self.载入测试数值用图()

        self.字符字典 = {' ': 1}
        self.索引字典 = {1: ' '}
        with open('标签/字符_数值.txt', 'r', encoding='utf8') as 文件:
            for 索引, 字符 in enumerate(文件):
                self.字符字典[字符.strip()] = 索引 + 2
                self.索引字典[索引 + 2] = 字符.strip()
            文件.close()
            # print(self.字符字典)
            # print(self.索引字典)

    def 载入训练数值用图(self):
        with open('标签/数值对应图片.txt', 'r', encoding='utf8') as 文件:
            for 一行字 in 文件:
                一行字 = 一行字.strip()
                图片名, 数值串 = tuple(一行字.split(' '))
                图片 = 读取图片(self.图片目录 + '/' + 图片名 + '.jpg')
                图片 = 图片.convert('L')  # 灰度图
                # 图片.show()
                self.图片列表.append(图片)
                self.图片名列表.append(' ' + 数值串 + ' ')
            # print(self.图片列表)
            # print(self.图片名列表)
            文件.close()

    def 载入测试数值用图(self):
        图片名列表 = os.listdir(self.图片目录)
        for 图片名 in 图片名列表:
            图片 = 读取图片(self.图片目录 + '/' + 图片名)
            图片 = 图片.convert('L')  # 灰度图
            # 图片.show()
            self.图片列表.append(图片)
            字符串 = 图片名.split('.')[0].strip()
            字符串 = ' ' + 字符串 + ' '
            self.图片名列表.append(字符串)

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
        return self.图片名列表[索引], self.图片列表[索引]


模型存储路径 = '已训练的模型/数值_模型.ckpt2'


def 评估(模型, 设备):
    测试用数据集 = 数值数据类('测试_数值', False)
    # 测试用数据集 = 数据类()
    测试用数据加载器 = DataLoader(测试用数据集, batch_size=1, collate_fn=调整尺寸并归一化())
    模型.load_state_dict(torch.load(模型存储路径))
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


if __name__ == "__main__":
    设备 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    模型 = 卷积重叠网络(1, 13, 256)
    print(模型)
    模型.to(设备)
    # 评估(已训练的模型, 设备)
    # exit()

    是否继续训练 = True
    if os.path.exists(模型存储路径) and 是否继续训练:
        模型.load_state_dict(torch.load(模型存储路径))
    else:
        模型.apply(初始化权重)

    损失值函数 = CTCLoss(reduction='sum', zero_infinity=True)
    损失值函数.to(设备)
    优化器 = optim.Adam(模型.parameters(), lr=0.0001, betas=(0.5, 0.999))

    训练用数据集 = 数值数据类()
    训练用数据加载器 = DataLoader(训练用数据集, batch_size=1, collate_fn=调整尺寸并归一化(), shuffle=True)
    总次数 = 0
    for 轮回 in range(80):
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

    torch.save(模型.state_dict(), 模型存储路径)
    print('总次数：', 总次数)
    # 评估
    评估(模型, 设备)
