from torch import nn


class 双向长短期记忆神经网络(nn.Module):
    def __init__(self, 输入长度, 藏层长度, 输出长度):
        super(双向长短期记忆神经网络, self).__init__()
        self.重叠神经网络 = nn.LSTM(输入长度, 藏层长度, bidirectional=True)
        self.嵌入层 = nn.Linear(藏层长度 * 2, 输出长度)

    def forward(self, 输入):
        重叠, _ = self.重叠神经网络(输入)
        T, b, h = 重叠.size()
        # print('重叠:', 重叠.size())
        t_重叠 = 重叠.view(T * b, h)

        输出 = self.嵌入层(t_重叠)

        输出 = 输出.view(T, b, -1)
        return 输出


class 卷积重叠网络(nn.Module):
    def __init__(self, 通道数, 分类数, 藏层长度):
        super(卷积重叠网络, self).__init__()

        self.卷积1 = nn.Conv2d(通道数, 32, 3, 1, 1)
        self.整流线性函数1 = nn.ReLU(True)
        self.池化1 = nn.MaxPool2d(2, 2)

        self.卷积2_1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.批标准化2 = nn.BatchNorm2d(64)
        self.整流线性函数2_1 = nn.ReLU(True)
        self.卷积2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.整流线性函数2_2 = nn.ReLU(True)
        self.池化2 = nn.MaxPool2d((2, 2), (2, 2), (0, 0))

        self.卷积3 = nn.Conv2d(64, 128, 3, 1, 0)
        self.批标准化3 = nn.BatchNorm2d(128)
        self.整流线性函数3 = nn.ReLU(True)
        self.池化3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        self.卷积4 = nn.Conv2d(128, 128, 3, 1, 0)
        self.批标准化4 = nn.BatchNorm2d(128)
        self.整流线性函数4 = nn.ReLU(True)

        self.重叠神经网络 = 双向长短期记忆神经网络(128, 藏层长度, 分类数)
        # self.重叠神经网络 = nn.Sequential(
        #     双向长短期记忆神经网络(128, 藏层长度, 藏层长度),
        #     双向长短期记忆神经网络(藏层长度, 藏层长度, 分类数)
        # )

    def forward(self, 输入):
        # print('0：', 输入.size())
        x = self.池化1(self.整流线性函数1(self.卷积1(输入)))
        # print('1：', x.size())
        x = self.池化2(self.整流线性函数2_2(self.卷积2_2(self.整流线性函数2_1(self.批标准化2(self.卷积2_1(x))))))
        # print('2：', x.size())
        x = self.池化3(self.整流线性函数3(self.批标准化3(self.卷积3(x))))
        # print('3：', x.size())
        卷积 = self.整流线性函数4(self.批标准化4(self.卷积4(x)))
        # print('4：', 卷积.size())
        # exit()
        卷积 = 卷积.squeeze(2)
        卷积 = 卷积.permute(2, 0, 1)
        # print('5：', 卷积.size())
        输出 = self.重叠神经网络(卷积)
        # print('6：', 输出.size())
        return 输出
