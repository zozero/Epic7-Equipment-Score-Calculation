from torch import nn


class 连接文本提取网络(nn.Module):
    def __init__(self):
        super(连接文本提取网络, self).__init__()

    def forward(self,输入):
        # 1*3*1080*1920
        print(输入.shape)
        pass