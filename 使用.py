import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as func

from 神经网络.卷积重叠网络类 import 卷积重叠网络
from 工具.截图工具 import 属性和数值图片截取
from 训练_属性 import 数据类
from 训练_属性 import 调整尺寸并归一化 as 属性_调整尺寸并归一化
from 训练_数值 import 调整尺寸并归一化 as 数值_调整尺寸并归一化, 数值数据类


def 使用属性模型(模型, 设备, 模型路径):
    测试用数据集 = 数据类('测试_属性')
    测试用数据加载器 = DataLoader(测试用数据集, batch_size=1, collate_fn=属性_调整尺寸并归一化())
    模型.load_state_dict(torch.load(模型路径))
    模型.eval()
    字符串列表 = []
    for 图片名列表, 图片列表 in 测试用数据加载器:
        图片列表 = 图片列表.to(设备)
        验证 = 模型(图片列表)
        验证 = func.log_softmax(验证, 2)
        _, 验证 = 验证.max(2)
        验证 = 验证.transpose(1, 0).contiguous().view(-1)
        验证_尺寸 = Variable(torch.IntTensor([验证.size(0)]))
        字符串列表.append(测试用数据集.解码文字(验证.data, 验证_尺寸.data))

    属性列表 = []
    for 值 in 字符串列表:
        属性列表.append(''.join(值).strip())
    return 属性列表


def 使用数值模型(模型, 设备, 模型路径):
    测试用数据集 = 数值数据类('测试_数值', 是否为训练=False)
    测试用数据加载器 = DataLoader(测试用数据集, batch_size=1, collate_fn=数值_调整尺寸并归一化())
    模型.load_state_dict(torch.load(模型路径))
    模型.eval()
    字符串列表 = []
    for 图片名列表, 图片列表 in 测试用数据加载器:
        图片列表 = 图片列表.to(设备)
        验证 = 模型(图片列表)
        验证 = func.log_softmax(验证, 2)
        _, 验证 = 验证.max(2)
        验证 = 验证.transpose(1, 0).contiguous().view(-1)
        验证_尺寸 = Variable(torch.IntTensor([验证.size(0)]))
        字符串列表.append(测试用数据集.解码文字(验证.data, 验证_尺寸.data))

    数值列表 = []
    for 值 in 字符串列表:
        数值列表.append(''.join(值).strip())
    return 数值列表


if __name__ == '__main__':
    属性和数值图片截取()

    设备 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    属性模型 = 卷积重叠网络(1, 21, 256)
    print(属性模型)
    属性模型.to(设备)
    属性列表 = 使用属性模型(属性模型, 设备, 模型路径='已训练的模型/当前属性最佳模型.ckpt')

    数值模型 = 卷积重叠网络(1, 13, 256)
    print(数值模型)
    数值模型.to(设备)
    数值列表 = 使用数值模型(数值模型, 设备, 模型路径='已训练的模型/当前数值最佳模型.ckpt')

    # 装备分数的基础倍数，固定值会在代码中额外处理
    倍数字典 = {'速度': 2, '防御力': 1, '效果抗性': 1, '效果命中': 1, '暴击率': 1.5, '暴击伤害': 1.1, '攻击力': 1, '生命值': 1}

    分数总计 = 0
    for 属性, 数值 in zip(属性列表, 数值列表):
        print(f"{属性}  {数值}")
        if '%' in 数值:
            数值 = 数值.replace('%', '')
        else:
            if '防御力' == 属性:
                倍数字典[属性] = 0.167
            elif '攻击力' == 属性:
                倍数字典[属性] = 0.1
            elif '生命值' == 属性:
                倍数字典[属性] = 0.0167
        分数总计 += 倍数字典[属性] * int(数值)
    print("分数总计：", 分数总计)
