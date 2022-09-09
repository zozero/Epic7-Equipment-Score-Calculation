from torch.utils.data import Dataset


class 数据处理(Dataset):
    def __init__(self):
        with open('标签/坐标/test.txt', 'r', encoding='utf8') as 文本:
            for 段 in 文本.readlines():
                print(段)


if __name__ == '__main__':
    训练用数据 = 数据处理()
    pass
