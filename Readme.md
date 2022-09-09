### 项目说明
为了给游戏“第七史诗“的装备计算分数，而制作的文字识别项目，它很专一，指只能做这件事。

该项目是个十分简易的项目，如果要修改它的话，可能需要注意各个文件的路径。

为了逻辑简练所以有许多函数没有用判断而是直接复制或者注释掉代码。

这是对卷积重叠神经网络（CRNN）的魔改，魔改是指减少运算而减少层数等等的修改。

为什么叫卷积重叠神经网络？因为我认为循环无法体现长短期记忆神经网络（LSTM）的使用，之前的权重会被记忆，就像垒沙堆一样。

为什么使用中文命名？因为它更易于理解，尽可能减少来自语言一词多义的问题，避免歧义。

项目已训练的预训练模型并非完美，由于需要更多的标注和图片，我选择小小的偷懒了，所以可能会出现识别有误的时候.....，这时候你可能需要自己训练下了

关于未标注的图片文件夹的图片，里面存放着我用于测试的图片，没什么货很尴尬。

部分文件夹可能需要自己手动创建。例如【图片/训练_属性】【图片/训练_数值】

gitee不能上传大量图片，所以数值对应图片.txt的内容需要删除，而且我重新删除了git版本控制的信息

连接文本提取网络（CTPN）尚未完成......

已训练的模型错误率有些高，还有待重新训练完善

### 使用说明
#### 非专业
0、运行前，请先下载好相应的python包，使用pip安装。

你可能需要以下的包[PIL](https://pillow.readthedocs.io/en/stable/installation.html)、[PyTorch](https://pytorch.org/get-started/locally/)、OpenCV，安装方法请自行研究。

1、直接截一张1920*1080的图片放到文件夹“未标注的图片"中，图片名要包含Screenshot，否则你得修改一下代码。

其他大小的图片会出问题，因为我没有训练其他大小的图片，而且截图时的位置也会有问题，该项目没有连接文本提取网络（CTPN）。

训练时我游戏使用的是高清材质包

2、然后打开使用.py来运行，之后的每次运行请务必删除”./图片/测试_属性“下的文件和”./图片/测试_数值“下的文件。（再重申一次这是个简易的项目）

3、运行后结果会在控制台输出。

#### 专业
请随意......

提示，请勿轻易使用来源不明的预训练模型，因为载入时好像会执行eval()这个函数，使得模型充满危险。
### 命名说明
    文件命名
        训练_提取   寻找图片中文字的部分，然后提取出来
        训练_属性   对属性相关的文字进行训练
        训练_数值   对数值相关的文字进行训练
        数值对应图片  这个文件用记录图片名与对应的数值
    图片文件夹命名
        训练_属性   用来训练属性相关的文字的图片
        训练_数值   用来训练数值相关的文字的图片
        测试_属性   用来测试属性相关的文字的图片
        测试_数值   用来测试数值相关的文字的图片
        
### 连接提取分类损失值函数（CTCLoss）说明
注意连接提取分类损失值函数（CTCLoss），单个数据前后得加空格。

字符.txt文件存储的是需要训练的属性字符，总共20行，再加上空格项目为21分类的任务，其中“尾”字不是训练所需要的，但必须在末尾加上一个，

用来防止连接提取分类损失值函数出现<b>负数结果</b>，而且最后一个字符不会训练到，如果不加这个”尾“就会出现<b>最后一个字符训练不到的情况</b>。

字符_数值.txt存储的是需要训练的数值字符，总共12行，再加上空格项目为13分类的任务

损失值出现负数意味着输入有问题，但哪里出了问题就只能排查了......

### 写在最后
这里除了这行字什么都没有......