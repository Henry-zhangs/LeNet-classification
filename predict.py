import torchvision.transforms as transforms
import torch.nn as nn    # 神经网络库
import torch.nn.functional as F   # 函数功能库，如relu、sigmoid等等
import torch
from PIL import Image

class LeNet(nn.Module):     # 继承父类
    def __init__(self):
        super(LeNet,self).__init__()    #涉及到多继承需要，这个记住就行
        self.conv1 = nn.Conv2d(3,6,5)  # 输入（3，32，32） 输出（6，28，28）
        self.conv2 = nn.Conv2d(6,16,5)  # 输入（6，14，14） 输出（16，10，10）
        self.fc1 = nn.Linear(16*5*5,120)   # 输入16*5*5，输出120
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

        # 定义前向传播过程
    def forward(self,x):  # 输入x == 【batch，channel，height，width】
        # conv -> 激活函数 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)),2)   # 输入（3，32，32）-> （6，28，28）-> 输出（6，14，14）
        x = F.max_pool2d(F.relu(self.conv2(x)),2)   # 输入（6，14，14）-> （16，10，10）-> 输出（16，5，5）
        x = x.view(x.shape[0],-1) # -->output (1,16*5*5) * (16*5*5,120) = (1,120)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)         # 不需要softmax，交叉熵损失函数里面有
        return x


def predict(img):

    transform = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    net = LeNet()
    net.load_state_dict(torch.load('LeNet.pth'))  # 加载网络训练的参数

    im = Image.open(img)
    im = transform(im)  # 图像维度 （C，H，W）
    im = torch.unsqueeze(im,dim  = 0)  # 增加维度，第0维增加1 ，维度（1，C，H，W）

    with torch.no_grad():
        outputs  = net(im)

        predict = torch.max(outputs,dim = 1)[1].data.numpy()
        pro = torch.softmax(outputs, dim=1)
        rate = pro.max().numpy() * 100
        return classes[int(predict)] , str(rate) + '%'
