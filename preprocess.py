"""
@file: preprocess
@author: 姬小野
@time: 下午11:40 2019/8/3
@version: v0.1
"""
import torch
from torch import nn, optim
import os
from torchvision import datasets, transforms, models

train_on_gpu = torch.cuda.is_available()
# 关闭GPU, 因为我的GPU太渣了, 做迁移学习显存不够, 加载个模型就差不多满了, 训练不了...
train_on_gpu = False

# if train_on_gpu:
#     print("GPU is available, so I use GPU")
# else:
#     print("GPU is not available, so I use CPU.")

# 加载数据(训练和测试)的目录
data_dir = 'flower_photos/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

# 五种花的类别名, 也是存储图像的根目录的名称
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
classes_cn = ['菊花', '蒲公英', '玫瑰', '向日葵', '郁金香']

# 读取数据, 并进行格式的转化, 224*224大小
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)
# print(f'len of train_data: {len(train_data)}, len of test_data: {len(test_data)}')

# 规格化输入数据
batch_size = 20

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 加载模型, 已经被我放到~/.cache/torch/checkpoints/vgg16-397923af.pth
vgg16 = models.vgg16(pretrained=True)
# print(type(vgg16))

# 冻结features的参数, 只修改后面全连接层的参数
for param in vgg16.features.parameters():
    param.requires_grad = False

#
last_layer = nn.Linear(vgg16.classifier[6].in_features, len(classes))
vgg16.classifier[6] = last_layer

# 设置模型模式
if train_on_gpu:
    vgg16.cuda()
else:
    vgg16.cpu()

# 设置损失度量, 优化函数

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.001)