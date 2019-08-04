"""
@file: test_one_image
@author: 姬小野
@time: 上午1:08 2019/8/4
@version: v0.1
"""
import torch
from PIL import Image
import numpy as np
from torchvision import datasets, transforms, models
from preprocess import classes_cn


def softmax(x):
    x = x.detach().numpy()
    return np.exp(x) / np.sum(np.exp(x))


def test(image, model_name='my_vgg16_1epochs.pth'):
    model = torch.load(model_name)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])
    img = Image.open(image)
    img = transform(img)
    # print(img)
    # print(img.shape)
    img = img.view(1, 3, 224, 224)
    # print(img.shape)
    output = model(img)
    # print(softmax(output))
    result = np.argmax(softmax(output)[0])
    # print(result)
    print(classes_cn[result])