"""
@file: demo
@author: 姬小野
@time: 上午12:39 2019/8/4
@version: v0.1
"""

import sys
import getopt
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


def main(argv):
    how_to_use = '''
    ----------------usage----------------
    run the demo with:
    python demo.py -m model_name -i image_name.jpg
    python demo.py --image image_name.jpg
    python demo.py -i image_name.jpg
    or use `python demo.py -h` to get help
    -----------------end-----------------
    '''
    try:
        opts, args = getopt.getopt(argv, 'hi:m:', ["image=model="])
    except getopt.GetoptError:
        print(how_to_use)
        sys.exit(2)

    if len(opts) == 0:
        print(how_to_use)
    filename = ''
    modelname = ''
    for opt, arg in opts:
        if opt == '-h':
            print(how_to_use)
        elif opt in ('-i', '--image'):
            filename = arg
        elif opt in ('-m', '--model'):
            modelname = arg
    test(filename, modelname)


if __name__ == '__main__':
    main(sys.argv[1:])