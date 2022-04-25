'''
Description: 
coding: utf-8
language: python3.8, tensorflow2.6
Author: JiaHao Shum
Date: 2022-03-23 16:42:47
'''

import sys
sys.path.append(r'D:\Study_Data\AI\YOLO\yolox-tf2-master')
from config import cfg

from nets.yolox_CSPdarknet import yolox_body

if __name__ == "__main__":
    input_shape = [640, 640, 3]
    num_classes = 80

    model = yolox_body(input_shape, num_classes, 's')

    model.summary()
    for i in range(len(model.layers)): 
        print(i, model.layers[i].name)