from functools import wraps

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Layer, MaxPooling2D,
                                     ZeroPadding2D, Lambda, UpSampling2D, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from utils.utils import compose
from nets.yolox_Train import get_yolo_loss

class SiLU(Layer):
    """
    Math:
        SiLU(x) = x * sigmoid(x)
    """
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

class Focus(Layer):
    """
    Func:
        将长宽/2 通道数*4
    Args:
        x shape(b, w, h, c)第一个是batch轴
    """
    def __init__(self):
        super(Focus, self).__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2 if input_shape[1] != None else input_shape[1], input_shape[2] // 2 if input_shape[2] != None else input_shape[2], input_shape[3] * 4)

    def call(self, x):
        return tf.concat(
            [x[...,  ::2,  ::2, :],
             x[..., 1::2,  ::2, :],
             x[...,  ::2, 1::2, :],
             x[..., 1::2, 1::2, :]],
             axis=-1
        )


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """
    Func:
        单次卷积DarknetConv2D, 如果步长为2则自己设定padding方式。
    """
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)



def DarknetConv2D_BN_SiLU(*args, **kwargs):
    """
    Func:
        DarknetConv2D + BatchNormalization + SiLU激活函数
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(momentum = 0.97, epsilon = 0.001, name = kwargs['name'] + '.bn'),
        SiLU())




def Bottleneck(x, out_channels, shortcut=True, weight_decay=5e-4, name = ""):
    """
    Func:
        建立两种Bottleneck
        DarknetConv2D_BN_SiLU * 2 + shortcut(可选)
    """
    y = compose(
            DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1'),
            DarknetConv2D_BN_SiLU(out_channels, (3, 3), weight_decay=weight_decay, name = name + '.conv2'))(x)
    if shortcut:
        y = Add()([x, y])
    return y


def CSPLayer(x, num_filters, num_blocks, shortcut=True, expansion=0.5, weight_decay=5e-4, name=""):
    """
    Func:
        两种CSPLayer
    """

    hidden_channels = int(num_filters * expansion)  # hidden channels   
    #   主干部分会对num_blocks进行循环，循环内部是残差结构。
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    
    #   然后建立一个大的残差边shortconv、这个大残差边绕过了很多的残差结构
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name = name + '.m.' + str(i))
    #   将大残差边再堆叠回来
    route = Concatenate()([x_1, x_2])
    #   最后对通道数进行整合
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), weight_decay=weight_decay, name = name + '.conv3')(route)


def SPPBottleneck(x, out_channels, weight_decay=5e-4, name = ""):
    """
    Func:
        构建SPP,不同尺度的最大池化堆叠
    """

    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)
    return x


def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, weight_decay=5e-4, name = ""):
    """
    Func: 
        resblock_body -> CSPLayer + Conv2D_BN_SiLU
    """
    
    #   利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(num_filters, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = name + '.0')(x)
    if last:
        x = SPPBottleneck(x, num_filters, weight_decay=weight_decay, name = name + '.1')
    return CSPLayer(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, weight_decay=weight_decay, name = name + '.1' if not last else name + '.2')




def backbone_darkent53(x, dep_mul, wid_mul, weight_decay=5e-4):
    """
    Func:
        backbone Darknet53
    Returns:
        feat1: 80, 80, 128
        feat2: 40, 40, 256
        feat3: 20, 20, 512
    """

    base_channels   = int(wid_mul * 64)             # YOLOX-S: 32
    base_depth      = max(round(dep_mul * 3), 1)    # YOLOX-S: 1
    # 640, 640, 3 => 320, 320, 12
    x = Focus()(x)
    #   Num Filter: Conv 32     1, 320, 320, 12 => 1, 320, 320, 32
    x = DarknetConv2D_BN_SiLU(base_channels, (3, 3), weight_decay=weight_decay, name = 'backbone.backbone.stem.conv')(x)
    #   Num Filter: Conv 64     1, 320, 320, 32 => 1, 160, 160, 64 
    x = resblock_body(x, base_channels * 2, base_depth, weight_decay=weight_decay, name = 'backbone.backbone.dark2')
    #   Num Filter: Conv 128    1, 160, 160, 64 => 1, 80, 80, 128
    x = resblock_body(x, base_channels * 4, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark3')
    feat1 = x
    #   Num Filter: Conv 256    1, 80, 80, 128  => 1, 40, 40, 256
    x = resblock_body(x, base_channels * 8, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark4')
    feat2 = x
    #   Num Filter: Conv 512    1, 40, 40, 256  => 1, 20, 20, 512
    x = resblock_body(x, base_channels * 16, base_depth, shortcut=False, last=True, weight_decay=weight_decay, name = 'backbone.backbone.dark5')
    feat3 = x
    return feat1,feat2,feat3




def yolox_body(input_shape, num_classes, phi, weight_decay=5e-4):
    """
    Func: 
        Backbone + Neck + DecoupledHead
        Neck = FPN + PANet
    """

    depth_dict      = {'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict      = {'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    depth, width    = depth_dict[phi], width_dict[phi]
    in_channels     = [256, 512, 1024]
    
    inputs      = Input(input_shape)
    
    #   feat1 80, 80, 128
    #   feat2 40, 40, 256
    #   feat3 20, 20, 512
    feat1, feat2, feat3 = backbone_darkent53(inputs, depth, width)

    #   FPNet
    #   Num Filter Conv 256     1, 20, 20, 512 => 1, 20, 20, 256
    P5          = DarknetConv2D_BN_SiLU(int(in_channels[1] * width), (1, 1), weight_decay=weight_decay, name = 'backbone.lateral_conv0')(feat3)  
    P5_upsample = UpSampling2D()(P5)  # 1, 20, 20, 256 => 1, 40, 40, 256
    P5_upsample = Concatenate(axis = -1)([P5_upsample, feat2])  
    #   Num Filter Conv 256     1, 40, 40, 512 => 1, 40, 40, 256
    P5_upsample = CSPLayer(P5_upsample, int(in_channels[1] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_p4')  # 1024->512/16

    #   Num Filter Conv 128     1, 40, 40, 256 => 1, 40, 40, 128
    P4          = DarknetConv2D_BN_SiLU(int(in_channels[0] * width), (1, 1), weight_decay=weight_decay, name = 'backbone.reduce_conv1')(P5_upsample)  # 512->256/16
    P4_upsample = UpSampling2D()(P4)  # 1, 40, 40, 128 => 1, 80, 80, 128
    P4_upsample = Concatenate(axis = -1)([P4_upsample, feat1])  
    #   Num Filter Conv 128     1, 80, 80, 256 => 1, 80, 80, 128
    P3_out      = CSPLayer(P4_upsample, int(in_channels[0] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_p3')  # 1024->512/16

    #   PANet
    P3_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P3_out)
    #   Num Filter Conv 128     1, 80, 80, 128 => 1, 40, 40, 128
    P3_downsample   = DarknetConv2D_BN_SiLU(int(in_channels[0] * width), (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.bu_conv2')(P3_downsample)  # 256->256/16
    P3_downsample   = Concatenate(axis = -1)([P3_downsample, P4])
    #   Num Filter Conv 256     1, 40, 40, 256 => 1, 40, 40, 256  
    P4_out          = CSPLayer(P3_downsample, int(in_channels[1] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_n3')  # 1024->512/16

    P4_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P4_out)
    #   Num Filter Conv 128     1, 40, 40, 256 => 1, 20, 20, 256
    P4_downsample   = DarknetConv2D_BN_SiLU(int(in_channels[1] * width), (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.bu_conv1')(P4_downsample)  # 256->256/16
    P4_downsample   = Concatenate(axis = -1)([P4_downsample, P5])
    #   Num Filter Conv 512     1, 20, 20, 512 => 1, 20, 20, 512  
    P5_out          = CSPLayer(P4_downsample, int(in_channels[2] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_n4')  # 1024->512/16

    #   P3_out 1, 80, 80, 128
    #   P4_out 1, 40, 40, 256
    #   P5_out 1, 20, 20, 512
    fpn_outs    = [P3_out, P4_out, P5_out]
    yolo_outs   = []

    #   Decoupled Head
    for i, out in enumerate(fpn_outs):
        # 利用1x1卷积进行通道整合
        stem    = DarknetConv2D_BN_SiLU(int(256 * width), (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.stems.' + str(i))(out)
        
        # 利用3x3卷积进行特征提取
        cls_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_convs.' + str(i) + '.0')(stem)
        cls_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_convs.' + str(i) + '.1')(cls_conv)
        
        #   判断特征点所属的种类
        #   1, 80, 80, num_classes
        #   1, 40, 40, num_classes
        #   1, 20, 20, num_classes
        cls_pred = DarknetConv2D(num_classes, (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_preds.' + str(i))(cls_conv)

        # 利用3x3卷积进行特征提取
        reg_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_convs.' + str(i) + '.0')(stem)
        reg_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_convs.' + str(i) + '.1')(reg_conv)
        
        #   特征点的回归系数
        #   1, 80, 80, 4
        #   1, 40, 40, 4
        #   1, 20, 20, 4
        reg_pred = DarknetConv2D(4, (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_preds.' + str(i))(reg_conv)
        
        #   判断特征点是否有对应的物体
        #   1, 80, 80, 1
        #   1, 40, 40, 1
        #   1, 20, 20, 1
        obj_pred = DarknetConv2D(1, (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.obj_preds.' + str(i))(reg_conv)

        #   yolox的输出是个list[]包含三个head的输出
        #   1, 80, 80, 4 + 1 + num_classes
        #   1, 40, 40, 4 + 1 + num_classes
        #   1, 20, 20, 4 + 1 + num_classes
        output   = Concatenate(axis = -1)([reg_pred, obj_pred, cls_pred])
        yolo_outs.append(output)
    return Model(inputs, yolo_outs)



def get_train_model(model_body, input_shape, num_classes):
    y_true = [Input(shape = (None, 5))]
    model_loss  = Lambda(
        get_yolo_loss(input_shape, len(model_body.output), num_classes), 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
    )([*model_body.output, *y_true])
    
    model       = Model([model_body.input, *y_true], model_loss)
    return model
