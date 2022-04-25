import colorsys
import os
import time
import datetime

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from config import cfg
import sys
sys.path.append(cfg.FILE.ABSPATH)  

from nets.yolox_CSPdarknet import yolox_body
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox



class YOLOXDetect(object):
    _defaults = {
        #   model的路径
        "model_path"        :   cfg.Detect.Model_Path,
        "classes_path"      :   cfg.Data.Classes_Path,
        #   输入图片的大小，必须为32的倍数。
        "input_shape"       :   cfg.Train.Param.Input_Shape,
        #   所使用的YoloX的版本。tiny、s、m、l、x
        "phi"               :   'tiny',
        #   只有得分大于置信度的预测框会被保留下来
        "confidence"        :   cfg.Detect.Confidence,
        #   非极大抑制所用到的nms_iou大小
        "nms_iou"           :   cfg.Detect.NMS_IOU ,
        #   最多预测框的数量
        "max_boxes"         :   cfg.Detect.MaxBoxes,
        #   是否使用letterresize
        "letterbox_image"   :   cfg.Detect.LetterBox_Resize,
        #   crop的保存路径
        'crop_save_path'    :   cfg.Detect.Crop_Save_Path,

        #   PyQt警告标志位
        'warning'           :   None
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"



    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
    
        #   获得种类和先验框的数量
        self.class_names, self.num_classes = get_classes(self.classes_path)

        #   画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 2) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        
        self.generate()



    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        self.yolo_model = yolox_body([None, None, 3], num_classes = self.num_classes, phi = self.phi)
        self.yolo_model.load_weights(self.model_path)

        print('{} model, and classes loaded.'.format(model_path))

        #   在DecodeBox函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        self.input_image_shape = Input([2,],batch_size=1)
        inputs  = [*self.yolo_model.output, self.input_image_shape]
        outputs = Lambda(
            DecodeBox, 
            output_shape = (1,), 
            name = 'yolo_eval',
            arguments = {
                'num_classes'       : self.num_classes, 
                'input_shape'       : self.input_shape, 
                'confidence'        : self.confidence, 
                'nms_iou'           : self.nms_iou, 
                'max_boxes'         : self.max_boxes, 
                'letterbox_image'   : self.letterbox_image
             }
        )(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes

    
    def detect_image(self, image, crop = False):
        """
        Func:
            预测单张图片
        Args:
            image:      PIL的Image对象
            crop:       是否进行目标剪裁
            is_print:   是否打印检测结果
        Returns:
            image:      完成绘制的PIL Image对象

        """
        
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image       = cvtColor(image)
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #   添加上batch_size维度，并进行归一化
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        #   进行预测
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #   设置字体与边框厚度
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        #   是否进行目标的裁剪
        if crop:
            if not os.path.exists(self.crop_save_path):
                os.makedirs(self.crop_save_path)
            for i, cls_id in enumerate(out_classes):
                predicted_class             = self.class_names[int(cls_id)]
                top, left, bottom, right    = out_boxes[i]      # y1,x1,y2,x2
                score                       = out_scores[i]

                #   对边框范围进行限幅
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))

                crop_image = image.crop([left, top, right, bottom])
                fnt = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(5e-2 * crop_image.size[1] + 0.5).astype('int32'))
                crop_draw = ImageDraw.Draw(crop_image)


                crop_text       = '{0}:{1:.2f}'.format(predicted_class, score)
                crop_draw.text((0,0), crop_text, fill=self.colors[cls_id], font=fnt)
                time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d%H%M%S')
                crop_image.save(os.path.join(self.crop_save_path, 'class_' + str(predicted_class) + '_score_{:.2f}'.format(score) + '_' + time_str + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + self.crop_save_path)

        #   重置标志位
        self.warning = 0 

        #   图像绘制
        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            box             = out_boxes[i]
            score           = out_scores[i]
            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            # label = label.encode('utf-8')
            print('Class: {0}\t|\tBox: {1}'.format(label, (top, left, bottom, right)))
            print('-'*60)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0,0,0), font=font)
            del draw

            #   发出警告
            if predicted_class == 'without_mask':
                self.warning = 1
            elif predicted_class == 'incorrect':
                self.warning = 2

        return image

    def get_FPS(self, image, test_interval):
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image       = cvtColor(image)
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #   添加上batch_size维度，并进行归一化
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        
        #   将图像输入网络当中进行预测！
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time


    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image       = cvtColor(image)
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #   添加上batch_size维度，并进行归一化
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        #   将图像输入网络当中进行预测！
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        for i, c in enumerate(out_classes):
            predicted_class             = self.class_names[int(c)]
            try:
                score                   = str(out_scores[i].numpy())
            except:
                score                   = str(out_scores[i])
            top, left, bottom, right    = out_boxes[i]
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
