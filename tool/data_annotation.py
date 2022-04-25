import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

import sys
sys.path.append(r'D:\Study_Data\AI\YOLO\yolox-tf2-master')

from config import cfg
from utils.utils import get_classes


classes_path        = cfg.Data.Classes_Path
classes, _          = get_classes(classes_path)


class DatasetAnnotation(object):
    _defaults = {
        #   xml文件的存储路径
        'xml_dir_path'          :   os.path.join(cfg.Data.DataSet_Path , 'Annotations'),
        #   iamges的存储路径
        'image_dir_path'        :   os.path.join(cfg.Data.DataSet_Path , 'JPEGImages'),
        #   划分ids的保存路径
        'ids_save_path'         :   os.path.join(cfg.Data.DataSet_Path , 'ImageSets/Main'),
        #   划分annotation的保存路径
        'annotation_save_path'  :   os.path.join(cfg.Data.DataSet_Path , 'ImageSets/'),
        #   数据集划分比例
        'trainval_percent'      :   cfg.Data.TrainVal_Percent,
        'train_percent'         :   cfg.Data.Train_Percent
    }

    @classmethod
    def get_default(cls, key):
        if key in cls._defaults:
            return cls._defaults[key]
        else:
            raise 'Unrecognize key:{}, Pleas Try Again'.format(key)
        
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for key, value in kwargs:
            setattr(self, key, value)


    def xml_writer(self, list_file, xml_id):
        """
        Func:
            负责解析xml,写入cls和bndbox
        Args:
            list_file:  filewriter对象
            xml_id:     xml文件id
        """

        xml_path = os.path.join(self.xml_dir_path, '{}.xml'.format(xml_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            # difficult = int(obj.find('difficult').text) if obj.find('difficult')!=None else 0

            cls = obj.find('name').text
            # if cls not in classes or difficult == 1:
            #     continue
            cls_id = classes.index(cls)

            bndbox = obj.find('bndbox')
            box = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            box.append(cls_id)

            list_file.write(' ' + ','.join(str(x) for x in box))



    def dataset_split(self):
        """
        Func:
            划分图片的ids
        """

        if not os.path.exists(self.ids_save_path):
            os.makedirs(self.ids_save_path)

        random.seed(0)
        total_xml = [xml for xml in os.listdir(self.xml_dir_path) if xml.endswith('.xml')]
        
        tv = int(self.trainval_percent * len(total_xml))
        tr = int(self.train_percent * self.trainval_percent * len(total_xml))
        trainval = random.sample(total_xml, tv)
        train = random.sample(trainval, tr)

        ftrainval = open(os.path.join(self.ids_save_path, 'trainval.txt'), 'w')
        ftrain = open(os.path.join(self.ids_save_path, 'train.txt'), 'w')
        fval = open(os.path.join(self.ids_save_path, 'validation.txt'), 'w')
        ftest = open(os.path.join(self.ids_save_path, 'test.txt'), 'w')

        for xml in tqdm(total_xml):
            xml_id = xml[:-4] + '\n'
            if xml in trainval:
                ftrainval.write(xml_id)
                if xml in train:
                    ftrain.write(xml_id)
                else:
                    fval.write(xml_id)
            else:
                ftest.write(xml_id)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()

        print('-'*50)
        print('Generate ids split')



    def generate_trainval(self):
        """
        Func:
            生成划分的annotation
        """

        if not os.path.exists(self.annotation_save_path):
            os.makedirs(self.annotation_save_path)

        save_train_path = os.path.join(self.annotation_save_path, 'annotation_train.txt')
        save_val_path = os.path.join(self.annotation_save_path, 'annotation_validation.txt')
        save_test_path = os.path.join(self.annotation_save_path, 'annotation_test.txt')

        train_path = os.path.join(self.ids_save_path, 'train.txt')
        val_path = os.path.join(self.ids_save_path, 'validation.txt')
        test_path = os.path.join(self.ids_save_path, 'test.txt')


        for save, path in [(save_train_path, train_path), (save_val_path, val_path)]:
            list_file = open(save, 'w')
            xml_ids = [x.strip() for x in open(path, 'r').readlines()]
            for xml_id in tqdm(xml_ids):
                image_path = os.path.join(self.image_dir_path, '{}.jpg'.format(xml_id))
                image_path = os.path.abspath(image_path)
                # print(image_path)
                list_file.write(image_path)
                self.xml_writer(list_file, xml_id)
                list_file.write('\n')
            list_file.close()
        
        
        image_ids = [x.strip() for x in open(test_path, 'r').readlines()]
        test_file = open(save_test_path, 'w')
        for image_id in image_ids:
            image_path = os.path.abspath(os.path.join(self.image_dir_path, '{}.jpg'.format(image_id)))
            test_file.write(image_path + '\n')
        test_file.close()


        print('-'*50)
        print('Generate annotationlines')

if __name__ == '__main__':
    data_anno = DatasetAnnotation()
    data_anno.dataset_split()
    data_anno.generate_trainval()