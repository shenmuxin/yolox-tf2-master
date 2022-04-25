'''
Description: 
coding: utf-8
language: python3.8, tensorflow2.6
Author: JiaHao Shum
Date: 2022-03-21 11:14:54
'''

from easydict import EasyDict as edict

cfg = edict()

#   File AbsPath:
cfg.FILE                                        =   edict()
cfg.FILE.ABSPATH                                =   r'D:\Study_Data\AI\YOLO\yolox-tf2-master'       #   注意修改自己为本地工程的文件夹路径   

#   YOLO Param：
cfg.YOLO                                        =   edict()
cfg.YOLO.Model_Load_Path                        =   'model_data\yolox_tiny(voc).h5'                 #   载入预训练的weight,为空则不载入
cfg.YOLO.Model_Save_Path                        =   './logs/Mask/'                                   #   模型的保存路径


#   Data Param:
cfg.Data                                        =   edict()
cfg.Data.DataSet_Path                           =   r'D:\Study_Data\AI\YOLO\DataSets\FaceMask\MaskFace'     #   数据集的存放地址
cfg.Data.Classes_Path                           =   'Datasets\Mask\mask_classes.txt'
cfg.Data.TrainVal_Percent                       =   0.9
cfg.Data.Train_Percent                          =   0.9 


#   Train
#   Train Path:
cfg.Train                                       =   edict()
cfg.Train.Path                                  =   edict()  
cfg.Train.Path.Tensorboard                      =   './logs/tensorboard/mask'
cfg.Train.Path.Loss                             =   './logs/loss/mask'

#   Train Param
cfg.Train.Param                                 =   edict() 
cfg.Train.Param.Input_Shape                     =   [640, 640]                              #   指定输入图片的大小
cfg.Train.Param.Eager                           =   False
cfg.Train.Param.Mosaic                          =   True
cfg.Train.Param.Consine_Anneling                =   True
cfg.Train.Param.Label_Smoothing                 =   0

cfg.Train.Param.Freeze_Train                    =   True                                    #   是否使用冻结训练
cfg.Train.Param.Init_Epoch                      =   0
cfg.Train.Param.First_Stage_Epoch               =   50      
cfg.Train.Param.First_Stage_BatchSize           =   20
cfg.Train.Param.First_Stage_LearningRate        =   1e-3
cfg.Train.Param.Second_Stage_Epoch              =   100
cfg.Train.Param.Second_Stage_BatchSize          =   8     if  cfg.Train.Param.Freeze_Train else cfg.Train.Param.First_Stage_BatchSize  # 如果OOM请调小batchsize 为4的话很容易就OOM了
cfg.Train.Param.Second_Stage_LearningRate       =   1e-4  if  cfg.Train.Param.Freeze_Train else cfg.Train.Param.First_Stage_LearningRate

cfg.Train.Param.NumWorkers                      =   1                                       #   设置线程数


#   Detect Param:
cfg.Detect                                      =   edict()
cfg.Detect.Model_Path                           =   'model_data\yolox-tiny-mask.h5'         #   进行推断所使用的模型
cfg.Detect.Crop_Save_Path                       =   'logs\crop_save'                        #   模型裁剪的保存地址

cfg.Detect.Confidence                           =   0.5                                     #   显示置信度大于0.5的预测框
cfg.Detect.NMS_IOU                              =   0.3                                     #   非极大值抑制IOU
cfg.Detect.MaxBoxes                             =   500                                     #   一张图片中最多的boxes数
cfg.Detect.LetterBox_Resize                     =   True                                    #   是否使用不失真resize


if __name__ == '__main__':
    print(cfg)