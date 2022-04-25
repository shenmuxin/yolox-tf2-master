import datetime
import os
from functools import partial

from config import cfg
import sys
sys.path.append(cfg.FILE.ABSPATH)

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets.yolox_CSPdarknet import yolox_body, get_train_model
from nets.yolox_Train import get_lr_scheduler, get_yolo_loss
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint, WarmUpCosineDecayScheduler)
from utils.dataloader import YoloDatasets
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch


#   设置GPU内存动态增长
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":

    #   是否使用eager模式进行训练
    eager = False
    #   数据集类别路径
    #   预训练模型路径
    classes_path        =   cfg.Data.Classes_Path
    model_path          =   cfg.YOLO.Model_Load_Path
    #   模型的输入大小
    input_shape         =   cfg.Train.Param.Input_Shape
    #   phi             所使用的YoloX的版本。tiny、s、m、l、x
    phi                 =   'tiny'
    #   是否使用mosaic数据增强
    mosaic              =   cfg.Train.Param.Mosaic
    #   冻结训练
    #   Init_Epoch          模型当前开始的训练世代,其值可以大于Freeze_Epoch,如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段,直接从60代开始,并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    Init_Epoch          =   cfg.Train.Param.Init_Epoch
    Freeze_Epoch        =   cfg.Train.Param.First_Stage_Epoch 
    Freeze_batch_size   =   cfg.Train.Param.First_Stage_BatchSize            #   s-16 tiny-20
    #   解冻阶段训练参数
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    UnFreeze_Epoch      =   cfg.Train.Param.Second_Stage_Epoch
    Unfreeze_batch_size =   cfg.Train.Param.Second_Stage_BatchSize             #   s-6 tiny-8
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    #                   如果设置Freeze_Train=False,建议使用优化器为sgd
    Freeze_Train        =   cfg.Train.Param.Freeze_Train
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率,默认为最大学习率的0.01
    Init_lr             =   1e-2
    Min_lr              =   Init_lr * 0.01
    #   optimizer_type  使用到的优化器种类,可选的有adam、sgd
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减,可防止过拟合
    optimizer_type      =   "sgd"
    momentum            =   0.937
    weight_decay        =   5e-4
    #   lr_decay_type   使用到的学习率下降方式,可选的有'step'、'cos'
    lr_decay_type       =   'cos'
    #   save_period     多少个epoch保存一次权值,默认每个世代都保存
    save_period         =   1
    #   save_dir        权值与日志文件保存的文件夹
    save_dir            =   'logs'
    #   num_workers     用于设置是否使用多线程读取数据,1代表关闭多线程
    num_workers         =   1
    #   train_annotation_path   训练图片路径和标签
    #   val_annotation_path     验证图片路径和标签
    train_annotation_path   = os.path.join(cfg.Data.DataSet_Path, 'ImageSets/annotation_train.txt')
    val_annotation_path     = os.path.join(cfg.Data.DataSet_Path, 'ImageSets/annotation_validation.txt')

    #   获取classes
    class_names, num_classes = get_classes(classes_path)

    
    #   创建yolo模型
    model_body  = yolox_body([None, None, 3], num_classes = num_classes, phi = phi, weight_decay=weight_decay)
    if model_path != '':
        
        #   载入预训练权重
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)

    if not eager:
        model       = get_train_model(model_body, input_shape, num_classes)
    else:
        yolo_loss   = get_yolo_loss(input_shape, len(model_body.output), num_classes)
        
    #   读取数据集
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    
    #   主干特征提取网络特征通用,冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    if True:
        if Freeze_Train:
            freeze_layers = {'tiny': 125, 's': 125, 'm': 179, 'l': 234, 'x': 290}[phi]
            for i in range(freeze_layers): model_body.layers[i].trainable = False
            print('-'*100)
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
            print('-'*100)
            
        batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        

        nbs     = 64
        Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小,无法进行训练,请扩充数据集。')

        train_dataloader    = YoloDatasets(train_lines, input_shape, batch_size, num_classes, Init_Epoch, UnFreeze_Epoch, mosaic = mosaic, train = True)
        val_dataloader      = YoloDatasets(val_lines, input_shape, batch_size, num_classes, Init_Epoch, UnFreeze_Epoch, mosaic = False, train = False)

        optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        if eager:
            start_epoch     = Init_Epoch
            end_epoch       = UnFreeze_Epoch
            UnFreeze_flag   = False

            gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            loss_history    = LossHistory(log_dir)

            #   开始模型训练
            for epoch in range(start_epoch, end_epoch):
    
                #   如果模型有冻结学习部分
                #   则解冻,并设置参数
                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_size      = Unfreeze_batch_size

                    #   判断当前batch_size与64的差别,自适应调整学习率
                    nbs     = 64
                    Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                    Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)
        
                    #   获得学习率下降的公式
                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                    for i in range(len(model_body.layers)): 
                        model_body.layers[i].trainable = True

                    epoch_step      = num_train // batch_size
                    epoch_step_val  = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小,无法继续进行训练,请扩充数据集。")

                    train_dataloader.batch_size    = batch_size
                    val_dataloader.batch_size      = batch_size

                    gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
                    gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

                    gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    
                    UnFreeze_flag = True

                lr = lr_scheduler_func(epoch)
                K.set_value(optimizer.lr, lr)

                fit_one_epoch(model_body, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                                            end_epoch, input_shape, num_classes, save_period, save_dir)
                            
                train_dataloader.on_epoch_end()
                val_dataloader.on_epoch_end()
        else:   #   不使用eager
            start_epoch = Init_Epoch
            end_epoch   = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch

            model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            
            #   训练参数的设置
            #   logging         用于设置tensorboard的保存地址
            #   checkpoint      用于设置权值保存的细节,period用于修改多少epoch保存一次
            #   lr_scheduler       用于设置学习率下降的方式
            #   early_stopping  用于设定早停,val_loss多次不下降自动结束训练,表示模型基本收敛
            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            logging         = TensorBoard(log_dir)
            loss_history    = LossHistory(log_dir)
            checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
            early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            callbacks       = [logging, loss_history, checkpoint, lr_scheduler]

            if start_epoch < end_epoch:
                print('-'*100)
                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                print('-'*100)
                model.fit(
                    train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )

            #   如果模型有冻结学习部分
            #   则解冻,并设置参数

            if Freeze_Train:
                batch_size  = Unfreeze_batch_size
                start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
                end_epoch   = UnFreeze_Epoch
                    
                
                #   判断当前batch_size与64的差别,自适应调整学习率
                nbs     = 64
                Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)
    
                #   获得学习率下降的公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
                callbacks       = [logging, loss_history, checkpoint, lr_scheduler]
                
                #   解冻
                for i in range(len(model_body.layers)): 
                    model_body.layers[i].trainable = True
                model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小,无法继续进行训练,请扩充数据集。")

                train_dataloader.batch_size    = Unfreeze_batch_size
                val_dataloader.batch_size      = Unfreeze_batch_size
                print('-'*100)
                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                print('-'*100)
                model.fit(
                    train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )
