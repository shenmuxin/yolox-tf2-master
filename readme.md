<!--
 * @Description: 
 * @coding: utf-8
 * @language: python3.8, tensorflow2.6
 * @Author: JiaHao Shum
 * @Date: 2022-04-17 11:03:30
-->

## 毕业设计-东北大学

**Author: Shen Jiahao**
### 一、项目简介
---
基于YOLOX-Tiny的口罩佩戴实时检测，附数据集，PyQt程序界面，能够进行模型训练、推断。
### 二、环境依赖
---
python == 3.8
tensorflow == 2.6
PySide2 == 5.15

### 三、文件结构说明
---
1. Dataset:      用于存放数据集
2. logs          用于存放训练日志
3. model_data    存放预训练模型
4. nets          存放网络结构
5. utils         存放后处理函数等
6. PyQt          打包的程序
7. config        设置默认配置
8. predict       未打包的预测程序
9. train         训练函数

### 四、模型训练说明
---
1. 数据集采用Pascal VOC格式
2. 模型默认采用迁移学习载入预训练模型
3. 训练完成模型保存在logs中


### 五、模型推断
---
1. 使用predict.py进行模型推断
2. 使用PyQt/Detection_QT.py进行打包程序运行
3. 效果如下：
<img width="448" alt="image" src="https://user-images.githubusercontent.com/87429023/165014136-14d8895d-a037-4d7c-a657-24b17b5cc264.png">


### 六、使用说明
---
1. 在config.py文件中进行参数配置
2. train.py文件中进行模型训练
3. predict.py文件中直接进行模型推断
4. 在PyQt/Detection_QT.py中进行打包模型的运行
---
### 七、数据集下载
自制数据集[MaskFace数据集](https://pan.baidu.com/s/1Y3gv9vVZdG7fwwxvgLhPvQ) 提取码：ipyn

自制数据集包含三个类别：1、正确佩戴口罩，2、错误佩戴口罩，3、未佩戴口罩。
数据集的组成如下所示：
![数据集构成](https://user-images.githubusercontent.com/87429023/165014173-7d38c6df-97cc-451a-bc1c-e5f559be8487.png)

