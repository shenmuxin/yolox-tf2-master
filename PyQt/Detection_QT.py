'''
Description: 
coding: utf-8
language: python3.8, tensorflow2.6
Author: JiaHao Shum
Date: 2022-04-05 15:07:59
'''
from GUI import Ui_MainWindow
from PySide2.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide2.QtGui import QPixmap, QImage, QIcon
from PySide2.QtCore import Qt, QTimer

import os
import cv2
from PIL import Image
import numpy as np
import time

import sys
sys.path.append(r'D:\Study_Data\AI\YOLO\yolox-tf2-master')          #   注意修改为自己本地文件路径

from config import cfg
#   防止加载jpg图片出错
from PySide2 import QtGui, QtCore
QtCore.QCoreApplication.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), "plugins"))

from yolox_Detection import YOLOXDetect


class LoginGui(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(LoginGui, self).__init__()
        self.setupUi(self)
        self.initUI()
        self.yolo = YOLOXDetect()

    def initUI(self):

        self.initWindow()

        #   标志位初始化
        self.mode           = None
        self.detectflag     = False
        self.crop           = False
        #   初始化timer
        self.timer_camera   = QTimer() 

        #   slot bounding
        self.buttonGroup.buttonClicked.connect(self.modeChoose)
        self.btn_filedialog.setEnabled(False)
        self.btn_camera.setEnabled(False)

        self.btn_savepath.setEnabled(False)
        self.btn_savepath.clicked.connect(self.cropSavePath)
        self.btn_cropsave.clicked.connect(self.cropSave)

        self.btn_detect.clicked.connect(self.detectFlag)
        self.btn_stop.clicked.connect(self.videoSlotStop)


    def initWindow(self):
        """外观设置"""

        self.label_img.resize(640, 640)
        self.setWindowTitle('口罩佩戴检测')
        # set icon
        self.setWindowIcon(QIcon('PyQt\icon\yolo_icon.jpg'))

        # label_img 风格设置
        self.label_img.setStyleSheet('''background: rgba(177, 177, 177, 0.8);
                               font-family: YouYuan;
                               font-size: 12pt;
                               color: white;
                               ''')
        self.label_img.setAlignment(Qt.AlignCenter)  # 设置字体居中现实
        self.label_img.setText("Shum 木心")  # 默认显示文字

        self.label_text.setStyleSheet('''background: rgba(177, 177, 177, 0.8);
                               font-family: YouYuan;
                               font-size: 16pt;
                               color: white;
                               ''')
        self.label_text.setAlignment(Qt.AlignCenter)  
        self.label_text.setText("Show Information")  
        #   按键颜色设置
        self.btn_detect.setStyleSheet('QPushButton{background:pink;}')
        

    def modeChoose(self):
        """模式选择 image | video"""

        #   重置检测标志位
        self.detectflag = False

        self.btn_filedialog.setEnabled(True)

        text = self.buttonGroup.checkedButton().text()
        if text == 'Image':
            self.mode = 'Image'
            self.btn_filedialog.setText('选择图片文件')
            self.label_img.setText('Waiting for Image')

            #   断开slot连接
            try:
                self.btn_filedialog.clicked.disconnect(self.videoSlotStart)
            except:pass
            self.btn_filedialog.clicked.connect(self.imageFileDialog)
            print('Current mode is Image')

        elif text == 'Video':
            self.mode = 'Video'
            self.btn_filedialog.setText('选择视频文件')
            self.label_img.setText('Waiting for Video')

            try:
                self.btn_filedialog.clicked.disconnect(self.imageFileDialog)
            except:pass
            self.btn_filedialog.clicked.connect(self.videoSlotStart)

            self.btn_camera.setEnabled(True)
            self.btn_camera.clicked.connect(self.cameraSlotStart)
            print('Current mode is Video')
    

    def imageFileDialog(self):
        """图片路径读取,显示原始图片"""

        if self.mode == 'Image':
            flt = '图片文件(*.jpg *.png *.jpeg)'

        self.imagename, _ = QFileDialog.getOpenFileName(self, '文件选择', './', flt)
        if not self.imagename:
            QMessageBox.warning(self, '警告', '打开文件失败请重试', QMessageBox.Yes)
            return None

        #   显示图片
        if self.mode == 'Image':
            _, tail = os.path.split(self.imagename)
            if tail.endswith(('jpg', 'jpeg', 'png')):
                q_image = QImage(self.imagename).scaled(self.label_img.width(), self.label_img.height())
                self.label_img.setPixmap(QPixmap.fromImage(q_image))
            
    def cropSave(self):
        """crop选择"""

        if self.btn_cropsave.isChecked():
            self.crop = True
            print(self.crop)
            self.btn_savepath.setEnabled(True)
        elif not self.btn_cropsave.isChecked():
            self.crop = False
            self.btn_savepath.setEnabled(False)
    
    def cropSavePath(self):
        """crop save path 选择"""

        self.savepath = QFileDialog.getExistingDirectory(self, '选择保存的路径', './')  
        self.yolo.crop_save_path = self.savepath
        print(self.savepath)

    def detectFlag(self):
        
        if self.mode == 'Image':
            self.detectflag = True
            if self.detectflag == True: 
                #   进行检测
                image = Image.open(self.imagename)
                frame = np.array(self.yolo.detect_image(image, crop=self.crop))
                #   显示图片
                height, width, channel = frame.shape
                q_image = QImage(frame.data, width, height,
                                    QImage.Format_RGB888).scaled(self.label_img.width(), self.label_img.height())
                self.label_img.setPixmap(QPixmap.fromImage(q_image))

            #   处理警告
            self.handlewarning()

            #   恢复按键状态
            print('reset flag')
            self.btn_detect.setStyleSheet('QPushButton{background:pink;}')
            self.detectflag = False

        elif self.mode == 'Video':
            self.detectflag = not self.detectflag

            if self.detectflag == True:
                self.btn_detect.setStyleSheet("QPushButton{background:green;}")
            else:
                self.btn_detect.setStyleSheet('QPushButton{background:pink;}')


    def videoSlotStart(self):
        """启动video"""
        # print('Start Video!')
        if self.mode == 'Video':
            videoname, _ = QFileDialog.getOpenFileName(self, '选择检测的视频', './', '视频(*.mp4 *.avi)')
            if not videoname:
                # QMessageBox.information(self, '提示', '默认打开摄像头进行检测', QMessageBox.Yes)
                QMessageBox.warning(self, '警告', '选择文件出错请重试', QMessageBox.Yes)

            # videoname = videoname if videoname else 0
            self.cap = cv2.VideoCapture(videoname)
            #   获得fps
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.timer_camera.start(int(1000 / fps))
            self.timer_camera.timeout.connect(self.openFrame)
    
    def cameraSlotStart(self):

            self.cap = cv2.VideoCapture(0)
            #   获得fps
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.timer_camera.start(int(1000 / fps))
            self.timer_camera.timeout.connect(self.openFrame)

    def openFrame(self):
        """显示一帧图像"""

        if self.mode == 'Video':
            if self.cap.isOpened():
                ret, self.frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                     #   进行detect
                    if self.detectflag:
                        #   转换成Image
                        frame = Image.fromarray(np.uint8(frame))
                        #   进行检测
                        frame = np.array(self.yolo.detect_image(frame, crop=self.crop))

                    #   设置label_img显示图像
                    height, width, channel = frame.shape
                    q_image = QImage(frame.data, width, height,
                                        QImage.Format_RGB888).scaled(self.label_img.width(), self.label_img.height())
                    self.label_img.setPixmap(QPixmap.fromImage(q_image))

                else:
                    self.cap.release()
                    self.timer_camera.stop()

        self.handlewarning()


    def handlewarning(self):
        """
            处理未带口罩或则错误戴口罩的警告
        """ 

        print(self.yolo.warning)
        if self.yolo.warning == 0:
             self.label_text.setText('正确佩戴口罩')
             self.label_text.setStyleSheet('QLabel{background:green;}')
        elif self.yolo.warning == 1:
                self.label_text.setText('请佩戴口罩!')
                self.label_text.setStyleSheet('QLabel{background:red;}')
        elif self.yolo.warning == 2:
                self.label_text.setText('请正确佩戴口罩!')
                self.label_text.setStyleSheet('QLabel{background:pink;}')

    

    def videoSlotStop(self):
        """停止Video"""

        try:
            if self.cap:
                self.cap.release()
                self.timer_camera.stop()
                #   按键复位
                self.detectflag = False
                self.btn_detect.setStyleSheet('QPushButton{background:pink;}')
                print('Video Done!')
        except: pass
      
        



if __name__ == '__main__':
    app = QApplication([])
    gui = LoginGui()
    gui.show()
    app.exec_()
    # print(QtGui.QImageReader.supportedImageFormats())
