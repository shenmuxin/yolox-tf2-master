# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'GUI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(900, 1200)
        MainWindow.setBaseSize(QSize(1000, 1000))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_3)

        self.label_img = QLabel(self.centralwidget)
        self.label_img.setObjectName(u"label_img")
        self.label_img.setMinimumSize(QSize(640, 640))
        self.label_img.setMaximumSize(QSize(16777215, 16777215))
        self.label_img.setBaseSize(QSize(640, 640))

        self.horizontalLayout_4.addWidget(self.label_img)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 30)
        self.horizontalLayout_4.setStretch(2, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout = QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.btn_Image = QRadioButton(self.groupBox)
        self.buttonGroup = QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName(u"buttonGroup")
        self.buttonGroup.addButton(self.btn_Image)
        self.btn_Image.setObjectName(u"btn_Image")

        self.horizontalLayout.addWidget(self.btn_Image)

        self.btn_Video = QRadioButton(self.groupBox)
        self.buttonGroup.addButton(self.btn_Video)
        self.btn_Video.setObjectName(u"btn_Video")

        self.horizontalLayout.addWidget(self.btn_Video)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 2)

        self.verticalLayout.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btn_cropsave = QRadioButton(self.groupBox_2)
        self.btn_cropsave.setObjectName(u"btn_cropsave")

        self.horizontalLayout_2.addWidget(self.btn_cropsave)

        self.btn_savepath = QPushButton(self.groupBox_2)
        self.btn_savepath.setObjectName(u"btn_savepath")

        self.horizontalLayout_2.addWidget(self.btn_savepath)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.horizontalLayout_2.setStretch(0, 2)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 1)

        self.verticalLayout.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.horizontalLayout_3 = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.btn_filedialog = QPushButton(self.groupBox_3)
        self.btn_filedialog.setObjectName(u"btn_filedialog")

        self.verticalLayout_3.addWidget(self.btn_filedialog)

        self.btn_camera = QPushButton(self.groupBox_3)
        self.btn_camera.setObjectName(u"btn_camera")

        self.verticalLayout_3.addWidget(self.btn_camera)


        self.horizontalLayout_3.addLayout(self.verticalLayout_3)

        self.btn_detect = QPushButton(self.groupBox_3)
        self.btn_detect.setObjectName(u"btn_detect")

        self.horizontalLayout_3.addWidget(self.btn_detect)

        self.btn_stop = QPushButton(self.groupBox_3)
        self.btn_stop.setObjectName(u"btn_stop")

        self.horizontalLayout_3.addWidget(self.btn_stop)


        self.verticalLayout.addWidget(self.groupBox_3)

        self.label_text = QLabel(self.centralwidget)
        self.label_text.setObjectName(u"label_text")

        self.verticalLayout.addWidget(self.label_text)


        self.verticalLayout_2.addLayout(self.verticalLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_img.setText("")
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"\u6a21\u5f0f\u9009\u62e9", None))
        self.btn_Image.setText(QCoreApplication.translate("MainWindow", u"Image", None))
        self.btn_Video.setText(QCoreApplication.translate("MainWindow", u"Video", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"\u88c1\u526a\u9009\u62e9", None))
        self.btn_cropsave.setText(QCoreApplication.translate("MainWindow", u"crop(Yes/No)", None))
        self.btn_savepath.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u4fdd\u5b58\u8def\u5f84", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"\u63a7\u5236\u6309\u94ae", None))
        self.btn_filedialog.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u6587\u4ef6", None))
        self.btn_camera.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u542f\u6444\u50cf\u5934\u68c0\u6d4b", None))
        self.btn_detect.setText(QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b", None))
        self.btn_stop.setText(QCoreApplication.translate("MainWindow", u"\u505c\u6b62", None))
        self.label_text.setText("")
    # retranslateUi

