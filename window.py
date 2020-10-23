import os
from os import write
import sys

import numpy as np
from utils import min_pooling
import cv2 as cv
from PyQt5.QtWidgets import QApplication, QFileDialog, QPushButton, QWidget
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt, QPoint

from models.naive_bayes import NaiveBayes
from models.gaussian_naive_bayes import GaussianNaiveBayes


class DrawWindow(QWidget):
    def __init__(self, parent=None):
        super(DrawWindow, self).__init__(parent)
        self.setWindowTitle("绘图例子")
        # self.pix = QPixmap()  # 实例化一个 QPixmap 对象
        self.lastPoint = QPoint()  # 起始点
        self.endPoint = QPoint()  # 终点
        self.initUi()
        self.bayes_cls=NaiveBayes(class_num=10)
        # self.bayes_cls=GaussianNaiveBayes(class_num=10)

    def initUi(self):
        # 窗口大小设置为600*500
        self.resize(600, 500)
        # 画布大小为400*400，背景为白色
        self.pix = QPixmap(400, 400)
        self.pix.fill(Qt.white)
        self.bin_pix = QPixmap(100,100)
        self.bin_pix.fill(Qt.white)
        self.train_button=QPushButton(self)
        self.test_button=QPushButton(self)
        self.clear_button=QPushButton(self)
        self.save_button=QPushButton(self)
        self.train_button.setGeometry(450,100,50,30)
        self.train_button.setText('训练')
        self.train_button.clicked.connect(self.train)
        self.test_button.setGeometry(450,200,50,30)
        self.test_button.setText('测试')
        self.test_button.clicked.connect(self.test)
        self.clear_button.setGeometry(450,300,50,30)
        self.clear_button.setText('清除')
        self.clear_button.clicked.connect(self.clear)
        self.save_button.setGeometry(450,400,50,30)
        self.save_button.setText('保存')
        self.save_button.clicked.connect(self.save)


    # 重绘的复写函数 主要在这里绘制
    def paintEvent(self, event):
        pp = QPainter(self.pix)

        pen = QPen() # 定义笔格式对象
        pen.setWidth(15)  # 设置笔的宽度
        pp.setPen(pen) #将笔格式赋值给 画笔

        # 根据鼠标指针前后两个位置绘制直线
        if self.lastPoint!=self.endPoint:
            pp.drawLine(self.lastPoint, self.endPoint)
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)  # 在画布上画出

        bin_painter=QPainter(self)
        bin_painter.drawPixmap(400,0,self.bin_pix)


   # 鼠标按压事件
    def mousePressEvent(self, event) :   
        # 鼠标左键按下  
        if event.button() == Qt.LeftButton :
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint

    # 鼠标移动事件
    def mouseMoveEvent(self, event):	
        # 鼠标左键按下的同时移动鼠标
        if event.buttons() and Qt.LeftButton :
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        # 鼠标左键释放   
        if event.button() == Qt.LeftButton :
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()
    def train(self):
        self.bayes_cls.train('../MNIST-JPG-master/mnist_data/training')
        print('train finished! ')
    
    def test(self,):
        qimg = self.pix.toImage()
        img=self.pix2img(qimg)
        # log new images
        # write_img_dir='./write_imgs'
        # if not os.path.exists(write_img_dir):
        #     os.makedirs(write_img_dir)
        # img_num=len(os.listdir(write_img_dir))
        pred_class=self.bayes_cls.test(img)
        print(pred_class)

        # show the bin img
        bin_img=self.bayes_cls.binary_img.astype(np.uint8)
        w=bin_img.shape[1]
        h=bin_img.shape[0]
        qimg=QImage(bin_img.data,w ,h, w, QImage.Format_Grayscale8)
        self.bin_pix=QPixmap().fromImage(qimg)
        self.update()

    def save(self,):
        qimg = self.pix.toImage()
        img=self.pix2img(qimg)
        collect_data_dir="./data/collect_data"
        for i in range(10):
            sub_data_dir=os.path.join(collect_data_dir,str(i))
            if not os.path.exists(sub_data_dir):
                os.makedirs(sub_data_dir)
        get_dir_path = QFileDialog.getExistingDirectory(self,
                                    "选取指定文件夹",
                                    collect_data_dir)

        rel_dir_path=os.path.relpath(get_dir_path) # avoid chinese char in path
        img_num=len(os.listdir(rel_dir_path))
        cv.imwrite(os.path.join(rel_dir_path,'{}.jpg'.format(img_num)),img)

    
    def clear(self,):
        self.pix.fill(Qt.white)
        self.update()

    def pix2img(self,qimg):
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        img = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        img = img[..., :3]
        return img


            
if __name__ == "__main__":  
        app = QApplication(sys.argv) 
        window = DrawWindow()
        window.show()
        sys.exit(app.exec_())
