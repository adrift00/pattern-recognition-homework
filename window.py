import os
from os import write
import sys

import numpy as np
from torch._C import set_anomaly_enabled
from utils import min_pooling
import cv2 as cv
from PyQt5.QtWidgets import QApplication, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QStatusBar, QTextEdit, QVBoxLayout, QWidget
from PyQt5.QtGui import QFont, QImage, QPainter, QPen, QPixmap, QTextBlock
from PyQt5.QtCore import Qt, QPoint

from models.bayes.naive_bayes import NaiveBayes
from models.bayes.gaussian_bayes import GaussianBayes
from models.fisher import Fisher
from models.alexnet import AlexnetTrainer
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# the dataset should be 'mnist' or 'mnist_100'
DATASET='mnist' # mnist_100


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.models = {'朴素贝叶斯': NaiveBayes, '高斯贝叶斯': GaussianBayes, 'Fisher线性判别': Fisher, '神经网络模型': AlexnetTrainer}
        self.model_names = list(self.models.keys())
        self.models2names = {v: k for k, v in self.models.items()}
        self.initUi()

        text = self.select_model_box.currentText()
        self.classfier = self.models[text](num_classes=10)
        
        if isinstance(self.classfier, AlexnetTrainer):
            self.classfier.load_model()
            self.statusbar.showMessage(f'当前模型是 “{self.models2names[type(self.classfier)]}”模型，已经加载预训练模型，可直接测试。')
        else:
            self.statusbar.showMessage(f'当前模型是 “{self.models2names[type(self.classfier)]}”模型，请先按“训练”键进行训练，再进行测试。')


    def initUi(self):
        self.setWindowTitle("手写数字识别DEMO")
        self.lastPoint = QPoint()  # 起始点
        self.endPoint = QPoint()  # 终点
        # 窗口大小设置为600*500
        self.resize(600, 410)
        # 画布大小为400*400，背景为白色
        self.pix = QPixmap(400, 400)
        self.pix.fill(Qt.white)
        self.pix_lab = QLabel()
        self.pix_lab.setPixmap(self.pix)

        self.bin_pix = QPixmap(100, 100)
        self.bin_pix.fill(Qt.white)
        self.bin_pix_lab = QLabel()
        self.bin_pix_lab.setPixmap(self.bin_pix)

        self.train_button = QPushButton(self)
        self.test_button = QPushButton(self)
        self.clear_button = QPushButton(self)
        self.save_button = QPushButton(self)
        self.train_button.setText('训练')
        self.train_button.clicked.connect(self.train)
        self.test_button.setText('测试')
        self.test_button.clicked.connect(self.test)
        self.clear_button.setText('清除')
        self.clear_button.clicked.connect(self.clear)
        self.save_button.setText('保存')
        self.save_button.clicked.connect(self.save)

        self.select_model_box = QComboBox()
        for model_name in self.model_names:
            self.select_model_box.addItem(model_name)
        self.select_model_box.currentIndexChanged.connect(self.selectionchange)

        self.pred_box = QLabel()
        ft = QFont()
        ft.setPointSize(24)
        self.pred_box.setFont(ft)

        left_widget = self.create_left_widget()
        right_widget = self.create_right_widget()

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 4)
        main_widget = QWidget()
        main_widget.setLayout(main_layout)

        self.statusbar = QStatusBar(self)

        self.setStatusBar(self.statusbar)
        self.statusbar.setSizeGripEnabled(False)
        self.setCentralWidget(main_widget)

    def create_left_widget(self):
        left_widget = QGroupBox('请在此处画图:')
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.pix_lab)
        left_layout.addStretch(5)
        left_widget.setLayout(left_layout)
        return left_widget

    def create_right_widget(self):
        upper_right_widget = QGroupBox("二值化后图像:")
        upper_right_layout = QVBoxLayout()
        upper_right_layout.addWidget(self.bin_pix_lab)
        upper_right_widget.setLayout(upper_right_layout)

        mid_right_widget = QGroupBox('模型识别的数字：')
        mid_right_layout = QVBoxLayout()
        mid_right_layout.addWidget(self.pred_box)
        mid_right_widget.setLayout(mid_right_layout)

        lower_right_widget = QGroupBox("选择操作")
        lower_right_layout = QVBoxLayout()
        lower_right_layout.addWidget(QLabel("请选择模型："))
        lower_right_layout.addWidget(self.select_model_box)
        lower_right_layout.addWidget(self.train_button)
        lower_right_layout.addWidget(self.test_button)
        lower_right_layout.addWidget(self.clear_button)
        lower_right_layout.addWidget(self.save_button)
        lower_right_layout.addStretch(5)
        lower_right_widget.setLayout(lower_right_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(upper_right_widget)
        right_layout.addWidget(mid_right_widget)
        right_layout.addWidget(lower_right_widget)
        right_layout.setStretch(0, 1)
        right_layout.setStretch(1, 2)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        return right_widget

    # 重绘的复写函数 主要在这里绘制

    def paintEvent(self, event):
        pp = QPainter(self.pix)
        # a little ugly here.
        pp.setWindow(18, 33, 400, 400)

        pen = QPen()  # 定义笔格式对象
        pen.setWidth(15)  # 设置笔的宽度
        pp.setPen(pen)  # 将笔格式赋值给 画笔

        # 根据鼠标指针前后两个位置绘制直线
        if self.lastPoint != self.endPoint:
            pp.drawLine(self.lastPoint, self.endPoint)
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint

        self.pix_lab.setPixmap(self.pix)

   # 鼠标按压事件
    def mousePressEvent(self, event):
        # 鼠标左键按下
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        # 鼠标左键按下的同时移动鼠标
        if event.buttons() and Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

    def selectionchange(self, i):
        text = self.select_model_box.currentText()
        self.classfier = self.models[text](num_classes=10)
        if isinstance(self.classfier, AlexnetTrainer):
            self.classfier.load_model()
            self.statusbar.showMessage(f'当前模型是 “{self.models2names[type(self.classfier)]}”模型，已经加载预训练模型，可直接测试。')
        else:
            self.statusbar.showMessage(f'当前模型是 “{self.models2names[type(self.classfier)]}”模型，请先按“训练”键进行训练，再进行测试。')

    def train(self):
        msg=''
        if DATASET=='mnist':
            msg='开始训练模型，使用完整的mnist数据集，请稍等几分钟……'
        elif DATASET=='mnist_100':
            msg='开始训练模型，每类使用mnist的张图片，请稍等……'
        else:
            msg='开始训练模型，请稍等……'
        self.statusbar.showMessage(msg)
        # need to repait.
        self.repaint()
        self.classfier.train(f'./data/{DATASET}/training')
        self.statusbar.showMessage('训练结束，可以进行测试。')

    def test(self,):
        qimg = self.pix.toImage()
        img = self.pix2img(qimg)
        pred_class = self.classfier.test(img)
        # print(pred_class)
        self.pred_box.setText(str(pred_class))

        # show the bin img
        if not isinstance(self.classfier, AlexnetTrainer):
            bin_img = self.classfier.binary_img.astype(np.uint8)
            w = bin_img.shape[1]
            h = bin_img.shape[0]
            qimg = QImage(bin_img.data, w, h, w, QImage.Format_Grayscale8)
            qimg=qimg.scaled(100,100)
            self.bin_pix = QPixmap().fromImage(qimg)
            self.bin_pix_lab.setPixmap(self.bin_pix)
            self.update()

    def save(self,):
        qimg = self.pix.toImage()
        img = self.pix2img(qimg)
        collect_data_dir = "./data/collect_data"
        for i in range(10):
            sub_data_dir = os.path.join(collect_data_dir, str(i))
            if not os.path.exists(sub_data_dir):
                os.makedirs(sub_data_dir)
        get_dir_path = QFileDialog.getExistingDirectory(self,
                                                        "选取指定文件夹",
                                                        collect_data_dir)

        rel_dir_path = os.path.relpath(get_dir_path)  # avoid chinese char in path
        img_num = len(os.listdir(rel_dir_path))
        cv.imwrite(os.path.join(rel_dir_path, '{}.jpg'.format(img_num)), img)

    def clear(self,):
        self.pix.fill(Qt.white)
        self.bin_pix.fill(Qt.white)
        self.bin_pix_lab.setPixmap(self.bin_pix)
        self.update()

    def pix2img(self, qimg):
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        img = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        img = img[..., :3]
        return img


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
