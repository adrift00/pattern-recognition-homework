import os
import cv2 as cv
import numpy as np
from .base_bayes import BaseBayes


class NaiveBayes(BaseBayes):
    def __init__(self, class_num=2):
        super().__init__()
        self.class_num = class_num
        self.feature_shape = (28, 28)
        self.feature_length = self.feature_shape[0]*self.feature_shape[1]
        self.sample_num = np.zeros(class_num)
        self.total_num = 0
        self.pw = np.zeros(class_num)
        self.features = [None]*class_num
        self.p = np.zeros((self.class_num, self.feature_length))

    def train(self, train_dir):
        # collect features and calc some useful information
        self.collect_features(train_dir)
        # calc the prior prob
        self.pw = self.sample_num/self.total_num
        # calc the prob matrix for eath class and feature indenti
        for i in range(self.class_num):
            self.p[i] = self.features[i].sum(axis=0)
            self.p[i] /= self.sample_num[i]

    def val(self, val_dir):
        error_num = np.zeros(self.class_num)
        validation_num = np.zeros(self.class_num)
        for i in range(self.class_num):
            sub_val_dir = os.path.join(val_dir, str(i))
            img_names = os.listdir(sub_val_dir)
            validation_num[i] = len(img_names)
            for img_name in img_names:
                img=cv.imread(os.path.join(sub_val_dir, img_name))
                pred_class = self.test(img)
                if pred_class != i:
                    error_num[i] += 1
        error_rate=error_num/validation_num
        total_error_rate = error_rate.mean()
        return error_rate, total_error_rate

    def test(self, img):
        X = self.img2feat(img)
        pxw = np.zeros(self.class_num)
        for i in range(self.class_num):
            pos_prob=self.p[i][X]
            neg_prob=(1-self.p[i])[~X]
            pxw[i]=np.prod(pos_prob)*np.prod(neg_prob)
        pred_prob = self.pw*pxw
        # print('pred prob is: {}'.format(pred_prob))
        pred_class = np.argmax(pred_prob)
        return pred_class


