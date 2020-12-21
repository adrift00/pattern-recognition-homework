import os
import cv2 as cv
import numpy as np
from numpy.lib.function_base import average
from ..base import BaseClassifier

class NaiveBayes(BaseClassifier):
    def __init__(self, num_classes=2):
        super().__init__()
        self.class_num = num_classes
        self.feature_shape = (10,10)
        self.feature_length = self.feature_shape[0]*self.feature_shape[1]
        self.sample_num = np.zeros(num_classes)
        self.total_num = 0
        self.pw = np.zeros(num_classes)
        self.features = [None]*num_classes
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

    def test(self, img):
        X = self.img2feat(img)
        pxw = np.zeros(self.class_num)
        for i in range(self.class_num):
            pos_prob = self.p[i][X]
            neg_prob = (1-self.p[i])[~X]
            pxw[i] = np.prod(pos_prob)*np.prod(neg_prob)
        pred_prob = self.pw*pxw
        # print('pred prob is: {}'.format(pred_prob))
        pred_class = np.argmax(pred_prob)
        return pred_class
