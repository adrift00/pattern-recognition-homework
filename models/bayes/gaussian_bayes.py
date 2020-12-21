import os
import cv2
import numpy as np
from numpy.lib.scimath import log
from ..base import BaseClassifier


class GaussianBayes(BaseClassifier):
    def __init__(self, num_classes):
        super().__init__()
        self.class_num = num_classes
        self.feature_shape = (10, 10)
        self.feature_length = self.feature_shape[0]*self.feature_shape[1]
        self.sample_num = np.zeros(num_classes)
        self.total_num = 0
        self.pw = np.zeros(num_classes)
        self.features = [None]*num_classes
        self.mean = np.zeros((num_classes, self.feature_length))
        self.cov = np.zeros((num_classes, self.feature_length, self.feature_length))

    def train(self, train_dir):
        self.collect_features(train_dir)
        for i in range(self.class_num):
            self.sample_num[i] = len(self.features[i])
            self.mean[i] = self.features[i].mean(axis=0)
            # calc the cov matrix
            for j in range(len(self.features[i])):
                self.cov[i] += (self.features[i][j]-self.mean[i])[:,
                                                                  None].dot((self.features[i][j]-self.mean[i])[None, :])
            self.cov[i]/self.sample_num[i]
        self.total_num = self.sample_num.sum()
        self.pw = self.sample_num/self.total_num

    def test(self, img):
        eps = 1e-9
        X = self.img2feat(img)
        pred_prob = np.zeros(self.class_num)
        for i in range(self.class_num):
            pred_prob[i] = -0.5*(X-self.mean[i]).dot(np.linalg.pinv(self.cov[i])).dot(X-self.mean[i])
            pred_prob[i] += -0.5*log(np.linalg.det(self.cov[i])+eps)
            pred_prob[i] += 0.5*log(self.pw[i])

        pred_class = np.argmax(pred_prob)
        return pred_class
