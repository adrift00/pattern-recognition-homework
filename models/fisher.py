import os
import cv2 as cv
import numpy as np
from .base import BaseClassifier


class Fisher1vs1(object):
    def __init__(self, class_nmu=2):
        super().__init__()
        self.W = None

    def train(self, features1, features2):
        self.mean1 = np.mean(features1, axis=0)
        self.mean2 = np.mean(features2, axis=0)
        shift_feat1 = features1-self.mean1
        shift_feat2 = features2-self.mean2
        S1 = (shift_feat1.T).dot(shift_feat1)
        S2 = (shift_feat2.T).dot(shift_feat2)
        Sw = S1+S2
        self.W = np.linalg.pinv(Sw).dot(self.mean1-self.mean2)

    def calc_dis(self, x):
        dist = (self.W.T).dot(x-0.5*(self.mean1+self.mean2))
        return dist


class Fisher(BaseClassifier):
    def __init__(self, class_num=10):
        super().__init__()
        self.class_num = class_num
        self.feature_shape = (10, 10)
        self.feature_length = self.feature_shape[0]*self.feature_shape[1]
        self.sample_num = np.zeros(class_num)
        self.total_num = 0
        self.features = [None]*class_num
        self.classifiers = [None]*(class_num-1)

    def train(self, train_dir):
        self.collect_features(train_dir)
        for i in range(self.class_num-1):
            classifers = []
            for j in range(i+1, self.class_num):
                fisher_1vs1 = Fisher1vs1()
                fisher_1vs1.train(self.features[i], self.features[j])
                classifers.append(fisher_1vs1)
            self.classifiers[i] = classifers

    def val(self, val_dir):
        error_num = np.zeros(self.class_num)
        validation_num = np.zeros(self.class_num)
        for i in range(self.class_num):
            sub_val_dir = os.path.join(val_dir, str(i))
            img_names = os.listdir(sub_val_dir)
            validation_num[i] = len(img_names)
            for img_name in img_names:
                img = cv.imread(os.path.join(sub_val_dir, img_name))
                pred_class = self.test(img)
                if pred_class != i:
                    error_num[i] += 1
        error_rate = error_num/validation_num
        total_error_rate = error_rate.mean()
        return error_rate, total_error_rate

    def test(self, img):
        X = self.img2feat(img)
        dist = np.zeros((self.class_num, self.class_num))
        for i in range(self.class_num-1):
            for j in range(i+1, self.class_num):
                fisher_1vs1 = self.classifiers[i][j-i-1]
                dis = fisher_1vs1.calc_dis(X)
                if dis > 0:
                    dist[i, j] = dis
                else:
                    dist[j, i] = -dis
        # vote for the pred class
        pred_class = np.argmax(np.sum(dist > 0, axis=1))
        return pred_class
