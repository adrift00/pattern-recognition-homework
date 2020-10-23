import os
import cv2
import numpy as np
from numpy.lib.scimath import log
from .base_bayes import BaseBayes


class GaussianNaiveBayes(BaseBayes):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        self.feature_shape = (10, 10)
        self.feature_length = self.feature_shape[0]*self.feature_shape[1]
        self.sample_num = np.zeros(class_num)
        self.total_num = 0
        self.pw = np.zeros(class_num)
        self.features = [None]*class_num
        self.mean = np.zeros((class_num, self.feature_length))
        self.var = np.zeros((class_num, self.feature_length))

    def train(self, train_dir):
        self.collect_features(train_dir)
        for i in range(self.class_num):
            self.sample_num[i] = len(self.features[i])
            self.mean[i] = self.features[i].mean(axis=0)
            self.var[i] = self.features[i].var(axis=0)
        self.total_num = self.sample_num.sum()
        self.pw = self.sample_num/self.total_num

    def val(self, val_dir):
        error_num = np.zeros(self.class_num)
        validation_num = np.zeros(self.class_num)
        for i in range(self.class_num):
            sub_val_dir = os.path.join(val_dir, str(i))
            img_names = os.listdir(sub_val_dir)
            validation_num[i] = len(img_names)
            for img_name in img_names:
                img = cv2.imread(os.path.join(sub_val_dir, img_name))
                pred_class = self.test(img)
                if pred_class != i:
                    error_num[i] += 1
        error_rate = error_num/validation_num
        total_error_rate = error_rate.mean()
        return error_rate, total_error_rate

    def test(self, img):
        eps = 1e-9
        X = self.img2feat(img)
        pred_prob = -0.5*((X-self.mean)**2/(self.var+eps)).sum(axis=1)-0.5*(np.log(2*np.pi*(self.var+eps))).sum(axis=1)
        print(pred_prob)
        pred_class = np.argmax(pred_prob)
        return pred_class


# class GaussianNaiveBayes(BaseBayes):
#     def __init__(self, class_num):
#         super().__init__()
#         self.class_num = class_num
#         self.feature_shape = (28, 28)
#         self.feature_length = self.feature_shape[0]*self.feature_shape[1]
#         self.sample_num = np.zeros(class_num)
#         self.total_num = 0
#         self.pw = np.zeros(class_num)
#         self.features = [None]*class_num
#         self.feat_mean = np.zeros((class_num, self.feature_length))
#         self.cov_matrix = np.zeros((class_num, self.feature_length, self.feature_length))

#     def train(self, train_dir):
#         self.collect_features(train_dir)
#         for i in range(self.class_num):
#             self.sample_num[i] = len(self.features[i])
#             self.feat_mean[i] = self.features[i].mean(axis=0)
#             # calc the cov matrix
#             for j in range(len(self.features[i])):
#                 self.cov_matrix[i] += (self.features[i][j]-self.feat_mean[i])[:,
#                                                                               None].dot((self.features[i][j]-self.feat_mean[i])[None, :])
#             self.cov_matrix[i]/self.sample_num[i]
#         self.total_num = self.sample_num.sum()
#         self.pw = self.sample_num/self.total_num

#     def val(self, val_dir):
#         error_num = np.zeros(self.class_num)
#         validation_num = np.zeros(self.class_num)
#         for i in range(self.class_num):
#             sub_val_dir = os.path.join(val_dir, str(i))
#             img_names = os.listdir(sub_val_dir)
#             validation_num[i] = len(img_names)
#             for img_name in img_names:
#                 img = cv2.imread(os.path.join(sub_val_dir, img_name))
#                 pred_class = self.test(img)
#                 if pred_class != i:
#                     error_num[i] += 1
#         error_rate = error_num/validation_num
#         total_error_rate = error_rate.mean()
#         return error_rate, total_error_rate

#     def test(self, img):
#         eps=1e-9
#         X = self.img2feat(img)
#         pred_prob = np.zeros(self.class_num)
#         for i in range(self.class_num):
#             pred_prob[i] = -1/2*(X-self.feat_mean[i]).dot(np.linalg.pinv(self.cov_matrix[i])).dot(X-self.feat_mean[i])\
#                 - 1/2*log(np.linalg.det(self.cov_matrix[i])+eps)+1/2*log(self.pw[i])

#         pred_class = np.argmax(pred_prob)
#         return pred_class
