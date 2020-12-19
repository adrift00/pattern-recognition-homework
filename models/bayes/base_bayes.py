import os
import cv2 as cv
import numpy as np
from utils import min_pooling
from ..base import BaseClassifier

class BaseBayes(BaseClassifier):
    def __init__(self, class_num=2):
        super().__init__()
        self.class_num = class_num

    def train(self,train_dir):
        raise NotImplementedError
    
    def val(self,val_dir):
        raise NotImplementedError

    def test(self, X):
        raise NotImplementedError

    def collect_features(self, train_dir):
        dirs = os.listdir(train_dir)
        for i, dir in enumerate(dirs):
            sub_dir = os.path.join(train_dir, dir)
            img_names = os.listdir(sub_dir)
            self.sample_num[i] = len(img_names)
            features = np.zeros((len(img_names),self.feature_length))
            for j, img_name in enumerate(img_names):
                img_path = os.path.join(sub_dir, img_name)
                img = cv.imread(img_path)
                feature=self.img2feat(img)
                features[j] = feature
            self.features[i] = features
        self.total_num = sum(self.sample_num)
    
    def img2feat(self,img):
        img= cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        _, binary = cv.threshold(img,0,1,cv.THRESH_BINARY)
        img=min_pooling(binary,self.feature_shape)
        self.binary_img=img*255
        img=(img==0)
        feature = img.reshape(-1)
        return feature




