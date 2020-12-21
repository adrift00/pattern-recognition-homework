import os
import cv2 as cv
import numpy as np
from utils import min_pooling
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score

class BaseClassifier(object):
    def __init__(self, num_classes=2):
        super().__init__()
        self.class_num = num_classes

    def train(self,train_dir):
        raise NotImplementedError
    
    def val(self,val_dir):
        pred_scores=[]
        true_labels=[]
        for i in range(self.class_num):
            sub_val_dir = os.path.join(val_dir, str(i))
            img_names = os.listdir(sub_val_dir)
            for img_name in img_names:
                img = cv.imread(os.path.join(sub_val_dir, img_name))
                pred_class = self.test(img)
                pred_scores.append(pred_class)
                true_labels.append(i)
        acc=accuracy_score(true_labels,pred_scores)
        micro_f1_scores=f1_score(true_labels,pred_scores,average='micro')
        macro_f1_scores=f1_score(true_labels,pred_scores,average='macro')
        return acc,micro_f1_scores,macro_f1_scores

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