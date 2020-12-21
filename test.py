import cv2 as cv
import torch
from models.bayes.naive_bayes import NaiveBayes
from models.bayes.gaussian_bayes import GaussianBayes
from models.fisher import Fisher
from models.alexnet import AlexnetTrainer
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    # 朴素贝叶斯
    # print('Training model now, please wait...')
    # cls = NaiveBayes(num_classes=10)
    # cls.train('./data/mnist/training')
    # print('train finished')

    # 正态分布贝叶斯
    print('Training model now, please wait ...')
    cls = GaussianBayes(num_classes=10)
    cls.train('./data/mnist/training')
    print('train finished')

    # Fisher线性判别
    # print('Training model now, please wait ...')
    # cls=Fisher(num_classes=10)
    # cls.train('./data/mnist/training')
    # print('train finished')

    # 深度学习模型
    # cls=AlexnetTrainer(num_classes=10)
    # cls.load_model()
    print('Testing model now, please wait ...')
    acc, micro_f1_score, macro_f1_score = cls.val('./data/mnist/testing')
    print(f'acc: {acc},micro f1: {micro_f1_score}, macro f1: {macro_f1_score}')
