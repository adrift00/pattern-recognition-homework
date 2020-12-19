import cv2 as cv
import torch
from models.bayes.naive_bayes import NaiveBayes
from models.bayes.gaussian_naive_bayes import GaussianNaiveBayes
from models.fisher import Fisher
from models.alexnet import AlexnetTrainer


if __name__ == '__main__':
    # bayes_cls = BinaryCodeBayes(class_num=10)
    # bayes_cls = GaussianNaiveBayes(class_num=10)
    # cls=Fisher(class_num=10)
    cls=AlexnetTrainer(num_classes=10)
    cls.train('../MNIST-JPG-master/mnist_data/training')
    error_rate = cls.val('../MNIST-JPG-master/mnist_data/testing')
    # cls.train('./data/100test/training')
    # error_rate = cls.val('./data/100test/testing')
    # cls.train('./data/output_100_resize/training')
    # error_rate = cls.val('./data/output_100_resize/testing')
    print('error rate is: {}'.format(error_rate))

