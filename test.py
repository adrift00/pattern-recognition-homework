import cv2 as cv
from models.naive_bayes import NaiveBayes
from models.gaussian_naive_bayes import GaussianNaiveBayes


if __name__ == '__main__':
    # bayes_cls = BinaryCodeBayes(class_num=10)
    bayes_cls = GaussianNaiveBayes(class_num=10)
    bayes_cls.train('../MNIST-JPG-master/mnist_data/training')
    error_rate = bayes_cls.val('../MNIST-JPG-master/mnist_data/testing')
    print('error rate is: {}'.format(error_rate))
