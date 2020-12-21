import numpy as np


def min_pooling(img, out_size):
    out_img = np.zeros(out_size)
    ori_h, ori_w = img.shape[:2]
    out_h, out_w = out_size
    y_stride = round(ori_h/out_h)
    x_stride = round(ori_w/out_w)
    thresh=0.05
    for y in range(out_h):
        for x in range(out_w):
            if y == out_h-1 and x != out_w-1:
                if img[y*y_stride:ori_h, x*x_stride:(x+1)*x_stride].sum()>(1-thresh)*(ori_h-y*y_stride)*x_stride:
                    out_img[y, x] = 1
            elif y != out_h-1 and x == out_w-1:
                if img[y*y_stride:(y+1)*y_stride, x*x_stride:ori_w].sum()>(1-thresh)*y_stride*(ori_w-x*x_stride):
                    out_img[y, x] = 1
            elif y == out_h-1 and x == out_w-1:
                if img[y*y_stride:ori_h, x*x_stride:ori_w].sum()>(1-thresh)*(ori_h-y*y_stride)*(ori_w-x*x_stride):
                    out_img[y, x] = 1
            else:
                if img[y*y_stride:(y+1)*y_stride, x*x_stride:(x+1)*x_stride].sum()>(1-thresh)*y_stride*x_stride:
                    out_img[y, x] = 1
    return out_img
