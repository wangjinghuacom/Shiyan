'''最终实验所用,不保存平均用时 0.14s, 保存平均用时 0.24s'''
import os
import cv2
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
from moviepy.editor import VideoFileClip
import line
import tensorflow as tf
from PIL import Image
import time


def thresholding(img):
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=70 ,thresh_max=200)#采用 x 方向的索贝尔算子
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 170))#计算梯度的大小
    #Thresholding combination
    thresholded = np.zeros_like(x_thresh)
    thresholded[(x_thresh == 1) & (mag_thresh == 1)] = 1

    return thresholded

def process(img):
    # pts = np.float32([(750, 520), (1000, 520), (1200, 1080), (450, 1080)])#pic2
    # pts = np.float32([(900, 500), (990, 500), (1250, 1080), (500, 1080)])#pic4
    pts = np.float32([(925, 410), (970, 410), (1250, 1080), (500, 1080)])#290,294
    # pts = np.float32([(860, 500), (970,500), (1250, 1080), (600, 1080)])# 1508,1528
    # pts = np.float32([(890, 330), (940,330), (1250, 1080), (640, 1080)])# 1508.1
    # pts = np.float32([(850, 310), (1000,310), (1350, 1080), (500, 1080)])# 20
    # pts = np.float32([(1160, 550), (1320, 550), (1410, 1080), (880, 1080)])#1008
    # pts = np.float32([(777, 530), (1020, 530), (1200, 1080), (500, 1080)])#374
    M,Minv,maxHeight,maxWidth = utils.get_M_Minv(pts)
    # image转换成array
    img = np.array(img)
    thresholded = thresholding(img)
    # plt.imshow(thresholded)
    # plt.show()
    # 透视变换
    thresholded_wraped = cv2.warpPerspective(thresholded, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    # plt.imshow(thresholded_wraped)
    # plt.show()
    # 执行检测
    left_line = line.Line()
    right_line = line.Line()
    if left_line.detected and right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)

    # area_img, gre1 = utils.draw_area(img,thresholded_wraped,Minv,left_fit, right_fit)
    Mask = utils.roi(img, thresholded_wraped, Minv, left_fit, right_fit)
    
    return Mask, thresholded_wraped

if __name__ == "__main__":
    ########################### 单张处理
    img = cv2.imread('/home/wjh/Rail-Lane-Lines/pic3/290.jpg')
    plt.imshow(img)
    plt.show()
    im, thresholded_wraped = process(img)
    cv2.imshow('1', im)
    cv2.waitKey(5000)

    # 去白边
    fig, ax = plt.subplots()
    # plt.axis('off')
    height, width, channels = img.shape
    fig.set_size_inches(width, height)#设置图像大小
    plt.gca().xaxis.set_major_locator(plt.NullLocator())#当前的图表和子图可以使用plt.gcf()和plt.gca()获得，分别表示Get Current Figure和Get Current Axes
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.grid(axis='y', which='major')
    ax.imshow(thresholded_wraped)
    plt.show()

    ########################### 多张处理
    # path = '/home/wjh/Rail-Lane-Lines/pic2/'
    # test_imgs = utils.get_images_by_dir(path)

    # result = []
    # n = 0
    # prev_time = time.time()
    # for img in test_imgs:
    #     # plt.imshow(img)
    #     # plt.show()
    #     a = time.time()   
    #     mask = process(img)

    #     # result.append(mask)
    #     # k = 20
    #     # for i in result:
    #     #     cv2.imwrite('/home/wjh/Rail-Lane-Lines/pic/'+str(k) + '.jpg',i)
    #     #     k+=1

    #     n += 1
    #     b = time.time()
    #     t = b - a
    #     print("n: {}\ntime: {:.2f}s".format(n, t))
    # curr_time = time.time()
    # exec_time = curr_time - prev_time
    # print("n: {}\ntotal time: {:.2f}s \ntime: {:.2f}s".format(n, exec_time, exec_time / n))