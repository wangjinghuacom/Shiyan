import os
import cv2
import utils2
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
from moviepy.editor import VideoFileClip
import line
import tensorflow as tf
from PIL import Image
import time


# 图片预处理
def thresholding(img):
    #setting all sorts of thresholds
    x_thresh = utils2.abs_sobel_thresh(img, orient='x', thresh_min=90 ,thresh_max=280)#采用 x 方向的索贝尔算子
    mag_thresh = utils2.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 170))#计算梯度的大小
    dir_thresh = utils2.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))#计算梯度方向
    hls_thresh = utils2.hls_select(img, thresh=(160, 255))
    lab_thresh = utils2.lab_select(img, thresh=(155, 210))
    luv_thresh = utils2.luv_select(img, thresh=(225, 255))
 
    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    # threshholded[(x_thresh == 1) & (mag_thresh == 1)]=1
    # plt.imshow(threshholded)
    # plt.show()
    return threshholded

# 处理
def process(img):
    # pts = np.float32([(904, 305), (1000, 305), (1450, 1080), (655, 1080)])
    pts = np.float32([(900, 500), (990, 500), (1250, 1080), (500, 1080)])
    M,Minv,maxHeight,maxWidth = utils2.get_M_Minv(pts)
    # array转换成image
    # img = Image.fromarray(img)
    # image转换成array
    img = np.array(img)
    thresholded = thresholding(img)
    # 透视变换
    # thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)  
    # 取遍图像的所有通道数，-1是反向取值,行列不变，通道数方向，由R、G、B更改为B、G、R
    thresholded_wraped = cv2.warpPerspective(thresholded, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    # perform detection 执行检测, 两条曲线拟合出来，左边一条，右边一条
    left_line = line.Line()
    right_line = line.Line()
    if left_line.detected and right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils2.find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils2.find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)

    # draw the detected laneline and the information　保存
    # array转换成image
    # undist = Image.fromarray(img)
    # area_img, gre1 = utils2.draw_area(img,thresholded_wraped,Minv,left_fit, right_fit)
    Mask = utils2.roi(img, thresholded_wraped, Minv, left_fit, right_fit)
    # print('5:',t10-t9)
    # li = utils2.draw_line(undist,thresholded_wraped,Minv,left_fit, right_fit)

    return Mask

test_imgs = utils2.get_images_by_dir('/home/wjh/Rail-Lane-Lines/pic2')

# undistorted = []
# for img in test_imgs:
#    undistorted.append(img)

result=[]
n = 0
beforet = time.time()
for img in test_imgs:
    prev_time = time.time()
    Mask = process(img)
    curr_time = time.time()
    exec_time = curr_time - prev_time
    # info = "time: %.2f ms" % (1000 * exec_time)
    
    # cv2.line(res, (610, 400), (1180, 1280), color=(0, 255, 0), thickness=4)
    # cv2.line(res, (700, 400), (30, 1280), color=(0, 255, 0), thickness=4)
    result.append(Mask)
    n += 1
    print("time: %.2f ms" % (1000 * exec_time))
print(n)
endt = time.time()
print('mtime:',(endt - beforet)/n)
# plt.figure(0)
# plt.imshow(result[0])
# plt.figure(1)
# plt.imshow(result[1])
# plt.figure(2)
# plt.imshow(t2[0])
# print(result[0].shape)
# plt.show()