'''逐步检测
显示有效监测区域预处理过程中每一个阶段的图片 
'''
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


img = cv2.imread('/home/wjh/Rail-Lane-Lines/pic3/20.jpg')
plt.imshow(img)
plt.show()
def thresholding(img):
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=90 ,thresh_max=200)#采用 x 方向的索贝尔算子
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 170))#计算梯度的大小
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))#计算梯度方向
    hls_thresh = utils.hls_select(img, thresh=(160, 255))
    lab_thresh = utils.lab_select(img, thresh=(155, 210))
    luv_thresh = utils.luv_select(img, thresh=(225, 255))
    #Thresholding combination
    thresholded = np.zeros_like(x_thresh)
    thresholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    
    return thresholded

# pts = np.float32([(900, 500), (990, 500), (1250, 1080), (500, 1080)])#290,294
# pts = np.float32([(860, 500), (970,500), (1250, 1080), (600, 1080)])# 1508,1528
pts = np.float32([(850, 310), (1000,310), (1350, 1080), (500, 1080)])# 20
# pts = np.float32([(1160, 550), (1320, 550), (1410, 1080), (880, 1080)])#1008
# pts = np.float32([(777, 530), (1020, 530), (1200, 1080), (500, 1080)])#374
M,Minv,maxHeight,maxWidth = utils.get_M_Minv(pts)
# array转换成image
# img = Image.fromarray(img)
# image转换成array
img = np.array(img)
thresholded = thresholding(img)
plt.imshow(thresholded)
plt.show()
# 透视变换
thresholded_wraped = cv2.warpPerspective(thresholded, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
plt.imshow(thresholded_wraped)
plt.show()
# perform detection 执行检测, 两条曲线拟合出来，左边一条，右边一条
left_line = line.Line()
right_line = line.Line()

if left_line.detected and right_line.detected:
    left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
else:
    left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholded_wraped)
left_line.update(left_fit)
right_line.update(right_fit)

area_img, gre1 = utils.draw_area(img,thresholded_wraped,Minv,left_fit, right_fit)
Mask = utils.roi(img, thresholded_wraped, Minv, left_fit, right_fit)

cv2.imshow('1', gre1)
cv2.waitKey(5000)
cv2.imshow('1', area_img)
cv2.waitKey(5000)
cv2.imshow('1', Mask)
cv2.waitKey(5000)

# 去除图像周围的白边
fig, ax = plt.subplots()
# plt.axis('off')
height, width, channels = img.shape
fig.set_size_inches(width/100, height/100)#设置图像大小
plt.gca().xaxis.set_major_locator(plt.NullLocator())#当前的图表和子图可以使用plt.gcf()和plt.gca()获得，分别表示Get Current Figure和Get Current Axes
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.margins(0,0)
plt.grid(axis='y', which='major')
ax.imshow(thresholded_wraped)
plt.show()


# # ax.set_xlabel("n_components")
# # ax.set_ylabel("ARI")
# # fig.suptitle("GMM")
# plt.grid(axis='y')
# plt.imshow(area_img)
# plt.show()
# plt.imshow(gre1)
# plt.show()
# plt.imshow(Mask)
# plt.show()



