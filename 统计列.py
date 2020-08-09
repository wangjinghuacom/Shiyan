'''逐步检测'''
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


img = cv2.imread('/home/wjh/Rail-Lane-Lines/pic3/290.jpg')

def thresholding(img):
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=40 ,thresh_max=200)#采用 x 方向的索贝尔算子
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 170))#计算梯度的大小
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))#计算梯度方向
    hls_thresh = utils.hls_select(img, thresh=(160, 255))
    lab_thresh = utils.lab_select(img, thresh=(155, 210))
    luv_thresh = utils.luv_select(img, thresh=(225, 255))
    #Thresholding combination
    thresholded = np.zeros_like(x_thresh)
    thresholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    
    return thresholded

pts = np.float32([(900, 500), (990, 500), (1250, 1080), (500, 1080)])#290,294
# pts = np.float32([(860, 500), (970,500), (1250, 1080), (600, 1080)])# 1508,1528
M,Minv,maxHeight,maxWidth = utils.get_M_Minv(pts)
# array转换成image
# img = Image.fromarray(img)
# image转换成array
img = np.array(img)
thresholded = thresholding(img)

# 透视变换
thresholded_wraped = cv2.warpPerspective(thresholded, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
plt.imshow(thresholded_wraped)
plt.show()

img = np.array(thresholded_wraped)
print(img.shape[0], thresholded_wraped.shape[1])
for j in range(750):
    with open("/home/wjh/Rail-Lane-Lines/t.txt",'a') as f:
        f.write(str(j)+':')
        f.close()
    for i in range(703, 350, -1):
        if thresholded_wraped[i][j] == 1:
            with open("/home/wjh/Rail-Lane-Lines/t.txt",'a') as f:
                f.write('1'+',')
    with open("/home/wjh/Rail-Lane-Lines/t.txt",'a') as f:
        f.write("\n")

