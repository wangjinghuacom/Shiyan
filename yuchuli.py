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
    # plt.imshow(x_thresh)
    # plt.show()
    mag_thresh = utils2.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 170))#计算梯度的大小
    dir_thresh = utils2.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))#计算梯度方向
    # hls_thresh = utils2.hls_select(img, thresh=(160, 255))
    # lab_thresh = utils2.lab_select(img, thresh=(155, 210))
    # luv_thresh = utils2.luv_select(img, thresh=(225, 255))
 
    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    # threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    threshholded[(x_thresh == 1) & (mag_thresh == 1)] = 1
    return threshholded

# 处理
def process(img):
    pts = np.float32([(904, 305), (1000, 305), (1450, 1080), (655, 1080)])
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
    # plt.imshow(thresholded_wraped)
    # plt.show()
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
    area_img, gre1 = utils2.draw_area(img,thresholded_wraped,Minv,left_fit, right_fit)
    Mask = utils2.roi(img, thresholded_wraped, Minv, left_fit, right_fit)
    return Mask

if __name__ == "__main__":
    # test_imgs = utils2.get_images_by_dir('/home/wjh/Rail-Lane-Lines/pic2')
    path = '/home/wjh/Rail-Lane-Lines/pic2/'
    test_imgs = os.listdir(path)

    result = []
    for img in test_imgs:
        image = cv2.imread(os.path.join(path, img))
        prev_time = time.time()
        mask  = process(image)
        curr_time = time.time()
        exec_time = curr_time - prev_time

        result.append(mask )
                                                        # plt.imshow(res)
        # plt.show()
        k = 20
        for i in result:
            cv2.imwrite('/home/wjh/Rail-Lane-Lines/pi/'+str(k) + '.jpg',i)
            k+=1
        print("time: %.2f s" % (exec_time))
        # print("{:.2f}".format(exec_time))
        