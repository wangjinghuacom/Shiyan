# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:37:10 2017

@author: yang
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#get all image in the given directory persume that this directory only contain image files
def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname+'/'+img_name for img_name in img_names]
    imgs = [cv2.imread(path) for path in img_paths]
    return imgs

#function take the chess board image and return the object points and image points
def calibrate(images,grid=(9,6)):
    object_points=[]# 真实世界的3D点
    img_points = []# 图像的2D点
    for img in images:
        object_point = np.zeros( (grid[0]*grid[1],3),np.float32 )
        object_point[:,:2]= np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)#转置函数.T，将原shape为（n，m）的数组转置为（m，n）
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 找到棋盘边界，角点检测
        ret, corners = cv2.findChessboardCorners(gray, grid, None)
        # 如果找到，则添加对象点和图像点
        if ret:
            object_points.append(object_point)
            img_points.append(corners)
    return object_points,img_points

def order_points(pts):
    #参数 pts，是一个包含矩形四个点的(x, y)坐标的列表。
    #使用 np.zeros 为四个点分配内存。实际的排序可以是任意的，只要它在整个实现过程中是一致的。此处按照 “左上，右上，右下，左下” 进行排序。
    rect = np.zeros((4, 2), dtype = "float32")
    #根据 x 与 y 之和最小找到左上角的点，x 与 y 之和最大找到右下角的点
    s = pts.sum(axis = 1)#xis=1以后就是将一个矩阵的每一行向量相加
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    #使用 np.diff 函数，根据 x 与 y 之差（y-x）最小找到右上角的点，x 与 y 之差最大找到左下角的点
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]#numpy.argmin表示最小值在数组中所在的位置
    rect[3] = pts[np.argmax(diff)]
    #最后将排好序的点返回给调用函数
    return rect

def get_M_Minv(pts):
    #先调用 order_points 函数,按顺序获得点的坐标
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    #新图片的宽度就是右下角和左下角的点的x坐标之差
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    #同理新图片的高度就是右上角和右下角的点的y坐标之差。
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # dst = np.array([
    #     [0, 0],
    #     [maxWidth -1 , 0],
    #     [maxWidth -1 , maxHeight -1],
    #     [0, maxHeight -1]], dtype = "float32")
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype = "float32")
    #列表第一个元素 (0, 0) 表示左上角的点，第二个元素 (maxWidth - 1, 0) 表示右上角的点，
    #第三个元素 (maxWidth - 1, maxHeight - 1) 表示右下角的点，最后第四个元素 (0, maxHeight - 1) 表示左下角的点,顺序不变

    M = cv2.getPerspectiveTransform(rect, dst)
    Minv = cv2.getPerspectiveTransform(dst,rect)
    return M,Minv, maxHeight, maxWidth 

def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort() 摄像机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        #Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型，即cv2.CV_64F
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer 重新缩放为8位整数
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx)) #返回给定的 X 及 Y 坐标值的反正切值
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img,channel='s',thresh=(0, 255)): #使用hls颜色空间的通道进行阈值过滤：
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel=hls[:,:,1]
    else:
        channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:,:,2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output

def find_line(binary_warped):
    # Take a histogram of the bottom half of the image
    #统计每一列的非零点数，从而形成列直方图，车道因为是垂直于x轴的，所以每一列的非零点数比其他的要多
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)#/ 就表示 浮点数除法，返回浮点结果; // 表示整数除法
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    #左右车道分别在左半图和右半图查找直方图的最大值
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # 获取到所有非零点的坐标，分别按照列坐标，行坐标，存储
    nonzero = binary_warped.nonzero()
    # print(nonzero)
    nonzeroy = np.array(nonzero[0]) #列坐标
    nonzerox = np.array(nonzero[1]) #行坐标
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        # 对于区域内的点进行统计
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices 将所有的所有的点进行合并
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions, 将一维化的坐标重新映射回二维化的点坐标
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # 最小二乘法计算拟合多项式系数,x，y为拟合数据向量，要求维度相同，n为拟合多项式次数,返回left_fit向量保存多项式系数，由最高次向最低次排列。
    # 例如 n=1，即一次函数，left_fit返回系数 a，b
    left_fit = np.polyfit(lefty, leftx, 3)
    right_fit = np.polyfit(righty, rightx, 3)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds

def find_line_by_previous(binary_warped,left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**3) + left_fit[1]*(nonzeroy**2) +
    left_fit[2]*nonzeroy +left_fit[3] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**3) +
    left_fit[1]*(nonzeroy**2) + left_fit[2]*nonzeroy + left_fit[3] + margin)))
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**3) + right_fit[1]*(nonzeroy**2) +
    right_fit[2]*nonzeroy + right_fit[3] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**3) +
    right_fit[1]*(nonzeroy**2) + right_fit[2]*nonzeroy + right_fit[3] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial(多项式) to each
    left_fit = np.polyfit(lefty, leftx, 3)
    right_fit = np.polyfit(righty, rightx, 3)
    return left_fit, right_fit, left_lane_inds, right_lane_inds

def expand(img):
    image = img
    _,green,_ = cv2.split(image)
    # s = (np.sum(green,axis=1))/2 #当axis为1时,是压缩行,每一列相加,将矩阵压缩为一行
    s = np.sum(green,axis=1)
    a = range(img.shape[0])
    for i in reversed(a):
        if s[i] <255:
            break
        for j in range(1920):  #min x
            if green[i][j] == 255:
                break
        for k in reversed(range(1920)): #max x
            if green[i][k] == 255:
                break
        # for l in range(int(s[i]/255)): # s[i]/255  the number
        #     image[i,j-l,0] = 255   #对通道0（r）赋值
        # for l in range(int(s[i]/255)):
        #     image[i,k+l,0] = 255
        for l in range(int(s[i]/255)):     # 归一化
            if j>l:
                # l = int(l/2)
                image[i,j-l,1] = 255 #对通道2（b）赋值
        for l in range(int(s[i]/255)):
            if (k+l)<1920:
                # l = int(l/2)
                image[i,k+l,1] = 255

    return image

# 画轨道曲线
def draw_line(undist,binary_warped,Minv,left_fit, right_fit):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**3 + left_fit[1]*ploty**2 + left_fit[2]*ploty+left_fit[3]
    # print(left_fitx)
    # print(ploty)
    right_fitx = right_fit[0]*ploty**3 + right_fit[1]*ploty**2 + right_fit[2]*ploty+right_fit[3]
    left_x = np.array(left_fitx)
    left_y = np.array(ploty)
    right_x = np.array(right_fitx)
    right_y = np.array(ploty)
    #用3次多项式拟合
    f1 = np.polyfit(left_x, left_y, 3)
    f2 = np.polyfit(right_x, right_y, 3)
    l_y = np.polyval(f1, left_x)
    r_y = np.polyval(f2, right_x)
    # 若属性用的是全名则不能用*fmt*参数来组合赋值，应该用关键字参数对单个属性赋值如：
    plot1 = plt.plot(left_x, l_y, 'r', right_x, r_y, 'r', linewidth=5.0)
   
    plt.show()

def find_point(green,list_x,list_y,x,y):
    # neighbor = [[-2,-1],[-1,-1],[0,-1],[1,-1],[2,-1]] # 上一行从左向右
    # neighbor_len = len(neighbor)
    flag=0
    start_x = x
    start_y = y
        # print(start_x)
        # start_y = start_y + neighbor[i][1]
    if (green[start_y-1][start_x-2]) == 255:
        list_x.append(start_x-2)
        list_y.append(start_y-1)
        start_x = start_x-2
        start_y = start_y-1
        flag=1
        return start_x,start_y,flag
    if (green[start_y-1][start_x-1]) == 255:
        list_x.append(start_x)
        list_y.append(start_y)
        start_x = start_x-1
        start_y = start_y-1
        flag=1
        return start_x,start_y,flag
    if (green[start_y-1][start_x]) == 255:
        list_x.append(start_x)
        list_y.append(start_y)
        start_x = start_x
        start_y = start_y-1
        flag=1
        return start_x,start_y,flag
    if (green[start_y-1][start_x+1]) == 255:
        list_x.append(start_x)
        list_y.append(start_y)
        start_x = start_x+1
        start_y = start_y-1
        flag=1
        return start_x,start_y,flag
    if (green[start_y-1][start_x-2]) == 255:
        list_x.append(start_x)
        list_y.append(start_y)
        start_x = start_x+2
        start_y = start_y-1
        flag=1
        return start_x,start_y,flag
    return 0,0,flag
        
def find_point2(green,list_x,list_y,x,y):
    # r1,r2,r3=find_point(green,list_x,list_y,x,y)
    r1,r2,r3=find_point(green,list_x,list_y,x,y)
    while(r3!=0):
        r1,r2,r3=find_point(green,list_x,list_y,r1,r2)
    return list_x,list_y

def expand1(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    img_x = cv2.convertScaleAbs(x)   # 转回uint8  
    ####################################显示轨道线＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
    # plt.imshow(img_x)
    # plt.show()
    # cv2.imshow('1', img_x)
    # cv2.waitKey(5000)
    _,green,_ = cv2.split(img_x)
    # print(green.shape)

    j = img_x.shape[0]-10
    leftx_base = 0
    rightx_base = 0
    for i in range(500, 750):###########################################
        if green[j][i] == 255:
            leftx_base = i
            break       
    for k in range(950, 1100):############################################
        if green[j][k] == 255:
            rightx_base = k
            break
    # print(leftx_base,j,  rightx_base,j)
    list_x=[]
    list_y=[]
    list_m=[]
    list_n=[]

    a,b = find_point2(green,list_x,list_y,x=leftx_base,y=1070) #左侧xy
    c,d = find_point2(green,list_m,list_n,x=rightx_base,y=1070) #右侧xy
    # print(len(a))
    # print(len(b))
    # print(len(c))
    # print(len(d))
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(green).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    l = [c[i] - a[i] for i in range(len(a))]#图中轨道宽 x右-x左
    # a=[i-l for i in a]
    a = [a[i] - l[i]/2 for i in range(len(a))]#左侧轨道减去轨道宽
    # c=[i+l for i in c]
    c = [c[i] + l[i]/2 for i in range(len(c))]#右侧加

    pts_left = np.array([np.transpose(np.vstack([a, b]))])#垂直方向（行顺序）堆叠数组
    pts_right = np.array([np.flipud(np.transpose(np.vstack([c, d])))])#flipud上下翻转
                
    pts = np.hstack((pts_left, pts_right))#水平方向（列顺序）堆叠数组
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return color_warp

def draw_area(img,binary_warped,Minv,left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fitx = left_fit[0]*ploty**3 + left_fit[1]*ploty**2 + left_fit[2]*ploty+left_fit[3]
    right_fitx = right_fit[0]*ploty**3 + right_fit[1]*ploty**2 + right_fit[2]*ploty+right_fit[3]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # print(left_fitx)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # plt.imshow(color_warp)
    # plt.show()
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # undist = np.array(img)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    newwarp = expand1(newwarp)
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result,newwarp

def roi(img,binary_warped,Minv,left_fit, right_fit):

    m1,m2 = draw_area(img,binary_warped,Minv,left_fit, right_fit)

    Mask_inv = cv2.bitwise_not(m2)
    M_gray = cv2.cvtColor(Mask_inv, cv2.COLOR_BGR2GRAY)

    ret, Mask = cv2.threshold(M_gray, 200, 255, cv2.THRESH_BINARY)#像素高于阈值时，给像素赋予新值
    roi = cv2.bitwise_not(Mask)
    img_fg = cv2.bitwise_and(img, img, mask=roi)

    # print(roi.shape)
    return img_fg
