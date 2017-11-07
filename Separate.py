#coding:utf-8
import cv2
import os
import sys
import numpy as np
#将其他文件夹的图片搬移到另外一个文件夹并重命名
ListDir='/home/kjin/caffe-master/examples/DeepID/CASIA-FaceV5-Train/'
List=os.listdir(ListDir)
num=2126
for list in List:
    img=cv2.imread(ListDir+list)
    cv2.imwrite('/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/MyImage/Test'+str(num)+'.jpg',img)
    num+=1
