#coding:utf-8
import cv2
import os
import sys
import numpy as np
#文件夹中有若干子文件夹，每个文件夹有若干图片，将所有子文件夹中的图片取出来
List=os.listdir('/home/kjin/caffe-master/examples/VGGNet/tmp/')
num=381
for PicList in List:
    SubList=os.listdir('/home/kjin/caffe-master/examples/VGGNet/tmp/'+PicList)
    for PicSubList in SubList:
        Pic=cv2.imread('/home/kjin/caffe-master/examples/VGGNet/tmp/'+PicList+'/'+PicSubList)
        cv2.imwrite('/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/MyImage/'+str(num)+'.jpg',Pic)
        num+=1

