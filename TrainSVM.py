#coding:utf-8
import os
import sklearn
import imghdr
import numpy as np
import matplotlib.pyplot as plt
import skimage
import sys
import Image
from PIL import Image
import math
import dlib
import glob
import cv2
import numpy
import datetime
import time
#函数功能：生成标签，训练的图片名字

_command = "rm -rf caffe\n"
_command += "ln -s " + os.path.expanduser('~/') + "caffe-master/python/caffe ./caffe"
os.system(_command)

import caffe
import sklearn.metrics.pairwise as pw

from lfw_test_deal import *

List=os.listdir('/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/MyImage/')

fp=open('/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/trainSVM.txt','w')
for list in List:
    fp.write(list+'\n')
fp.close()
fp=open('/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/Label.txt','w')
for list in List:
    if list[0]=='T':
        fp.write('1'+'\n')
    else:
        fp.write('0'+'\n')


