#coding:utf-8
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
import caffe
import sklearn.metrics.pairwise as pw
# from DeepID import *
from lfw_test_deal import *

#从相机照相
cap=cv2.VideoCapture(0)
frame2 = '/home/kjin/MatConNet/vgg_face_matconvnet/cam.jpg'
while True:
    ret,frame=cap.read()

    cv2.imwrite('/home/kjin/MatConNet/vgg_face_matconvnet/cam.jpg', frame)
    cv2.namedWindow('latest picture')
    #pic=cv2.imread(frame2)
    cv2.imshow('latest picture',frame)
    cv2.waitKey(50)