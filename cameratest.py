#coding:utf-8
import cv2
import os
import sys
#测试相机能否使用
cap = cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    cv2.imshow('MyVideo',frame)
    cv2.waitKey(25)
