import sys
import os
import dlib
import glob
from skimage import io
import cv2
import numpy
import matplotlib.pyplot as plot

predictor_path = '/home/kjin/Qt/Face/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
#facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()

# Now process all the images
# for f in glob.glob(os.path.join("/home/kjin/caffe-master/examples/VGGNet/", "a.jpg")):
#     print("Processing file: {}".format(f))
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #img = io.imread("/home/kjin/caffe-master/examples/VGGNet/a.jpg")

    #win.clear_overlay()
 #   win.set_image(frame)
    cv2.namedWindow('original picture')
    cv2.imshow('original picture',frame)
    cv2.waitKey(50)
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(frame, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    for k, d in enumerate(dets):
     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
     if len(dets)>0:
         ROIimage=frame[d.top():d.bottom(), d.left():d.right()]
         cv2.imwrite('/home/kjin/caffe-master/examples/VGGNet/n.jpg', ROIimage)
         #ROIimage=cv2.resize(ROIimage, (224, 224))

    # ROIimage = frame[1:100, 1:404]
         cv2.namedWindow('latest picture')
         cv2.imshow('latest picture', ROIimage)
         #shape = sp(frame, d)
         cv2.waitKey(50)

        # Draw the face landmarks on the screen so we can see what face is currently being processed.
     # win.clear_overlay()
     # win.add_overlay(d)
     # win.add_overlay(shape)
