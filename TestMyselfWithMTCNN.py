#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Create on Tues 2015-11-24

@author: hqli
'''

# 运行前,先将pycaffe安装好
# 运行时在caffe主目录（一般为～/caffe-master）下执行python DeepIDTest.py

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
from sklearn import svm
from sklearn.externals import joblib
_command = "rm -rf caffe\n"
_command += "ln -s " + os.path.expanduser('~/') + "caffe-master/python/caffe ./caffe"
os.system(_command)

import caffe
import sklearn.metrics.pairwise as pw

# from DeepID import *
from lfw_test_deal import *
sys.path.append('/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/')
from main_Original import *


class DeepIDTest():
    pairs = ''  # pairs.txt的路径

    itera = 10000  # 训练好的模型的迭代次数

    model = ''  # 模型
    imgmean_npy = ''  # 图像均值的npy
    deploy = ''

    savepath = ''  # 准确率与ROC曲线图像存储路径,一般放在deepID/num/test/下

    left = ''  # pairs.txt分离的左图像路径的文本,放在工程文件夹下
    right = ''  # pairs.txt分离的右图像路径的文本
    label = ''  # pairs.txt分离的标签的文本

    accuracy = ''  # 准确率的文件
    predict = ''  # 预测值的文件

    lfwpath = ''  # 已经剪切好的lfw图像
    roc = ''  # roc图像

    def __init__(self, prj, caffepath, prjpath, datapath, num, types, pairs, itera, lfwpath):
        # DeepID.__init__(self,prj,caffepath,prjpath,datapath,num)

        self.itera = itera
        self.pairs = pairs
       # self.lfwpath = lfwpath

        self.model = '/home/kjin/caffe-master/examples/VGGNet/VGG_test_20170512/vgg_10575_iter_150000.caffemodel'
        self.deploy = prjpath + 'VGG_FACE' + '_deploy.prototxt'

        self.imgmean_npy = prjpath + prj + '_mean.npy'








      #  self.savepath = prjpath + 'test/'
      #  if not os.path.exists(self.savepath):
      #      os.makedirs(self.savepath)
       # self.accuracy = self.savepath + prj + '_' + str(num) + '_' + str(itera) + '_accuracy.txt'
        #self.predict = self.savepath + prj + '_' + str(num) + '_' + str(itera) + '_predict.txt'
     #   self.roc = self.savepath + prj + '_' + str(num) + '_' + str(itera) + '_roc'

    # def split_pairs(self):
    #     ext = 'ppm'
    #     print self.lfwpath
    #     pairs_result = testdeal(self.pairs, self.lfwpath, ext)
    #
    #     fp_left = open(self.left, 'w')
    #     fp_right = open(self.right, 'w')
    #     fp_label = open(self.label, 'w')
    #
    #     fp_left.write(pairs_result['path_left'])
    #     fp_right.write(pairs_result['path_right'])
    #     fp_label.write(pairs_result['label'])
    #
    #     fp_left.close()
    #     fp_right.close()
    #     fp_label.close()

    @staticmethod
    def fileopt(filename, content):
        fp = open(filename, 'w')
        fp.write(content)
        fp.close()

    @staticmethod
    def read_imagelist(X1):
        '''
        @brief：从列表文件中，读取图像数据到矩阵文件中
        @param： filelist 图像列表文件
        @return ：4D 的矩阵
        '''

        X = np.empty((1,3, 224, 224))#create array
        imgType=imghdr.what(X1)
        if(imgType=='jpeg'):
          im1 = skimage.io.imread(X1, as_grey=False)
          image = skimage.transform.resize(im1, (224, 224))*255
          if image.ndim < 3:
             print 'gray:' + X1
             X[0,0, :, :] = image[:, :]
             X[0,1, :, :] = image[:, :]
             X[0,2, :, :] = image[:, :]
          else:
             X[0,0, :, :] = image[:, :, 0]
             X[0,1, :, :] = image[:, :, 1]
             X[0,2, :, :] = image[:, :, 2]
          return X

    @staticmethod
    def read_labels(label):
        '''
        读取标签列表文件
        '''
        fin = open(label)
        lines = fin.readlines()
        labels = np.empty((len(lines),))
        k = 0
        for line in lines:
            labels[k] = int(line)
            k = k + 1
        fin.close()
        return labels

    @staticmethod
    def calculate_accuracy(distance, labels, num):
        '''
        #计算识别率,
        选取阈值，计算识别率
        '''
        accuracy = []
        predict = np.empty((num,))
        threshold = 0.25
        while threshold <= 0.8:
            for i in range(num):
                if distance[i] >= threshold:
                    predict[i] = 1
                else:
                    predict[i] = 0
            predict_right = 0.0
            for i in range(num):
                if predict[i] == labels[i]:
                    predict_right = 1.0 + predict_right
            current_accuracy = (predict_right / num)
            accuracy.append(current_accuracy)
            threshold = threshold + 0.001
        return np.max(accuracy)

    @staticmethod
    def draw_roc_curve(fpr, tpr, title='cosine', save_name='roc_lfw'):
        '''
        画ROC曲线图
        '''
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic using: ' + title)
        plt.legend(loc="lower right")
        #    plt.show()
        plt.savefig(save_name + '.png')




    #
    # def Distance(p1, p2):
    #     dx = p2[0] - p1[0]
    #     dy = p2[1] - p1[1]
    #     return math.sqrt(dx * dx + dy * dy)
    #
    #     # 根据参数，求仿射变换矩阵和变换后的图像。
    #
    # def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    #     if (scale is None) and (center is None):
    #         return image.rotate(angle=angle, resample=resample)
    #     nx, ny = x, y = center
    #     sx = sy = 1.0
    #     if new_center:
    #         (nx, ny) = new_center
    #     if scale:
    #         (sx, sy) = (scale, scale)
    #     cosine = math.cos(angle)
    #     sine = math.sin(angle)
    #     a = cosine / sx
    #     b = sine / sx
    #     c = x - nx * a - ny * b
    #     d = -sine / sy
    #     e = cosine / sy
    #     f = y - nx * d - ny * e
    #     return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)
    #     # 根据所给的人脸图像，眼睛坐标位置，偏移比例，输出的大小，来进行裁剪。
    #
    # def CropFace(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    #     # calculate offsets in original image 计算在原始图像上的偏移。
    #     offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    #     offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    #     # get the direction  计算眼睛的方向。
    #     eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    #     # calc rotation angle in radians  计算旋转的方向弧度。
    #     rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    #     # distance between them  # 计算两眼之间的距离。
    #     dist = Distance(eye_left, eye_right)
    #     # calculate the reference eye-width    计算最后输出的图像两只眼睛之间的距离。
    #     reference = dest_sz[0] - 2.0 * offset_h
    #     # scale factor   # 计算尺度因子。
    #     scale = float(dist) / float(reference)
    #     # rotate original around the left eye  # 原图像绕着左眼的坐标旋转。
    #     image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    #     # crop the rotated image  # 剪切
    #     crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)  # 起点
    #     crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)  # 大小
    #     image = image.crop(
    #         (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    #     # resize it 重置大小
    #     image = image.resize(dest_sz, Image.ANTIALIAS)
    #     return image


        #   for list in Lists:
        # X1 = '/home/kjin/caffe-master/examples/VGGNet/' + list
        # X = DeepIDTest.read_imagelist(X1)
        # out = net.forward_all(data=X)
        # print out
        # feature2 = np.float64(out['fc7'])
        # feature2 = np.reshape(feature2, (1, 4096))





    def evaluate(self, metric='cosine'):
        '''
        @brief: 评测模型的性能
        @param：itera： 模型的迭代次数
        @param：metric： 度量的方法
        '''
        cap = cv2.VideoCapture(0)
        caffe.set_mode_gpu()

        net = caffe.Classifier(self.deploy, self.model, mean=np.load(self.imgmean_npy))

        Lists = os.listdir('/home/kjin/caffe-master/examples/VGGNet/ContrastivePicNew/')
        Feature = [None] * len(Lists)
        i = 0
        for list in Lists:
            X1 = '/home/kjin/caffe-master/examples/VGGNet/ContrastivePicNew/' + list
            X = DeepIDTest.read_imagelist(X1)
            out = net.forward_all(data=X)
            Feature[i] = np.float64(out['fc7'])
            Feature[i] = np.reshape(Feature[i], 4096)
            i = i + 1

        prototxt = ['/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/model/' + x + '.prototxt' for x in
                    ['det1', 'det2', 'det3']]
        binary = ['/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/model/' + x + '.caffemodel' for x in
                  ['det1', 'det2', 'det3']]
        PNet = caffe.Net(prototxt[0], binary[0], caffe.TEST)
        RNet = caffe.Net(prototxt[1], binary[1], caffe.TEST)
        ONet = caffe.Net(prototxt[2], binary[2], caffe.TEST)

        Error = 70
        while True:
            while True:
                t1 = time.time()
                ret, im = cap.read()

                # Load image.
                # im = cv2.imread(frame)s
                # assert im is not None, 'Image is empty.'
                im_bk = im.copy()

                im = im.astype(np.float32)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = np.transpose(im, (1, 0, 2))  # Rotate image.

                image_height, image_width, num_channels = im.shape
                print('Image shape:', im.shape)

                # assert num_channels == 3, 'Error: only support RGB image.'

                MIN_FACE_SIZE = 24.  # Minimum face size.
                MIN_INPUT_SIZE = 12.  # Minimum input size.
                m = MIN_INPUT_SIZE / MIN_FACE_SIZE

                min_size = min(image_height, image_width)
                min_size = min_size * m

                scales = []
                counter = 0
                FACTOR = 0.709
                while min_size >= MIN_INPUT_SIZE:
                    scales.append(m * FACTOR ** counter)
                    min_size = min_size * FACTOR
                    counter = counter + 1

                # Load models.


                # Threshold for each stage.
                # THRESHOLD = [0.6, 0.7, 0.7]
                THRESHOLD = [0.9, 0.9, 0.9]
                t1 = time.time()

                # --------------------------------------------------------------
                # First stage.
                #
                total_boxes = []  # Bounding boxes of all scales.
                for scale in scales:
                    hs = int(math.ceil(image_height * scale))
                    ws = int(math.ceil(image_width * scale))

                    im_resized = cv2.resize(im, (ws, hs), interpolation=cv2.INTER_AREA)
                    print('Resize to:', im_resized.shape)

                    # H,W,C -> C,H,W
                    im_resized = np.transpose(im_resized, (2, 0, 1))

                    # Zero mean and normalization.
                    im_resized = (im_resized - 127.5) * 0.0078125

                    # Reshape input layer.
                    PNet.blobs['data'].reshape(1, 3, hs, ws)
                    PNet.blobs['data'].data[...] = im_resized
                    outputs = PNet.forward()

                    bboxes = get_pnet_boxes(outputs, scale, THRESHOLD[0])
                    bboxes, _ = non_max_suppression(bboxes, 0.5)

                    total_boxes.append(bboxes)

                total_boxes = np.vstack(total_boxes)

                bboxes, _ = non_max_suppression(total_boxes, 0.7)
                bboxes = bbox_regression(total_boxes)

                bboxes = bbox_to_square(bboxes)
                bboxes = padding(bboxes, image_height, image_width)

                print('After PNet bboxes shape: ', bboxes.shape)
                if bboxes.shape[0] == 0:
                    draw_and_show(im_bk, bboxes)
                    break

                # --------------------------------------------------------------
                # Second stage.
                #
                inputs = get_inputs_from_bboxes(im, bboxes, 24)
                N, C, H, W = inputs.shape

                RNet.blobs['data'].reshape(N, 3, H, W)
                RNet.blobs['data'].data[...] = inputs
                outputs = RNet.forward()

                bboxes = get_rnet_boxes(bboxes, outputs, THRESHOLD[1])

                bboxes, _ = non_max_suppression(bboxes, 0.7)
                bboxes = bbox_regression(bboxes)
                bboxes = bbox_to_square(bboxes)
                bboxes = padding(bboxes, image_height, image_width)

                print('After RNet bboxes shape: ', bboxes.shape)
                if bboxes.shape[0] == 0:
                    draw_and_show(im_bk, bboxes)
                    break

                # --------------------------------------------------------------
                # Third stage.
                #
                inputs = get_inputs_from_bboxes(im, bboxes, 48)
                N, C, H, W = inputs.shape

                ONet.blobs['data'].reshape(N, 3, H, W)
                ONet.blobs['data'].data[...] = inputs
                outputs = ONet.forward()

                bboxes, points = get_onet_boxes(bboxes, outputs, THRESHOLD[2])
                bboxes = bbox_regression(bboxes)

                bboxes, picked_indices = non_max_suppression(bboxes, 0.7, 'min')
                points = points[picked_indices]
                bboxes = padding(bboxes, image_height, image_width)

                print('After ONet bboxes shape: ', bboxes.shape, '\n')
                if bboxes.shape[0] == 0:
                    draw_and_show(im_bk, bboxes, points)
                    break

                t2 = time.time()
                print('Total time: %.3fs\n' % (t2 - t1))

              #  draw_and_show(im_bk, bboxes, points)

                num_boxes = bboxes.shape[0]
                Boxnum=0   #记录实际人脸个数
                CoordinateList=[None]*num_boxes   #记录每个人脸对应的坐标点
                ROIimage=[None]*num_boxes
                if num_boxes != 0:
                    for i in range(num_boxes):
                        box = bboxes[i]
                        x1 = int(box[0])
                        y1 = int(box[1])
                        x2 = int(box[2])
                        y2 = int(box[3])
                        if (x2-x1<50) or (y2-y1<50) or (y2-y1>x2-x1+30):
                            continue

                        cv2.rectangle(im_bk, (y1, x1), (y2, x2), (0, 255, 255), 2)
                        CoordinateList[Boxnum]=(y1,x1)
                        ROIimage[Boxnum]=im_bk[x1:x2+1,y1:y2+1]
                        Boxnum=Boxnum+1



               # predictor_path = '/home/kjin/Qt/Face/shape_predictor_68_face_landmarks.dat'
                #detector = dlib.get_frontal_face_detector()
                #sp = dlib.shape_predictor(predictor_path)



                # X1='/home/kjin/caffe-master/examples/VGGNet/a.jpg'
                # X = DeepIDTest.read_imagelist(X1)
                # # data_1 是输入层的名字
                # out = net.forward_all(data=X)
                # print out
                # feature1 = np.float64(out['fc7'])
                # feature1 = np.reshape(feature1,(1,4096))



                #win = dlib.image_window()
                #color_red=dlib.rgb_pixel(255,0,0)
                #color_green=dlib.rgb_pixel(0,255,0)
                #clf = joblib.load('/home/kjin/caffe-master/examples/VGGNet/clf.model')
                   # starttime=datetime.datetime.now()
               # ret, frame = cap.read()
                #im_bk = im.copy()

               # cv2.imwrite('/home/kjin/caffe-master/examples/VGGNet/f.jpg',im)
               # img=skimage.io.imread('/home/kjin/caffe-master/examples/VGGNet/f.jpg')
               # endtime
                #win.set_image(img)
                #cv2.waitKey(50)
                #start1=time.clock()
               # dets = detector(frame, 1)
               # end1 = time.clock()
               # print("Number of faces detected: {}".format(len(dets)))
               # if len(dets)==0:
              #      win.clear_overlay()
                    # Now process each face we found.
             #   for k, d in enumerate(dets):
             #       print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))

              #      if len(dets) > 0 and d.left()>0 and d.top()>0 and d.right()>0 and d.bottom()>0: #由于d.top以及其他的会出现负值
                # ROIimage=im[d.top():d.bottom(), d.left():d.right()]
                for i in range(Boxnum):
                 cv2.imwrite('/home/kjin/caffe-master/examples/VGGNet/'+str(i)+'.jpg', ROIimage[i])

                     #win.clear_overlay()
                 # start2 = time.clock()
                 # shape = sp(img, d)
                 # end2 = time.clock()
                 #win.add_overlay(d)
                 #win.add_overlay(shape)
                 #win.add_overlay(d,)
                 # ROIimage=Image.open('/home/kjin/caffe-master/examples/VGGNet/e.jpg')
                 # CropFace(ROIimage, eye_left=shape.part(39), eye_right=shape.part(42), offset_pct=(0,0), dest_sz=(224,224)).save('/home/kjin/caffe-master/examples/VGGNet/e.jpg')
                 #cv2.waitKey(50)
                feature2=[None]*Boxnum
                for i in range(Boxnum):

                    X2='/home/kjin/caffe-master/examples/VGGNet/'+str(i)+'.jpg'
                 #X2 = '/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/Others/Repired/Test96.jpg'
                    X = DeepIDTest.read_imagelist(X2)
                    out = net.forward_all(data=X)
               # print out
                    feature2[i] = np.float64(out['fc7'])
            #  Feature_Final=[None]
            #  Feature_Final = np.reshape(feature2, 4096)
            #
            #  prob=clf.predict(Feature_Final)
            #  getscore=clf.score(Feature_Final,[1])
            #
            # # 计算每个特征之间的距离
            #  if prob[0]==1:
            #      accuracy='others'
            #  else:
            #      accuracy='me'


                min_predicts=1
                for Num in range(Boxnum):
                    i = 0
                    for list in Lists:
                        predicts = pw.pairwise_distances(Feature[i], feature2[Num], metric=metric)
                        i=i+1
                    # predicts = mt
                        if predicts[0][0] >0.12:
                            if predicts[0][0] < min_predicts:
                                min_predicts = predicts[0][0]
                            if i==(len(Lists)-1):
                                accuracy=1
                                #cv2.putText(im_bk,'Unknows',(y1,x1),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255))
                                cv2.putText(im_bk, 'Unknows', CoordinateList[Num], cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
                                cv2.imwrite('/home/kjin/caffe-master/examples/VGGNet/ContrastivePicNew/AddError_'+str(Error)+'.jpg', ROIimage[Num])
                                Error=Error+1
                            # win.clear_overlay()
                            # win.add_overlay(shape)
                            # win.add_overlay(d,color_red)
                                os.system('sh /home/kjin/caffe-master/examples/VGGNet/beep.sh')
                        else:
                            min_predicts=predicts[0][0]
                            accuracy=0
                            #cv2.putText(im_bk, 'Me', (y1, x1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0))
                            cv2.putText(im_bk, 'Me', CoordinateList[Num], cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0))
                        # win.clear_overlay()
                        # win.add_overlay(shape)
                        # win.add_overlay(d, color_green)
                            break
                    cv2.imshow('result', im_bk)
                    t2 = time.time()
                    cv2.waitKey(8)
                    print str(accuracy)+'  '+str(min_predicts)+' '+str(t2-t1)










def demo_test(num, itera):
    prj = 'VGGNet'
    home = os.path.expanduser('/home/kjin')
    caffepath = home + '/caffe-master/'
    prjpath = home + '/caffe-master/examples/VGGNet/'
    datapath = home + '/caffe-master/examples/VGGNet/Face_Classify_Color/'
    types = 1
    pairs = home + '/caffe-master/examples/VGGNet/pairs.txt'
    lfwpath = home + '/caffe-master/examples/VGGNet/lfwcrop_color/faces/'

    test = DeepIDTest(prj, caffepath, prjpath, datapath, num, types, pairs, itera, lfwpath)

    test.evaluate(metric='cosine')






def Distance(p1, p2):
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.sqrt(dx * dx + dy * dy)

        # 根据参数，求仿射变换矩阵和变换后的图像。

def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
        if (scale is None) and (center is None):
            return image.rotate(angle=angle, resample=resample)
        nx, ny = x, y = center
        sx = sy = 1.0
        if new_center:
            (nx, ny) = new_center
        if scale:
            (sx, sy) = (scale, scale)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        a = cosine / sx
        b = sine / sx
        c = x - nx * a - ny * b
        d = -sine / sy
        e = cosine / sy
        f = y - nx * d - ny * e
        return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)
        # 根据所给的人脸图像，眼睛坐标位置，偏移比例，输出的大小，来进行裁剪。

def CropFace(image, eye_left, eye_right, offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
        # calculate offsets in original image 计算在原始图像上的偏移。
        offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
        offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
        # get the direction  计算眼睛的方向。
        eye_direction = (eye_right.x - eye_left.x, eye_right.y - eye_left.y)
        # calc rotation angle in radians  计算旋转的方向弧度。
        rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
        # distance between them  # 计算两眼之间的距离。
        dist = Distance(eye_left, eye_right)
        # calculate the reference eye-width    计算最后输出的图像两只眼睛之间的距离。
        reference = dest_sz[0] - 2.0 * offset_h
        # scale factor   # 计算尺度因子。
        scale = float(dist) / float(reference)
        # rotate original around the left eye  # 原图像绕着左眼的坐标旋转。
        eyeleft_yuanzu=(eye_left.x,eye_left.y)
        image = ScaleRotateTranslate(image, center=eyeleft_yuanzu, angle=rotation)
        # crop the rotated image  # 剪切
        # crop_xy = (eye_left.x - scale * offset_h, eye_left.y - scale * offset_v)  # 起点
        # crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)  # 大小
        # image = image.crop(
        #     (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
        # # resize it 重置大小
        # image = image.resize(dest_sz, Image.ANTIALIAS)
        # image.show()
        return image

if __name__ == '__main__':
    num = 10575  # 人数
    itera = 122714  # 所选模型的迭代次数

    demo_test(num, itera)
