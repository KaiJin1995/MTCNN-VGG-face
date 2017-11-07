#coding:utf-8
#from sklearn import svm
import caffe
import os
import sys
import numpy as np
from sklearn.externals import joblib
import skimage
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


#产生SVM训练后的model
def read_labels(label):
    '''
    读取标签列表文件
    '''
    fin=open(label)
    lines=fin.readlines()
    labels=[None]*len(lines)
    k=0
    for line in lines:
        labels[k]=int(line)
        k=k+1
    fin.close()
    return labels




def read_imagelist(X1):
    '''
    @brief：从列表文件中，读取图像数据到矩阵文件中
    @param： filelist 图像列表文件
    @return ：4D 的矩阵
    '''
    # fid = open(filelist)
    # lines = fid.readlines()
    # test_num = len(lines)
    # fid.close()
    X = np.empty((1, 3, 224, 224))
    i = 0
    # for line in lines:
    #     word = line.split('\n')
    #     filename = word[0]
    im1 = skimage.io.imread(X1, as_grey=False)
    image = skimage.transform.resize(im1, (224, 224)) * 255
    if image.ndim < 3:
        print 'gray:' + filename
        X[i, 0, :, :] = image[:, :]
        X[i, 1, :, :] = image[:, :]
        X[i, 2, :, :] = image[:, :]
    else:
        X[i, 0, :, :] = image[:, :, 0]
        X[i, 1, :, :] = image[:, :, 1]
        X[i, 2, :, :] = image[:, :, 2]
    #i = i + 1
    return X


def Training():
   # X=read_imagelist('/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/trainSVM.txt')
    caffe.set_mode_gpu()
    net = caffe.Classifier('/home/kjin/caffe-master/examples/VGGNet/VGG_FACE_deploy.prototxt', '/home/kjin/caffe-master/examples/VGGNet/VGG_test_20170512/vgg_10575_iter_150000.caffemodel', mean=np.load('/home/kjin/caffe-master/examples/VGGNet/VGGNet_mean.npy'))
    Lists = os.listdir('/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/MyImageNew/')
    Feature = [None] * len(Lists)
    i=0
    for list in Lists:
        X1 = '/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/MyImage/' + list
        X = read_imagelist(X1)
        out = net.forward_all(data=X)
        Feature[i] = np.float64(out['fc7'])
        Feature[i] = np.reshape(Feature[i], 4096)
        i = i + 1



#     clf=svm.SVC(C = 1.0, cache_size = 200, class_weight = 'balanced', coef0 = 0.0,
# decision_function_shape = None, degree = 3, gamma = 0.0001, kernel = 'rbf',
# max_iter = -1, probability = False, random_state = None, shrinking = True,
# tol = 0.001, verbose = False)


    neigh = KNeighborsClassifier(n_neighbors=3,metric='euclidean')

    labels=read_labels('/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/NewLabel.txt')


    neigh.fit(Feature,labels)
    joblib.dump(neigh, 'knn.model')







if __name__ == '__main__':
    Training()


