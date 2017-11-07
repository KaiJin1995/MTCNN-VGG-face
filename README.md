### 该代码源于本科毕设:基于深度学习的人脸识别安防系统。代码主要由三部分组成



- 人脸检测(Face Detection)
- 人脸验证 (Face identification) 
- 报警系统
### 该系统使用caffe环境，并要安装opencv2.7.13，具体参考https://github.com/BVLC/caffe
### 人脸检测
人脸检测主要参考MTCNN论文，该论文利用了Cascaded CNN，实现real-time face detection。
代码在
pycaffe-mtcnn-master/main_Original.py中。通过修改det1,det2,det3阈值实现人脸检测的漏检率和误检率的变化。

### 人脸识别
人脸识别采用VGG-Net网络，网络通过提取fc8-2的特征向量，对特征向量进行比较，判断两张图片的相似度。应用代码时，请提前建立好自己的人脸对比库，对比库大致需要20张不同角度的图片以提高对比准确率。

### 报警装置
本代码采用台式机自带的蜂鸣器，通过调用beep.sh代码完成在检测到错误人脸时报警的操作。该操作的缺陷是会浪费大量的时间，因此，一旦检测到错误的人脸，会卡顿。因此，有一种解决方法是使用python的多进程加速。

### 如有问题 可以发送邮箱至jink.xidian@gmail.com