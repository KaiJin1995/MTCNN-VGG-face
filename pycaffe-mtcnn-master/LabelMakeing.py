import os
PicList=os.listdir('/home/kjin/caffe-master/examples/VGGNet/pycaffe-mtcnn-master/MyImageNew/')
Command=''

for list in PicList:
    if list[0] != 'T':

        Command += '0' + '\n'
    else:

        Command += '1\n'
fin.write(Command)
fin.close()
