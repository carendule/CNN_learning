import tensorflow as tf
import scipy
from PIL import Image
import numpy as np
import h5py
import time
import os

def resize_img():
    dirs = os.listdir("augmentation_pic")
    counter = 0
    for filename in dirs:
        im = tf.gfile.FastGFile("augmentation_pic//{}".format(filename),'rb').read()
        print("正在处理第%d张图片"%counter)
        with tf.Session() as sess:
            img_data = tf.image.decode_jpeg(im)
            image_float = tf.image.convert_image_dtype(img_data,tf.float32)
            resized = tf.image.resize_images(image_float,[60,107],method=3)
            resized_im = resized.eval()
            scipy.misc.imsave("resized_img//{}".format(filename),resized_im)
            print("正在保存第%d张图片"%counter)
            counter+=1

def img_to_h5():
    dirs = os.listdir("resized_img")
    Y = [] #标签
    X = [] #数据
    print("转换h5的文件数量%d"%len(dirs))
    for filename in dirs:
        label = filename.split('_')[0]
        Y.append(label)
        im = Image.open("resized_img//{}".format(filename)).convert('RGB')
        mat = np.asarray(im)
        X.append(mat)
    dt = h5py.special_dtype(vlen=str)
    file = h5py.File("h5img//data.h5","w")
    file.create_dataset('X',data=np.array(X))
    ds = file.create_dataset('Y',np.array(Y).shape,dtype=dt)
    ds[:] = Y
    file.close()

if __name__ == "__main__":
    print("正在转换图片大小===========")
    resize_img()
    print("正在保存为h5格式===========")
    img_to_h5()
    print("保存完成==============")
