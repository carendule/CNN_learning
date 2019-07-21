from keras_preprocessing.image import ImageDataGenerator,img_to_array,load_img
import os

datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.15
                             ,height_shift_range=0.15,zoom_range=0.15
                             ,shear_range=0.2,horizontal_flip=False
                             ,fill_mode='nearest')
dirs = os.listdir("picture")
print("文件总数%d"%len(dirs))
for filename in dirs:
    img = load_img("picture//{}".format(filename))
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)
    datagen.fit(x)
    prefix = filename.split('.')[0]
    print(prefix)
    counter = 0
    for batch in datagen.flow(x,batch_size=4,save_to_dir="augmentation_pic",save_prefix=prefix,save_format='jpg'):
        print("生成图片增强%s第%d张"%(filename,counter))
        counter+=1
        if counter > 400:
            break
