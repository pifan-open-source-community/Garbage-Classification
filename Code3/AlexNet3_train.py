#coding:utf-8
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
start = time.clock()
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
# from __future__ import division
import keras

from keras.layers import Flatten,BatchNormalization

from keras.layers.convolutional import Conv2D
from keras.layers.merge import add
from keras.models import Sequential

from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import numpy as np
#仅取卷积层和池化层，去掉最后一层全连接层
# MobileNet_model = MobileNet(weights = 'imagenet', include_top = False, input_shape=(128,128,3))

def AlexNet():
        model = Sequential()
        # input_shape = (64,64, self.config.channles)
        input_shape = input_shape=(128,128,3)
        model.add(Convolution2D(64, (11, 11), input_shape=input_shape,strides=(1, 1),  padding='valid',activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))#26*26
        model.add(Convolution2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Convolution2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Convolution2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Convolution2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(214, activation='softmax'))
        # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

model = AlexNet()

train_datagen = ImageDataGenerator(
rotation_range = 10, #随机旋转度数
width_shift_range = 0.1, #随机水平平移
height_shift_range = 0.1,#随机竖直平移
rescale = 1/255, #数据归一化
shear_range = 0.1, #随机裁剪
zoom_range = 0.1, #随机放大
horizontal_flip = True, #水平翻转
fill_mode = 'nearest', #填充方式
)

test_datagen = ImageDataGenerator(
rescale = 1/255, #数据归一化
#batch_size = 32
)
batch_size = 32
#生成训练数据
train_generator = train_datagen.flow_from_directory(
'../data3/train',
target_size = (128,128),
batch_size = batch_size,
)

#生成测试数据
test_generator = test_datagen.flow_from_directory(
'../data3/test',
target_size = (128,128),
batch_size = batch_size,
)

print(train_generator.class_indices)

filepath = 'trash_data3_AlexNet.h5'
# Callbacks
callbacks_list = [
ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True,mode='max',period=1)]

#定义优化器，代价函数，训练过程中计算准确率
model.compile(optimizer = SGD(lr=1e-4,momentum=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit_generator(train_generator,epochs=1000,validation_data=test_generator, callbacks=callbacks_list)

# model.save('vpice_model_vgg16_2.h5')
print(" model is save successfuly!")
end = time.clock()
print("本次训练一共一共运行了:%s秒----约等于%s分钟"%((end-start), (end-start)/60))
