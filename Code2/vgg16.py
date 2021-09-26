#coding:utf-8
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
start = time.clock()
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.callbacks import ModelCheckpoint
import numpy as np
#仅取卷积层和池化层，去掉最后一层全连接层
vgg16_model = VGG16(weights = 'imagenet', include_top = False, input_shape=(150,150,3))

#搭建全连接层
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:])) #降成一维 从1开始图片的宽，高，通道数，0是数量所以不要
top_model.add(Dense(256,activation = 'relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(214,activation = 'softmax'))

model = Sequential()
model.add(vgg16_model)
model.add(top_model)

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
target_size = (150,150),
batch_size = batch_size,
)

#生成测试数据
test_generator = test_datagen.flow_from_directory(
'../data3/test',
target_size = (150,150),
batch_size = batch_size,
)

print(train_generator.class_indices)

filepath = 'trash_data3_model_vgg16.h5'
# Callbacks
callbacks_list = [
ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True,mode='max',period=1)]

#定义优化器，代价函数，训练过程中计算准确率
model.compile(optimizer = SGD(lr=1e-4,momentum=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit_generator(train_generator,epochs=500,validation_data=test_generator, callbacks=callbacks_list)

# model.save('vpice_model_vgg16.h5')
print(" model is save successfuly!")
end = time.clock()
print("本次训练一共一共运行了:%s秒----约等于%s分钟"%((end-start), (end-start)/60))
