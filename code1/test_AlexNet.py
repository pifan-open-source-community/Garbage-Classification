#测试
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import numpy as np
import os,time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
label = np.array(['O','R'])
#载入模型
model = load_model('trash_data2_AlexNet.h5')

def predict(img_path):
    #导入图片
    image = load_img(img_path)
    # print("d导入图片是:cat")
    image = image.resize((128,128))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,0)
    result = label[model.predict_classes(image)]
    # print(result)
    return result

img_dir = '../data2/test/R/'
imgs_name = os.listdir(img_dir)

start = time.clock()
count = 0
for i in imgs_name:
    img_name = img_dir + i
    re = predict(img_name)
    print("当前物品预测为：{}".format(re[0]))
    if re == 'R':
        count =count+1
print("acc: {}".format(count/1112))
end = time.clock()
print("本次训练一共一共运行了:%s秒----约等于%s分钟"%((end-start), (end-start)/60))
print("单张:%s秒"%((end-start)/1112))