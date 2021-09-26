#测试
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import numpy as np
import os,time,cv2
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
label = np.array(['O','R'])
#载入模型
model = load_model('trash_data2_AlexNet.h5')


def predict(img_path):
    #导入图片
    #image = load_img(img_path)
    image = img_path
    # print("d导入图片是:cat")
    image = image.resize((128,128))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,0)
    result = label[model.predict_classes(image)]
    print(result)
    # return result


# video = "http://admin:940024@192.168.191.2:8081/"
camera = cv2.VideoCapture(0)
n = 0
while True:
    ret, frame = camera.read()
    cv2.imshow('frame',frame)
    n = n + 1
    # 这个地方得意思是20帧识别一次 每帧都识别的话实时性可能不好
    if n % 20 == 0:

        frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        predict(frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()