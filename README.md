![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/16-10.png)

```
代码仓库
1、码云Gitee：https://gitee.com/yangkun_monster/raspberrypi-Garbage-classification
2、Github：https://github.com/pifan-open-source-community/garbage-Classification
视频教程地址：
哔哩哔哩bilibili：树莓派爱好者基地
```

## 一、项目概述

简介：该垃圾分类项目主要在于对各种垃圾进行所属归类，本次项目采用keras深度学习框架搭建卷积神经网络模型实现图像分类，最终移植在树莓派上进行实时视频流的垃圾识别。

 

前期：主要考虑PC端性能，并尽可能优化模型大小，训练可采用GPU，但调用模型测试的时候用CPU运行，测试帧率和准确性（测试10张左右图像的运行时间取平均值或实时视频流的帧率）。

 

后期：部署在树莓派端，在本地进行USB摄像头实时视频流的垃圾分类（归类）。

 

框架语言：  keras+python。

PC端：

Keras:  2.2.0

Opencv:  3.4

Python: 3.6

Numpy:1.16

## 二、数据集



data1: <https://www.kaggle.com/asdasdasasdas/garbage-classification>

数据集包含6个分类：cardboard (393), glass (491), metal (400), paper(584), plastic (472) andtrash(127).

 

data2: <https://www.kesci.com/home/dataset/5d133d11708b90002c570588>

该数据集是图片数据，分为训练集85%（Train）和测试集15%（Test）。其中O代表Organic（有机垃圾），R代表Recycle（可回收）。

data3 : <https://copyfuture.com/blogs-details/2020083113423317484akwfwu4mzs89w>

一共 56528 张图片，214 类，总共 7.13 GB。



## 三、leNet5 模型搭建

本次项目采用深度学习来进行图像识别，如今深度学习中最流行的无疑是卷积神经网络，因此，我们搭建了包含5层卷积层的神经网络来进行垃圾分类。

 

由于本次项目包含三个数据集，对应三个类别（6分类，2分类，214分类），但是设计的模型都是一样的，因此，下面就以data1进行网络搭建、训练、测试讲解。

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%201.png)


在正式训练之前我们还使用了数据增广技术（ImageDataGenerator）来对我们的小数据集进行数据增强（对数据集图像进行随机旋转、移动、翻转、剪切等），以加强模型的泛化能力。

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%202.png)

#### 1、模型构建

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%203.png)

其中conv2d表示执行卷积，maxpooling2d表示执行最大池化，Activation表示特定的激活函数类型，Flatten层用来将输入“压平”，用于卷积层到全连接层的过渡，Dense表示全连接层（128-128-6，最后一位表示分类数目）。

 参数设置：为训练设置一些参数，比如训练的epoches，batch_szie，learning rate等

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%204.png)

在这里我们使用了SGD优化器，由于这个任务是一个多分类问题，可以使用类别交叉熵（categorical_crossentropy）。但如果执行的分类任务仅有两类，那损失函数应更换为二进制交叉熵损失函数（binary cross-entropy）

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%205.png)

#### 2、模型保存

将神经网络在data1数据集上训练的结果（参数，权重文件）进行保存，方便后期调用训练好的模型进行预测。

模型保存文件名为：trash_data1_AlexNet3.h5， 我们设置为保存模型效果最好的一次。

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%206.png)

## 四、训练并测试

首先是观察数据，看看我们要识别的垃圾种类有多少，以及每一类的图片有多少。

#### 1、训练结果

训练代码已经写好了，接下来开始训练（图片归一化尺寸为128，batch_size为32，epoches为5000，一般5k就已经算比较多的啦，效果好的话可以提前结束）。

进行训练

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%207.png)

 训练过程中的打印结果：

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%208.png)

#### 2、模型保存

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%209.png)

#### 3、预测单张图片

现在我们已经得到了我们训练好的模型trash_data1_AlexNet3.h5，然后我们编写一个专门用于预测的脚本predict.py

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2010.png)

预测脚本中的代码编写思路是：载入训练好的模型-》读入图片信息-》预测-》展示预测效果

我们这里写了一个循环测试，方便计算单张图像的预测时间

 

#### 4、测试结果

data1:

model size: 178M

acc accuracy(test) :86%

 硬件：AMD R5 3600  内存：16G

 测试100张图像耗时：2.37s

单张图像耗时：0.0237s

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2011.png)

data2:

model size: 128M

acc accuracy(test) :94%

硬件：AMD R5 3600  内存：16G

 测试1112张图像耗时：90.52s

单张图像耗时：0.0814 

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2012.png)

data3

 model size: 128M

acc accuracy(test) :72%

硬件：AMD R5 3600  内存：16G

 测试1112张图像耗时：8.69s

单张图像耗时：0.077

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2013.png)



## 五、树莓派端部署/配置深度学习环境

**系统环境：2020-08-20-raspios-buster-armhf-full**

**工程要求：Tensorflow 1.14.0+ Keras 2.2.4 + Python 3.7 **

#### 1、配置好ssh和vnc之后，换源：

第一步，先备份源文件

```
sudo cp/etc/apt/sources.list /etc/apt/sources.list.bak

sudo cp/etc/apt/sources.list.d/raspi.list /etc/apt/sources.list.d/raspi.list.bak
```

第二步，编辑系统源文件

```
sudo nano/etc/apt/sources.list
```

第三步，将初始的源使用#注释掉，添加如下两行清华的镜像源。Ctrl+O ++ Ctrl+X

【注意】这里的树莓派系统是Raspbian-buster系统，在写系统源链接时要注意是buster，网上很多教程都是之前stretch版本，容易出错！

```
debhttp://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ buster main contribnon-free rpi

deb-srchttp://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ buster main contribnon-free rpi
```

第四步，保存执行如下命令sudo apt-get update，完成源的更新软件包索引。

```
sudo apt-get update&&upgrade
```

第五步，还需要更改系统源

```
sudo nano/etc/apt/sources.list.d/raspi.list
```

用#注释掉原文件内容，用以下内容取代：用#注释掉原文件内容，用以下内容取代：

```
debhttp://mirrors.tuna.tsinghua.edu.cn/raspberrypi/ buster main ui

deb-srchttp://mirrors.tuna.tsinghua.edu.cn/raspberrypi/ buster main ui
```

第六步，配置换源脚本，更改pip源

新建文件夹：

```
mkdir ~/.pip

sudo nano~/.pip/pip.conf
```

在pip.conf文件中输入以下内容：

```
[global]

timeout=100

index-url=https://pypi.tuna.tsinghua.edu.cn/simple/

extra-index-url=http://mirrors.aliyun.com/pypi/simple/

[install]

trusted-host=

        pypi.tuna.tsinghua.edu.cn

        mirrors.aliyun.com
```



#### 2、python虚拟环境配置

首先进行系统软件包更新

```
sudo apt-getupdate 

sudo apt-getupgrade

sudorpi-update
```

然后更新自带的pip，由于Raspbian自带的pip3为9.0.1版本，较为老旧，我们使用以下命令来更新pip3：

```
python3 -mpip install --upgrade pip  
```

尝试在更新完pip3后，键入命令：

```
pip3 list
```

新建个文件夹（虚拟环境用）

```
cd Desktop

mkdir tf_pi

cd tf_pi
```

安装虚拟环境这个好东西

```
python3 -mpip install virtualenv
```

增加环境变量，使得该好东西可以用起来

```
sudo chmod -R777 /root/.bashrc

sudo nano ~/.bashrc
```

把exportPATH=/home/pi/.local/bin/:$PATH  放到最后,添加环境变量

```
source ~/.bashrc
```

成功了之后：整一个虚拟环境

```
virtualenvenv

sourceenv/bin/activate
```

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2014.png)

3、安装tensorflow1.14.0

用电脑下载：（链接）python3.7版本只能安装1.14.0-Buster版本的TensorFlow

<https://github.com/lhelontra/tensorflow-on-arm/releases/tag/v1.14.0-buster>

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2015.png)

用U盘将这个文件拷到树莓派上，建一个bag文件夹存放

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2016.png)

安装依赖包： 

```
sudo aptinstall libatlas-base-dev
```

安装一些环境

```
sudo apt-getinstall -y libhdf5-dev libc-ares-dev libeigen3-dev

python3 -mpip install keras_applications==1.0.8 --no-deps

python3 -mpip install keras_preprocessing==1.1.0 --no-deps

python3 -mpip install h5py==2.9.0

sudo apt-getinstall -y openmpi-bin libopenmpi-dev

sudo apt-getinstall -y libatlas-base-dev

python3 -mpip install -U six wheel mock
```

安装tensorflow

```
cd env

cd bag

pip3 install tensorflow-1.14.0-cp37-none-linux_armv7l.whl
```

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2017.png)

这里要安装二十分钟。。。出错了再来一遍就好了。。

测试是否成功并查看版本：

```
python

import tensorflow as tf

tf.version
```

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2019.png)

#### 4、安装keras

安装一些依赖

```
sudo apt-getinstall libhdf5-serial-dev

pip3 installh5py

sudo apt-getinstall gfortran

sudo aptinstall libopenblas-dev

pip3 install-i https://pypi.tuna.tsinghua.edu.cn/simple/ pillow

sudo pip3install pybind11
```

**第一个下载numpy：**第一次的时候发现安装成功但调用失败了，我觉得是numpy版本过高导致出错了****

下载keras还是tensorflow的时候会自动下载numpy(之前已存在，它会先卸载再安装高版本的numpy，之前不存在，它会直接安装高版本的numpy),所以要先下载keras,再卸载numpy,然后再安装低版本的numpy

看一下子numpy版本，太高了

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2020.png)

重新安装

```
pip3uninstall numpy

pip3 installnumpy==1.16.0
```



第二个下载scipy【直接pip安装百分之九十九的可能都会失败。所以先下载再安装。。。先下载这个链接复制到树莓派上，然后解压到指定文件夹/home/pi/Desktop/tf_pi/env/lib/python3.7/site-packages下】

https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/aa/d5/dd06fe0e274e579e1dff21aa021219c039df40e39709fabe559faed072a5/scipy-1.5.4.tar.gz

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2021.png)

```
cd/home/pi/Desktop/tf_pi/env/bag

tar -zxvf scipy-1.5.4.tar.gz-C /home/pi/Desktop/tf_pi/env/lib/python3.7/site-packages

```

然后进到这个文件夹里开启安装：【花里胡哨的各种代码配置呀啥的，会安装三十分钟左右】

```
cd /home/pi/Desktop/tf_pi/env/lib/python3.7/site-packages/scipy-1.5.4

pythonsetup.py install
```

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2022.png)

pip3 list看一看：【太六了，终于成功了】

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2023.png)

再使用这个命令安装keras：

```
pip3 install keras==2.2.4
```

请注意；由于在virtualenv里面，一定一定要避免sudo pip3 install，否则会安装到默认路径下！发现keras安装到默认环境了，所以调用不成功，pip list没有

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2024.png)

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2025.png)

**解决办法重新安装**

```
pip3install keras==2.2.4
```

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2026.png)



安装好了之后记得reboot重启一下子。



#### 5、开始测试；import keras前面加import os就能忽略提示

**因为keras可以配合很多框架，我们用的tf所以会有backend的提示**

**进入虚拟环境：**

```
cd ~/Desktop/tf_pi

sourceenv/bin/activate
```

```
python

import tensorflowas tf

tf.__version__

import keras

print(keras.__version__)
```

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2027.png)

## 六、用树莓派跑分类识别的代码

**系统环境：2020-08-20-raspios-buster-armhf-full**

**工程要求：Tensorflow 1.14.0+ Keras 2.2.4 + Python 3.7 **



#### 1、把代码还有图片集，拷到树莓派上

其实Filezilla这个FTP传输就很方便

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2028.png)

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2029.png)

#### 2、激活虚拟环境

```
cd ~/Desktop/tf_pi

sourceenv/bin/activate
```



#### 3、克隆代码并进入代码目录

克隆代码

```
cd ~/Desktop/tf_pi/env

git clone https://github.com/pifan-open-source-community/Garbage-Classification.git
```

若提示git命令未找到：

```
sudo apt-get install git

```

进入代码目录：

```
cd ~/Desktop/tf_pi/env/Garbage-Classification/code1
```

这里更改test.py的测试集路径

```
pythontest.py
```

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2030.png)

发现有个文件解码有问题，于是根据错误的消息的路径，去这里：

```
/home/pi/Desktop/tf_pi/env/lib/python3.7/site-packages/keras/engine
```

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2031.png)

在.decode('utf-8')前面加.encode('utf8')

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2032.png)

再次到测试这里运行python test.py，解决了!

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2033.png)

**测试AlexNet需要把test.py文件里的权重文件路径改了，把输入图片维度由(150,150) 改为(128,128)**



Code1测试结果：test100张玻璃

| **网络** | **VGG16** | **AlexNet** |
| ------ | --------- | ----------- |
| **精度** | **89%**   | **87%**     |
| **时间** | **307**秒  | **80**秒     |

Code2测试结果：test100张窗帘（两种，R可回收，O不可回收）

| **网络** | **VGG16** | **AlexNet** |
| ------ | --------- | ----------- |
| **精度** | **98%**   | **98%**     |
| **时间** | **309**秒  | **46**秒     |

## 七、树莓派安装opencv并测试视频接口

**系统环境：2020-08-20-raspios-buster-armhf-full**

**工程要求：opencv 3.4.6.27**



```
cd ~/Desktop/tf_pi

source env/bin/activate
```



```
cd ~/Desktop/tf_pi/env/Garbage-Classification/code1



python data1_video_test.py 
```



#### 1、安装必要的库

```
pip3 install numpy



sudo apt-get install libhdf5-dev -y build-dep libhdf5-dev

sudo apt-get install libatlas-base-dev -y

sudo apt-get install libjasper-dev -y

sudo apt-get install libqt4-test -y

sudo apt-get install libqtgui4 -y

sudo apt install libqt4-test

pip3 install libqtgui4



sudo apt-get install cmake

sudo apt  installcmake-qt-gui

sudo apt-get install libgtk2.0-dev

sudo apt-get install pkg-config



pip3 install boost

pip3 install dlib
```



#### 2、电脑浏览器下载以下两个文件

https://www.piwheels.org/simple/opencv-contrib-python/opencv_contrib_python-3.4.6.27-cp37-cp37m-linux_armv7l.whl

https://www.piwheels.org/simple/opencv-python/opencv_python-3.4.6.27-cp37-cp37m-linux_armv7l.whl

#### 3、将两个文件拷贝到树莓派上去

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2034.png)

#### 4、安装这两个文件，先更新pip

```
pip install --upgrade pip

pip3 install 文件位置  
```



注意，由于是虚拟环境，就不能做sudo，会安装到默认路径

```
cd env/bag

pip3 installopencv_contrib_python-3.4.6.27-cp37-cp37m-linux_armv7l.whl

pip3 install opencv_python-3.4.6.27-cp37-cp37m-linux_armv7l.whl
```



#### 5、测试

先打开摄像头设置

```
sudo raspi-config
```

然后运行摄像头程序

![image](https://github.com/pifan-open-source-community/garbage-Classification/blob/main/image/%E5%9B%BE%E7%89%87%2035.png)





