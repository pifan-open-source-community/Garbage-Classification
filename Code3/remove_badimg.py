import warnings,os,glob
warnings.filterwarnings("error", category=UserWarning)

path = '../data3/test/'

cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
for f in cate:
    # print(f)
    for im in glob.glob(f + '/*.jpg'):
        #print(im)
        try:
            img = Image.open(im)
        except:
            print('corrupt img', im)
            print(img)
            os.remove(im)