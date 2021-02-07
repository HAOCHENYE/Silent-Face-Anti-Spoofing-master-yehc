import os
import random
import math
from PIL import Image
def transferPictures(nowpath, newpath):
    # 将文件夹下的不同类别的文件夹中的部分图片转移到另一个文件夹下的相同类别的文件夹下，并删除原文件夹中的相应图片（类似于剪切）
        for roots,dirs,files in os.walk(nowpath):

            fnum = math.floor(len(files) * 0.3)  # 计算一半数量
            rdom_files = random.sample(files, fnum)  # 随机选一半数量的图片

            for imgname in rdom_files:
                imgpath = nowpath + imgname
                # print(imgpath)
                im = Image.open(imgpath)
                im.save(newpath + imgname)
                os.remove(imgpath)  # 转移完后删除原图片


transferPictures('/media/traindata/wangch/celeba_type/train/images/org_1_80-60/screen/', '/media/traindata/wangch/celeba_type/val/images/org_1_80-60/1/')