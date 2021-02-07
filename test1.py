# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


# SAMPLE_IMAGE_PATH = "/media/traindata/wangch/celeba_spoof_resize/test/images/org_1_80-60/0/"
SAMPLE_IMAGE_PATH = "images/sample/"
RESULT_IMAGE_PATH = "images/result/"

# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

# if __name__ == "__main__":
def arg_parse():
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=2,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="work_dirs/RepVGGNet",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F3.jpg",
        help="image used to test")
    args = parser.parse_args()
    return args

# if __name__ == "__main__":
args = arg_parse()
# test(args.image_name, args.model_dir, args.device_id)
# def test(image_name, model_dir, device_id):
image_name = args.image_name
model_dir = args.model_dir
device_id = args.device_id
model_test = AntiSpoofPredict(device_id)
image_cropper = CropImage()
image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
result = check_image(image)
if result is True:
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    model_names = [os.path.join(model_dir, "epoch_77.pth")]
    # sum the prediction from single model's result
    for model_name in model_names:
        # h_input, w_input, model_type, scale = parse_model_name(model_name)
        h_input, w_input, model_type, scale = 80, 80, None, 1
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": False,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, model_name)
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    num = 1
    value = prediction[0][label]/num
    if label == 0:
        print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result_MiniFASNetV2_cls_3_0.005" + format_)
    cv2.imwrite(RESULT_IMAGE_PATH + result_image_name, image)