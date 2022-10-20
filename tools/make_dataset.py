import argparse
import os, sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

parser = argparse.ArgumentParser(description="make dataset from given images")
parser.add_argument("--raw_image_path", type=str, default="/home/hugoliu/github/dataset/carla_ipm_1280_1280/", help="The path of the raw images")
parser.add_argument("--output_image_path", type=str, default="/home/hugoliu/github/dataset/carla_ipm_768_256/", help="The path of the output images")
parser.add_argument("--raw_image_w", type=int, default=1280, help="")
parser.add_argument("--raw_image_h", type=int, default=1280, help="")
parser.add_argument("--output_image_w", type=int, default=256, help="")
parser.add_argument("--output_image_h", type=int, default=768, help="")

class ImageWrapper(object):
    def __init__(self, args, image_path, image_name):
        self.args = args
        self.image_path = image_path
        self.image_name = image_name
        self.img = cv2.imread(self.image_path + self.image_name)
        self.output_path = self.args.output_image_path + self.image_name
        self.H = args.raw_image_h
        self.W = args.raw_image_w
        h, w, c = self.img.shape
        print("read image {}, h={}, w={}".format(self.image_name, h, w))
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB).astype(np.float32)

    def resize(self, h, w):
        self.img = cv2.resize(self.img, (w, h))

    def crop(self, h0, w0, h, w):
        h1 = h0 + h
        if h1 > self.H:
            h1 = self.H
        w1 = w0 + w
        if w1 > self.W:
            w1 = self.W
        self.img = self.img[h0:h1, w0:w1, :]

    def plot_bbx(self, bbx_list):
        #bbx: [x1, y1, x2, y2]
        img_np = self.img
        for bbx in bbx_list:
            start_point = (int(bbx[0]), int(bbx[1]))
            end_point = (int(bbx[2]), int(bbx[3]))
            color = (255, 255, 0)
            img_np = cv2.rectangle(img_np, start_point, end_point, color=color, thickness=6)
            # print("plot bbx: {}".format(bbx))
        return img_np

    def save(self):
        print("save image to {}".format(self.output_path))
        cv2.imwrite(self.output_path, self.img)
        # Image.fromarray(self.img).save(output_path)

    def make(self):
        self.crop(h0=200, w0=300, h=768, w=256)  # very good
        # self.crop(h0=0, w0=100, h=1275, w=425) # not so good
        self.resize(h=args.output_image_h, w=args.output_image_w)
        # cv2.imshow("cropped", self.img)
        # cv2.waitKey(0)
        self.save()


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.output_image_path, exist_ok=True)
    image_files = os.listdir(args.raw_image_path)
    print("image_files: {}".format(image_files))
    counter = 0

    for fname in image_files:
        if counter % 10 == 0:
            image_path = args.raw_image_path
            img_wrapper = ImageWrapper(args, image_path, fname)
            img_wrapper.make()
        counter += 1
