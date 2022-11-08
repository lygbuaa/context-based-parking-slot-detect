import argparse
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES']="-1"
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

from parking_context_recognizer.PcrWrapper import PCRWrapper
from parking_context_recognizer.config import *

from parking_context_recognizer import train as pcr_train
from parking_slot_detector import test_carla as psd_test
from parking_slot_detector import merge_three_type_result_files as merge_result
from parking_slot_detector.utils import eval_utils as eval_utils

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="context-based parking slot detector")

parser.add_argument("--carla_image_path", type=str, default="pil_park/carla_town04/image/",
                    help="The path of the parking slot detection dataset")
parser.add_argument("--result_path", type=str, default="result/",
                    help="The path of the parking slot detection result")

parser.add_argument("--pcr_test_weight", type=str, default="weight/weight_pcr/trained/trained.ckpt",
                    help="The path of the trained weights of pcr.")

parser.add_argument("--raw_image_w", type=int, default=640, help="")
parser.add_argument("--raw_image_h", type=int, default=640, help="")

parser.add_argument("--psd_test_weight_type0", type=str, default="weight/weight_psd/fine_tuned_type_0",
                    help="The path of the trained weights of fine-tuned to parallel type.")

parser.add_argument("--psd_test_weight_type1", type=str, default="weight/weight_psd/fine_tuned_type_1",
                    help="The path of the trained weights of fine-tuned to perpendicular type.")

parser.add_argument("--psd_test_weight_type2", type=str, default="weight/weight_psd/fine_tuned_type_2",
                    help="The path of the trained weights of fine-tuned to diagonal type.")

parser.add_argument("--threshold_score", type=float, default=0.8,
                    help="Threshold of prediction which is determined TRUE")

class ImageWrapper(object):
    def __init__(self, image_path, raw_h, raw_w):
        self.raw_h = raw_h
        self.raw_w = raw_w
        self.image_path = image_path
        self.img_tensor = None
        self.img_np = None

    def cv_read_resize(self, w, h):
        self.img_np = cv2.imread(self.image_path)
        self.img_np = cv2.resize(self.img_np, (w, h))

    def png_to_tensor(self):
        print("open image: {}".format(self.image_path))
        img_png = tf.gfile.FastGFile(self.image_path, 'rb').read()
        self.img_tensor = tf.image.decode_png(img_png, channels=3)
        with tf.Session("") as sess:
            self.img_np = sess.run(self.img_tensor)
        self.img_tensor = tf.image.resize(self.img_tensor, [self.raw_h, self.raw_w])
        print("img_tensor: {}".format(self.img_tensor.shape))
        return self.img_tensor

    def to_numpy(self):
        return self.img_np

    # output the same tensor with "parking_context_recognizer/utils.py"
    def resize_tensor(self, h=INPUT_HEIGHT, w=INPUT_WIDTH, standarization=True, expand_dim0=True):
        self.img_tensor = tf.image.resize(self.img_tensor, [h, w])
        # print("img_tensor: {}".format(self.img_tensor.shape))
        if standarization:
            self.img_tensor = tf.image.per_image_standardization(self.img_tensor)
        if expand_dim0:
            self.img_tensor = tf.expand_dims(self.img_tensor, axis=0)
        return self.img_tensor

    # output the same tensor with "parking_context_recognizer/utils.py"
    def crop_resize_tensor(self, out_h=INPUT_HEIGHT, out_w=INPUT_WIDTH, standarization=True, expand_dim0=True):
        img_size = self.img_tensor.shape
        h = int(img_size[0])
        w = int(h/3.0)
        h = int(3*w)
        print("crop size: h: {}, w: {}".format(h, w))
        self.img_tensor = self.img_tensor[0:h, 0:w, :]
        self.img_tensor = tf.image.resize(self.img_tensor, [out_h, out_w])
        if standarization:
            self.img_tensor = tf.image.per_image_standardization(self.img_tensor)
        if expand_dim0:
            self.img_tensor = tf.expand_dims(self.img_tensor, axis=0)
        return self.img_tensor

    def plot_bbx(self, bbx_list):
        #bbx: [x1, y1, x2, y2]
        img_np = self.img_np
        for bbx in bbx_list:
            start_point = (int(bbx[0]), int(bbx[1]))
            end_point = (int(bbx[2]), int(bbx[3]))
            color = (255, 255, 0)
            img_np = cv2.rectangle(img_np, start_point, end_point, color=color, thickness=6)
            # print("plot bbx: {}".format(bbx))
        return img_np

    def save(self, output_path, img_np=None):
        if img_np is None:
            img_np = self.img_np
        Image.fromarray(img_np).save(output_path)

    def plot_preds(self, dict, output_path):
        idx = dict["idx"]
        angle = dict["angle"]
        type = dict["type"]
        h = dict["det_h"]
        w = dict["det_w"]
        self.img_np = cv2.putText(
            img = self.img_np,
            text = "{:d}: {:d}, {:d}".format(idx, type, angle),
            org = (w-100, 20),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 0.6,
            color = (0, 255, 255),
            thickness = 1
        )

        pred_list = dict["pred"]
        for pred in pred_list:
            quads = pred["quad"]
            is_empty = (int(pred["label"])==0)
            xmin = int(pred["bbx"][0])
            if xmin<0:
                xmin = 0
            ymin = int(pred["bbx"][1])
            if ymin < 0:
                ymin = 0
            
            N = 4
            vertices = []
            for j in range(N):
                pt = (int(quads[2*j + 0]), int(quads[2*j + 1]))
                vertices.append(pt)
            # draw lines
            if is_empty:
                line_color = (0, 255, 0)
            else:
                line_color = (0, 0, 255)
            for i in range(N):
                cv2.line(
                    img = self.img_np, 
                    pt1 = vertices[i], 
                    pt2 = vertices[(i+1)%N], 
                    color = line_color,
                    thickness = 2,
                    lineType = cv2.LINE_8
                    )
            # print score
            self.img_np = cv2.putText(
                img = self.img_np,
                text = "{:.2f}".format(pred["score"]),
                org = (xmin, ymin),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 0.6,
                color = (0, 255, 255),
                thickness = 1
            )            

        cv2.imwrite(output_path, self.img_np)


class CarlaEvaluator(object):
    def __init__(self, args):
        self.args = args
        self.output_dir = self.args.result_path
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_files = os.listdir(self.args.carla_image_path)
        print("total image: {}".format(len(self.image_files)))
        self.pcr_model = PCRWrapper(self.args.pcr_test_weight)
        self.res_json_list = []
        # self.psd_model = PSDWrapper(self.args.psd_test_weight_type0)

    def make_pcr_inputs(self):
        counter = 0
        tensor_list = []
        for fname in self.image_files:
            
            image_path = self.args.carla_image_path + fname
            img_wrapper = ImageWrapper(image_path, self.args.raw_image_h, self.args.raw_image_w)
            img_wrapper.png_to_tensor()
            img_tensor = img_wrapper.crop_resize_tensor(out_h=INPUT_HEIGHT, out_w=INPUT_WIDTH, standarization=True)

            # if counter == 1:
            #     self.pcr_init_tensor = img_tensor
            tensor_list.append(img_tensor)

            dict = {}
            dict["idx"] = counter
            dict["img_path"] = image_path
            self.res_json_list.append(dict)
            counter += 1
            if counter > 0:
                break

        self.pcr_input_tensor = tf.concat(tensor_list, axis=0)
        print("pcr_input_tensor: {}".format(self.pcr_input_tensor.shape))

    def save_pcr_results(self, data_list):
        os.makedirs('result', exist_ok=True)
        os.makedirs('result/type_0', exist_ok=True)
        os.makedirs('result/type_1', exist_ok=True)
        os.makedirs('result/type_2', exist_ok=True)
        f_0 = open(os.path.join('result', 'result_pcr_type_0.txt'), 'wt')
        f_1 = open(os.path.join('result', 'result_pcr_type_1.txt'), 'wt')
        f_2 = open(os.path.join('result', 'result_pcr_type_2.txt'), 'wt')
        f_list = [f_0, f_1, f_2]
        type_count = [0, 0, 0]
        # for filename, type, angle in zip(filenames, self.type_predict, self.angle_predict):
        for dict in data_list:
            for type in [0, 1, 2]:
                # type = dict["type"]
                angle = dict["angle"]
                # angle = 0
                filepath = dict["img_path"]
                # if type > 2:
                #     continue
                f = f_list[type]
                count = type_count[type]
                f.write('{} {} {} {} {}\n'.format(count, filepath, INPUT_WIDTH, INPUT_HEIGHT, int(round(angle, 0))))
                type_count[type] += 1
        for f in f_list:
            f.close()

    def run_pcr(self):
        self.pcr_model.init(self.pcr_input_tensor)
        self.type_list, self.angle_list = self.pcr_model.run(self.pcr_input_tensor)
        print("type_list: {}, angle_list: {}".format(self.type_list, self.angle_list))
        for idx, dict in enumerate(self.res_json_list):
            dict["type"] = self.type_list[idx]
            dict["angle"] = int(self.angle_list[idx])

        self.save_pcr_results(self.res_json_list)

    def run_psd(self):
        # clear session to load new model
        tf.keras.backend.clear_session()
        tf.reset_default_graph()
        # psd_test.evaluate(self.args.psd_test_weight_type0, "result/result_pcr_type_0.txt", "result/result_psd_type_0.txt", "result/type_0")
        psd_test.evaluate(self.args.psd_test_weight_type1, "result/result_pcr_type_1.txt", "result/result_psd_type_1.txt", "result/type_1", self.res_json_list)
        # psd_test.evaluate(self.args.psd_test_weight_type2, "result/result_pcr_type_2.txt", "result/result_psd_type_2.txt", "result/type_2")

    def save_json(self):
        # plot images
        for dict in self.res_json_list:
            img_wrapper = ImageWrapper(dict["img_path"], self.args.raw_image_h, self.args.raw_image_w)
            img_wrapper.cv_read_resize(w=dict["det_w"], h=dict["det_h"])
            output_path = "{}{}.png".format(self.output_dir, dict["idx"])
            img_wrapper.plot_preds(dict, output_path)

        # save json
        json_file_path = self.output_dir + "/results.json"
        with open(json_file_path, 'wt') as fd:
            for dict in self.res_json_list:
                json.dump(dict, fd, ensure_ascii=False)
                fd.write("\n")

def test():
    pcr_model = tf.saved_model.load("./saved_model/pcr_192_64_tf")
    print(pcr_model)

if __name__ == '__main__':
    args = parser.parse_args()
    evaluator = CarlaEvaluator(args)
    evaluator.make_pcr_inputs()
    evaluator.run_pcr()
    evaluator.run_psd()
    evaluator.save_json()
    # test()