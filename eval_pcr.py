import argparse
import os, sys
os.environ['CUDA_VISIBLE_DEVICES']="-1"
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# from parking_context_recognizer import train as pcr_train
from parking_slot_detector import test as psd_test
from parking_slot_detector import merge_three_type_result_files as merge_result
from parking_slot_detector.utils import eval_utils as eval_utils

from parking_context_recognizer.PcrWrapper import PCRWrapper
# from parking_slot_detector.PsdWrapper import PSDWrapper

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="evaluate carla ipm images")
parser.add_argument("--carla_image_path", type=str, default="pil_park/carla_town04/image/",
                    help="The path of the parking slot detection dataset")
parser.add_argument("--pcr_test_weight", type=str, default="weight/weight_pcr/trained/trained.ckpt",
                    help="The path of the trained weights of pcr.")
# parser.add_argument("--psd_test_weight_type0", type=str, default="weight/weight_psd/fine_tuned_type_0",
#                     help="The path of the trained weights of fine-tuned to parallel type.")
# parser.add_argument("--psd_test_weight_type1", type=str, default="weight/weight_psd/fine_tuned_type_1",
#                     help="The path of the trained weights of fine-tuned to perpendicular type.")
# parser.add_argument("--psd_test_weight_type2", type=str, default="weight/weight_psd/fine_tuned_type_2",
#                     help="The path of the trained weights of fine-tuned to diagonal type.")
# parser.add_argument("--threshold_score", type=float, default=0.8,
#                     help="Threshold of prediction which is determined TRUE")


class ImageWrapper(object):
    def __init__(self, image_path):
        self.image_path = image_path
        self.img_tensor = None
        self.img_np = None

    def get_cv2_np(self, h=768, w=256):
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (w, h))
        # the input of yolo_v3 should be in range 0~1
        img = img / 255.0
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        return tf.expand_dims(img_tensor, axis=0)

    def jpeg_to_tensor(self):
        # to be done
        return None

    def png_to_tensor(self):
        img_png = tf.gfile.FastGFile(self.image_path, 'rb').read()
        self.img_tensor = tf.image.decode_png(img_png, channels=3)
        with tf.Session("") as sess:
            self.img_np = sess.run(self.img_tensor)
        return self.img_tensor

    def to_numpy(self):
        return self.img_np

    # output the same tensor with "parking_context_recognizer/utils.py"
    def resize_tensor(self, h, w, standarization=True, expand_dim0=True):
        self.img_tensor = tf.image.resize(self.img_tensor, [h, w])
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
        # img_u8 = tf.image.convert_image_dtype(img_png, dtype=tf.uint8)
        # img_np = img_u8.eval()
        # plt.figure(1)
        # plt.imshow(img_np)
        # plt.draw()
        Image.fromarray(img_np).save(output_path)

class CarlaEvaluator(object):
    def __init__(self):
        self.args = parser.parse_args()
        self.output_dir = "result/"
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_files = os.listdir(self.args.carla_image_path)
        print("total image: {}".format(len(self.image_files)))
        self.pcr_model = PCRWrapper(self.args.pcr_test_weight)
        self.data_list = []
        # self.psd_model = PSDWrapper(self.args.psd_test_weight_type0)

    def make_pcr_inputs(self):
        counter = 0
        tensor_list = []
        for fname in self.image_files:
            counter += 1
            image_path = self.args.carla_image_path + fname
            img_wrapper = ImageWrapper(image_path)
            img_wrapper.png_to_tensor()
            img_tensor = img_wrapper.resize_tensor(h=192, w=64, standarization=True)

            # if counter == 1:
            #     self.pcr_init_tensor = img_tensor
            tensor_list.append(img_tensor)

            dict = {}
            dict["img_path"] = image_path
            # dict["img"] = img_wrapper.get_cv2_np()
            self.data_list.append(dict)

        self.pcr_input_tensor = tf.concat(tensor_list, axis=0)
        print("pcr_input_tensor: {}".format(self.pcr_input_tensor.shape))

    def get_angle(self, angle):
        angle_batch = []
        angle_batch.append(angle)
        angle_np = np.asarray(angle_batch)
        return tf.convert_to_tensor(angle_np, dtype=tf.float32)

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
                if type > 2:
                    continue
                f = f_list[type]
                count = type_count[type]
                f.write('{} {} 256 768 {}\n'.format(count, filepath, int(round(angle, 0))))
                type_count[type] += 1
        for f in f_list:
            f.close()

    def run_pcr(self):
        self.pcr_model.init(self.pcr_input_tensor)
        self.type_list, self.angle_list = self.pcr_model.run(self.pcr_input_tensor)
        print("type_list: {}, angle_list: {}".format(self.type_list, self.angle_list))
        for idx, dict in enumerate(self.data_list):
            dict["type"] = self.type_list[idx]
            dict["angle"] = self.angle_list[idx]

        self.save_pcr_results(self.data_list)
        # psd_test.evaluate(self.args.psd_test_weight_type0, "result/result_pcr_type_0.txt", "result/result_psd_type_0.txt")
        # psd_test.evaluate(self.args.psd_test_weight_type1, "result/result_pcr_type_1.txt", "result/result_psd_type_1.txt")
        # psd_test.evaluate(self.args.psd_test_weight_type2, "result/result_pcr_type_2.txt", "result/result_psd_type_2.txt")
        # self.psd_model.run(self.data_list, self.args.psd_test_weight_type0)


if __name__ == '__main__':
    args = parser.parse_args()
    evaluator = CarlaEvaluator()
    evaluator.make_pcr_inputs()
    evaluator.run_pcr()

    sys.exit(0)

    # pcr_train.evaluate(os.path.join(args.data_path, "test"), args.pcr_test_weight)

    psd_test.evaluate(args.psd_test_weight_type0, "result/result_pcr_type_0.txt", "result/result_psd_type_0.txt")
    psd_test.evaluate(args.psd_test_weight_type1, "result/result_pcr_type_1.txt", "result/result_psd_type_1.txt")
    psd_test.evaluate(args.psd_test_weight_type2, "result/result_pcr_type_2.txt", "result/result_psd_type_2.txt")

    result_files = ["result/result_psd_type_0.txt","result/result_psd_type_1.txt","result/result_psd_type_2.txt"]
    merge_result.merge_result(os.path.join(args.data_path, "test.txt"), result_files, "result/result.txt")
    eval_utils.eval_result_file(os.path.join(args.data_path, "test.txt"), "result/result.txt", args.threshold_score)
