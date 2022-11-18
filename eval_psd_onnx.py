import argparse
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES']="-1"
import tensorflow as tf2
from PIL import Image
import numpy as np
import cv2
from parking_slot_detector.config import *
from export.onnx_wrapper import OnnxWrapper

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
        img_png = tf2.io.gfile.GFile(self.image_path, 'rb').read()
        self.img_tensor = tf2.image.decode_png(img_png, channels=3)
        with tf2.compat.v1.Session("") as sess:
            self.img_np = sess.run(self.img_tensor)
        self.img_tensor = tf2.image.resize(self.img_tensor, [self.raw_h, self.raw_w])
        print("img_tensor: {}".format(self.img_tensor.shape))
        return self.img_tensor

    def to_numpy(self):
        return self.img_np

    # output the same tensor with "parking_context_recognizer/utils.py"
    def resize_tensor(self, h=INPUT_HEIGHT, w=INPUT_WIDTH, standarization=True, expand_dim0=True):
        self.img_tensor = tf2.image.resize(self.img_tensor, [h, w])
        # print("img_tensor: {}".format(self.img_tensor.shape))
        if standarization:
            self.img_tensor = tf2.image.per_image_standardization(self.img_tensor)
        if expand_dim0:
            self.img_tensor = tf2.expand_dims(self.img_tensor, axis=0)
        return self.img_tensor

    # output the same tensor with "parking_context_recognizer/utils.py"
    def crop_resize_tensor(self, out_h=INPUT_HEIGHT, out_w=INPUT_WIDTH, standarization=True, expand_dim0=True):
        img_size = self.img_tensor.shape
        h = int(img_size[0])
        w = int(h/3.0)
        h = int(3*w)
        print("crop size: h: {}, w: {}".format(h, w))
        self.img_tensor = self.img_tensor[0:h, 0:w, :]
        self.img_tensor = tf2.image.resize(self.img_tensor, [out_h, out_w])
        if standarization:
            self.img_tensor = tf2.image.per_image_standardization(self.img_tensor)
        if expand_dim0:
            self.img_tensor = tf2.expand_dims(self.img_tensor, axis=0)
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
        self.json_file_path = self.output_dir + "/results.json"
        self.res_json_list = []
        self.onnx_psd = OnnxWrapper()
        self.onnx_psd.load_onnx_model("./export/psd.nms.onnx")

    def make_psd_inputs(self):
        counter = 0
        lines = []
        with open(self.json_file_path) as fd:
            lines = fd.readlines()
        for line in lines:
            obj = json.loads(line)
            self.res_json_list.append(obj)
            counter += 1
        print("total dataset: {}, json list: {}".format(counter, self.res_json_list))

    def post_process(self, idx, result):
        boxes = result[0]
        labels = result[1]
        quads = result[2]
        scores = result[3]
        print("boxes_: {}, scores_: {}, labels_: {}, quads_: {}".format(boxes.shape, scores.shape, labels.shape, quads.shape))
        pred_content = []
        for i in range(len(labels)):
            x_min, y_min, x_max, y_max = boxes[i]
            score = scores[i]
            label = labels[i]
            quad = quads[i]
            pred_content.append([idx, x_min, y_min, x_max, y_max, score, label, quad])

        pk_list = []
        self.res_json_list[idx]["det_h"] = int(INPUT_HEIGHT)
        self.res_json_list[idx]["det_w"] = int(INPUT_WIDTH)
        # self.res_json_list[idx]["type"] = 1
        if len(pred_content) > 0:
            for img_id, x_min, y_min, x_max, y_max, score, label, quad in pred_content:
                # print("img_id:{}, x_min:{}, y_min:{}, x_max:{}, y_max:{}, score:{}, label:{}, quad:{}".format(img_id, x_min, y_min, x_max, y_max, score, label, quad))
                bbx = [x_min, y_min, x_max, y_max]
                # bbx_list.append(bbx)
                pk_dict = {}
                pk_dict["bbx"] = [int(val) for val in bbx]
                pk_dict["quad"] = [int(val) for val in quad]
                pk_dict["score"] = float(score)
                pk_dict["label"] = int(label)
                pk_list.append(pk_dict)
        self.res_json_list[idx]["pred"] = pk_list

    def run_psd(self):
        for idx, dict in enumerate(self.res_json_list):
            img_angle = dict["angle"]
            angle_np = np.array([img_angle], dtype=np.float32)
            img_path = dict["img_path"]

            img_np = cv2.imread(img_path)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_np = cv2.resize(img_np, (INPUT_WIDTH, INPUT_HEIGHT))
            # the input of yolo_v3 should be in range 0~1
            img_np = img_np / 255.0
            img_np = np.expand_dims(img_np, axis=0)
            # print("img_np: {}".format(img_np.shape))
            try:
                result = self.onnx_psd.run_onnx_model(self.onnx_psd.ortss, [angle_np, img_np])
            except Exception as e:
                print("{} exception: {}".format(idx, e))
                continue
            self.post_process(idx, result)

    def save_images(self):
        # plot images
        for dict in self.res_json_list:
            img_wrapper = ImageWrapper(dict["img_path"], self.args.raw_image_h, self.args.raw_image_w)
            img_wrapper.cv_read_resize(w=dict["det_w"], h=dict["det_h"])
            output_path = "{}{}.png".format(self.output_dir, dict["idx"])
            img_wrapper.plot_preds(dict, output_path)

    def save_json(self):
        # save json
        json_file_path = self.output_dir + "/results.json"
        with open(json_file_path, 'wt') as fd:
            for dict in self.res_json_list:
                json.dump(dict, fd, ensure_ascii=False)
                fd.write("\n")

def test():
    pcr_model = tf2.saved_model.load("./saved_model/pcr_192_64_tf")
    print("pcr_model: {}".format(pcr_model.signatures))
    infer = pcr_model.signatures["serving_default"]
    print("pcr_model: {}".format(infer.graph))

if __name__ == '__main__':
    tf2.compat.v1.disable_eager_execution()
    args = parser.parse_args()
    evaluator = CarlaEvaluator(args)
    evaluator.make_psd_inputs()
    evaluator.run_psd()
    evaluator.save_json()
    evaluator.save_images()