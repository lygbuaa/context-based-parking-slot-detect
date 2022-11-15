# coding: utf-8
from __future__ import division, print_function
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES']="-1"
import tensorflow as tf2
# for v1 compatible
tf1 = tf2.compat.v1
# switch to TF1 mode
tf2.compat.v1.disable_eager_execution()
from tqdm import trange
import time, os, cv2
import numpy as np
from PIL import Image
from parking_slot_detector.config import *
from parking_slot_detector.utils.data_utils import get_batch_data, parse_line
from parking_slot_detector.utils.misc_utils import parse_anchors, read_class_names, AverageMeter
from parking_slot_detector.utils.eval_utils import evaluate_on_gpu, get_preds_gpu, parse_gt_rec, parse_gt_quadrangle, calc_park_score, park_eval, eval_result_file
from parking_slot_detector.utils.nms_utils import gpu_nms
from parking_slot_detector.utils.plot_utils import plot_one_box, plot_one_quad
from parking_slot_detector.model import yolov3
FLAG_CALC_ONLY_TIME = False

class PsdWrapper(object):
    def __init__(self, weight_path):
        # super(PsdWrapper, self).__init__()
        self.weight_path = weight_path
        self.ckpt_path = tf1.train.get_checkpoint_state(self.weight_path)
        # self.inspect_ckpt()
        self.anchors = parse_anchors('data/anchors.txt')
        self.classes = read_class_names("data/data.names")
        self.class_num = len(self.classes)

    # print key-list in checkpoint
    def inspect_ckpt(self):
        weight_file = tf1.train.latest_checkpoint(self.weight_path)
        inspect_list = tf1.train.list_variables(weight_file)
        print("{} inspect list: \n{}".format(weight_file, inspect_list))

    # pass-in image path via json_list
    def run(self, json_list):
        # init model session
        # pred_scores_flag = tf1.placeholder(tf1.float32, [1, None, None], name="pred_scores_flag")
        # pred_quads_flag = tf1.placeholder(tf1.float32, [1, None, None], name="pred_quads_flag")
        # gpu_nms_op = gpu_nms(pred_quads_flag, pred_scores_flag, self.class_num, NMS_TOPK, THRESHOLD_OBJ, THRESHOLD_NMS, apply_rotate=True)
        img_tf1 = tf1.placeholder(tf1.float32, [1, INPUT_HEIGHT, INPUT_WIDTH, 3], name="img_tf1")
        angle_tf1 = tf1.placeholder(tf1.float32, [1], name="angle_tf1")

        yolo_model = yolov3(self.class_num, self.anchors, NMS_TOPK, THRESHOLD_OBJ, THRESHOLD_NMS)
        with tf1.variable_scope('yolov3'):
            boxes, scores, labels, quads = yolo_model(img_tf1, angle_tf1)
        saver_to_restore = tf1.train.Saver()
        weight_file = tf1.train.latest_checkpoint(self.weight_path)
        print("load weight_file: {}".format(weight_file))
        # inspect model graph
        # print("yolov3 model graph: {}".format(tf1.global_variables()))

        with tf1.Session() as sess:
            sess.run(tf1.global_variables_initializer())
            saver_to_restore.restore(sess, weight_file)

            for idx, dict in enumerate(json_list):
                img_angle = dict["angle"]
                img_path = dict["img_path"]

                img_np = cv2.imread(img_path)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype(np.float32)
                img_np = cv2.resize(img_np, (INPUT_WIDTH, INPUT_HEIGHT))
                # the input of yolo_v3 should be in range 0~1
                img_np = img_np / 255.0
                img_np = np.expand_dims(img_np, axis=0)

                boxes_, scores_, labels_, quads_ = sess.run([boxes, scores, labels, quads], feed_dict={angle_tf1: [img_angle], img_tf1: img_np})

                print("boxes_: {}, scores_: {}, labels_: {}, quads_: {}".format(boxes_.shape, scores_.shape, labels_.shape, quads_.shape))
                # pred_content = get_preds_gpu(sess, gpu_nms_op, pred_quads_flag, pred_scores_flag, [idx], y_pred_)
                # print("pred_content: {}".format(len(pred_content)))

                # make pred_content from eval_utils.py::get_preds_gpu()
                pred_content = []
                for i in range(len(labels_)):
                    x_min, y_min, x_max, y_max = boxes_[i]
                    score = scores_[i]
                    label = labels_[i]
                    quad = quads_[i]
                    pred_content.append([idx, x_min, y_min, x_max, y_max, score, label, quad])

                pk_list = []
                json_list[idx]["det_h"] = int(INPUT_HEIGHT)
                json_list[idx]["det_w"] = int(INPUT_WIDTH)
                # json_list[idx]["type"] = 1
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
                json_list[idx]["pred"] = pk_list
                break

            # save model
            # signature: feature_map_1_: (1, 20, 20, 45), feature_map_2_: (1, 40, 40, 45), feature_map_3_: (1, 80, 80, 45)
            # signature: confs: (1, 25200, 1), probs: (1, 25200, 2), quads: (1, 25200, 8)
            # tf1.saved_model.simple_save(sess, "saved_model/psd_640_640_tf", inputs={"image": img_tf1, "angle": angle_tf1}, outputs={"confs": confs, "probs": probs, "quads": quads})
            # signature: boxes_: (16, 4), scores_: (16,), labels_: (16,), quads_: (16, 8)
            tf1.saved_model.simple_save(sess, "saved_model/psd_640_640_tf", inputs={"image": img_tf1, "angle": angle_tf1}, outputs={"boxes": boxes, "scores": scores, "labels": labels, "quads": quads})

    def __call__(self, img):
        pass

