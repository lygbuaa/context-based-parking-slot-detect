# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
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

class ImageWrapper(object):
    def __init__(self, image_path):
        self.image_path = image_path
        self.read()

    def read(self, h=INPUT_HEIGHT, w=INPUT_WIDTH):
        self.img_np = cv2.imread(self.image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        self.img_np = cv2.resize(self.img_np, (w, h))

    def plot_bbx(self, bbx_list):
        #bbx: [x1, y1, x2, y2]
        for bbx in bbx_list:
            start_point = (int(bbx[0]), int(bbx[1]))
            end_point = (int(bbx[2]), int(bbx[3]))
            color = (255, 255, 0)
            self.img_np = cv2.rectangle(self.img_np, start_point, end_point, color=color, thickness=2)
            # print("plot bbx: {}".format(bbx))
        return self.img_np

    def save(self, output_path):
        Image.fromarray(self.img_np).save(output_path)


def evaluate(weight_path, eval_file, result_file, save_path, json_list):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print(weight_path, eval_file, result_file, save_path)
    os.makedirs(save_path, exist_ok=True)

    # args params
    anchors = parse_anchors('data/anchors.txt')
    classes = read_class_names("data/data.names")
    class_num = len(classes)
    img_cnt = len(open(eval_file, 'r').readlines())
    
    # read filename from eval file
    # print(os.path.basename(os.path.split(weight_path)[-2]))
    # print(os.path.basename(weight_path))
    
    filenames = {}
    eval_f= open(eval_file, 'rt')
    for line in eval_f.readlines():
        img_idx, pic_path, boxes, labels, _, _, quads, angle = parse_line(line)
        # filename = os.path.basename(pic_path)
        filenames[img_idx] = pic_path
    
    # setting placeholders
    is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
    handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
    # pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_quads_flag = tf.placeholder(tf.float32, [1, None, None])
    gpu_nms_op = gpu_nms(pred_quads_flag, pred_scores_flag, class_num, NMS_TOPK, THRESHOLD_OBJ, THRESHOLD_NMS, apply_rotate=True)
    
    ##################
    # tf.data pipeline
    ##################
    # print(eval_file)
    val_dataset = tf.data.TextLineDataset(eval_file)
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.map(
        lambda x: tf.py_func(get_batch_data, [x, class_num, [INPUT_WIDTH, INPUT_HEIGHT], anchors], [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64]),
        num_parallel_calls=NUM_THREADS
    )
    val_dataset.prefetch(PREFETECH_BUFFER)
    iterator = val_dataset.make_one_shot_iterator()
    
    image_ids, image, y_true_13, y_true_26, y_true_52, image_angle = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]
    
    image_ids.set_shape([None])
    image.set_shape([1, INPUT_HEIGHT, INPUT_WIDTH, 3])
    
    for y in y_true:
        y.set_shape([None, None, None, None, None])
    image_angle.set_shape([None])
    image_angle = tf.dtypes.cast(image_angle, tf.float32)
    
    ##################
    # Model definition
    ##################
    yolo_model = yolov3(class_num, anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(image, is_training=is_training)
    loss = yolo_model.compute_loss(pred_feature_maps, y_true, image_angle)
    y_pred = yolo_model.predict(pred_feature_maps, image_angle)
    
    saver_to_restore = tf.train.Saver()
    weight_file = tf.train.latest_checkpoint(weight_path)
    print("weight_file: {}".format(weight_file))
    
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer()])
        saver_to_restore.restore(sess, weight_file)
        today = time.localtime()
        time_part = '{:02d}{:02d}{:02d}_{:02d}{:02d}'.format(today.tm_year, today.tm_mon, today.tm_mday, today.tm_hour,
                                                             today.tm_min)

        print('\n----------- start to eval -----------\n image: {}, img_cnt: {}'.format(image.shape, img_cnt))
    
        val_loss_total, val_loss_conf, val_loss_class, val_loss_quad = \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        f = open(result_file, 'wt')
        val_preds = []
        # gt_dict = parse_gt_quadrangle(eval_file, [INPUT_WIDTH, INPUT_HEIGHT], letterbox_resize)
    
        # start_time = time.time()
    
        for j in trange(img_cnt):
            __image_ids, __y_pred, __loss = sess.run([image_ids, y_pred, loss], feed_dict={is_training: False})
            print("confs: {}, probs: {}, quads: {}".format(__y_pred[0].shape, __y_pred[1].shape, __y_pred[2].shape))
            pred_content = get_preds_gpu(sess, gpu_nms_op, pred_quads_flag, pred_scores_flag, __image_ids, __y_pred)
            # print("pred_content: {}".format(pred_content))   
            if not FLAG_CALC_ONLY_TIME:
                pic_path = filenames[j]
                f.write('%d %s ' % (j, pic_path))
                print("evaluate {}-{}, json line: {}".format(j, pic_path, json_list[j]))

                # img_wrapper = ImageWrapper(pic_path)
                # bbx_list = []
                json_list[j]["det_h"] = int(INPUT_HEIGHT)
                json_list[j]["det_w"] = int(INPUT_WIDTH)

                pk_list = []
                if len(pred_content) > 0:
                    for img_id, x_min, y_min, x_max, y_max, score, label, quad in pred_content:
                        f.write("%d %.8f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "%(label, score, quad[0], quad[1], quad[2], quad[3], quad[4], quad[5], quad[6], quad[7]))
                        print("img_id:{}, x_min:{}, y_min:{}, x_max:{}, y_max:{}, score:{}, label:{}, quad:{}".format(img_id, x_min, y_min, x_max, y_max, score, label, quad))
                        bbx = [x_min, y_min, x_max, y_max]
                        # bbx_list.append(bbx)
                        pk_dict = {}
                        pk_dict["bbx"] = [int(val) for val in bbx]
                        pk_dict["quad"] = [int(val) for val in quad]
                        pk_dict["score"] = float(score)
                        pk_dict["label"] = int(label)
                        pk_list.append(pk_dict)
                json_list[j]["pred"] = pk_list
    
                f.write('\n')
                # img_wrapper.plot_bbx(bbx_list)
                # output_path = save_path + "/{}.jpg".format(j)
                # img_wrapper.save(output_path)
    
                val_preds.extend(pred_content)
                val_loss_total.update(__loss[0])
                val_loss_conf.update(__loss[1])
                val_loss_class.update(__loss[2])
                val_loss_quad.update(__loss[3])
    
        # end_time = time.time()
        # print("Inference time {} sec".format(end_time - start_time))
    
        f.close()
    
    #print params
    # param_stats = tf.profiler.profile(graph=sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    # flops_stats = tf.profiler.profile(graph=sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    # print("parameters: {}, flops: {}".format(param_stats.total_parameters, flops_stats.total_float_ops))

    sess.close()
    tf.reset_default_graph()
        # if not FLAG_CALC_ONLY_TIME:
        #     eval_result_file(eval_file, result_file)
