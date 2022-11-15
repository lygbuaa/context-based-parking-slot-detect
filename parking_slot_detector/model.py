# coding=utf-8
# for better understanding about yolov3 architecture, refer to this website (in Chinese):
# https://blog.csdn.net/leviopku/article/details/82660381

from __future__ import division, print_function

import tensorflow as tf2
tf=tf2.compat.v1
import math
# slim = tf.contrib.slim
# tf2.x deprecated tf1.contrib.slim, pip install --upgrade tf_slim
import tf_slim as slim

from parking_slot_detector.utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer
from parking_slot_detector.config import *

class yolov3(tf.Module):
    def __init__(self, class_num, anchors, max_boxes=7, score_thresh=0.5, nms_thresh=0.1):
        super(yolov3, self).__init__()
        self.class_num = class_num
        self.anchors = anchors
        self.max_boxes = max_boxes
        self.score_thresh = score_thresh # set score_thresh=0.001, ensure filter_score not empty
        self.nms_thresh = nms_thresh
        self.batch_norm_decay = 0.999
        self.use_label_smooth = False
        self.use_focal_loss = False
        self.weight_decay = 5e-4
        self.use_static_shape = True

    # @tf2.function(input_signature=[tf.TensorSpec(shape=[1, 640, 640, 3], dtype=tf.float32)])
    # def __call__(self, inputs):
    #     return self.forward(inputs=inputs, is_training=False, reuse=False)

    # @tf2.function(input_signature=[tf.TensorSpec([1, 640, 640, 3], tf.float32), tf.TensorSpec([1], tf.float32)])
    # def __call__(self, image, angle):
    #     pred_feature_maps = self.forward(inputs=image, is_training=False, reuse=False)
    #     return self.predict(pred_feature_maps, angle)

    # with postprocess
    def __call__(self, image, angle):
        pred_feature_maps = self.forward(inputs=image, is_training=False, reuse=False)
        confs, probs, quads = self.predict(pred_feature_maps, angle)
        return self.postprocess(confs, probs, quads)

    def postprocess(self, confs, probs, quads):
        pred_confs = confs[0:1]
        pred_probs = probs[0:1]
        pred_quads = quads[0:1]
        pred_scores = pred_confs*pred_probs
        return self.nms(quads=pred_quads, scores=pred_scores, max_boxes=self.max_boxes, score_thresh=self.score_thresh, nms_thresh=self.nms_thresh)

    def nms(self, quads, scores, max_boxes=7, score_thresh=0.5, nms_thresh=0.1):
        """
        Perform NMS on GPU using TensorFlow.

        params:
            boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
            scores: tensor of shape [1, 10647, self.class_num], score=conf*prob
            self.class_num: total number of classes
            max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
            score_thresh: if [ highest class probability score < score_threshold]
                            then get rid of the corresponding box
            nms_thresh: real value, "intersection over union" threshold used for NMS filtering
        """

        boxes_list, label_list, score_list, quads_list = [], [], [], []
        max_boxes = tf.constant(max_boxes, dtype='int32')

        quads = tf.reshape(quads, [-1, 8])
        # boxes = tf.stack([tf.minimum(quads[..., 0],quads[..., 2]), tf.minimum(quads[..., 1],quads[..., 7]), tf.maximum(quads[..., 4],quads[..., 6]), tf.maximum(quads[..., 3],quads[..., 5])], axis=-1)
        boxes = tf.stack([tf.minimum(tf.minimum(quads[..., 0],quads[..., 2]), tf.minimum(quads[..., 4],quads[..., 6])),
                        tf.minimum(tf.minimum(quads[..., 1],quads[..., 3]), tf.minimum(quads[..., 5],quads[..., 7])),
                        tf.maximum(tf.maximum(quads[..., 0],quads[..., 2]), tf.maximum(quads[..., 4],quads[..., 6])),
                        tf.maximum(tf.maximum(quads[..., 1],quads[..., 3]), tf.maximum(quads[..., 5],quads[..., 7]))],
                        axis=-1)
        # boxes = tf.gather(quads, indices = [0, 1, 4, 5], axis=-1)
        # since we do nms for single image, then reshape it
        boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
        score = tf.reshape(scores, [-1, self.class_num])
        labels = tf.argmax(score, axis=1)
        score = tf.reduce_max(score, axis=1)
        # score = tf.reduce_max(score[:,0], score[:,1])
        #
        # print("boxes size", tf.size(boxes))
        # print("quads size", tf.size(quads))

        # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
        mask = tf.greater_equal(score, tf.constant(score_thresh))
        # Step 2: Do non_max_suppression for each class
        # for i in range(self.class_num):
            # Step 3: Apply the mask to scores, boxes and pick them out
        filter_labels = tf.boolean_mask(labels, mask)
        filter_boxes = tf.boolean_mask(boxes, mask)
        filter_score = tf.boolean_mask(score, mask)
        filter_quads = tf.boolean_mask(quads, mask)

        # INVALID_ARGUMENT : Non-zero status code returned while running If node. Name:'yolov3_1/cond'
        # cond_flag = tf.shape(filter_score)[0]

        # set small score_thresh, ensure cond_flag None-Zero, always apply rotate
        # nms_indices = tf.cond(tf.greater(tf.shape(filter_score)[0], 0),
        #                     lambda:self.rot_nms(filter_score, filter_quads, max_boxes, nms_thresh),
        #                     lambda:tf.image.non_max_suppression(boxes=filter_boxes,
        #                                             scores=filter_score,
        #                                             max_output_size=max_boxes,
        #                                             iou_threshold=nms_thresh, name='nms_indices')
        #                     )

        # don't apply rotate, Non-zero status code returned while running Transpose node. Name:'Transpose__1294' 
        # Status Message: perm: [ 0 2 3 1 ] does not align with rank of input data: 5
        # nms_indices = self.rot_nms(filter_score, filter_quads, max_boxes, nms_thresh)
        # using normal nms
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                    scores=filter_score,
                                                    max_output_size=max_boxes,
                                                    iou_threshold=nms_thresh, name='nms_indices')

        label_list.append(tf.gather(filter_labels, nms_indices))
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))
        quads_list.append(tf.gather(filter_quads, nms_indices))

        boxes = tf.concat(boxes_list, axis=0)
        score = tf.concat(score_list, axis=0)
        label = tf.concat(label_list, axis=0)
        quad = tf.concat(quads_list, axis=0)

        return boxes, score, label, quad

    def rot_nms(self, filter_score, filter_quads, max_boxes, nms_thresh):
        max_score_idx = tf.argmax(filter_score)
        best_quad = filter_quads[max_score_idx]
        y_diff = best_quad[..., 7] + best_quad[..., 5] - best_quad[..., 3] - best_quad[..., 1]
        x_diff = best_quad[..., 6] + best_quad[..., 4] - best_quad[..., 2] - best_quad[..., 0]
        angle = tf.atan2(y_diff, x_diff)
        temp_quads = tf.reshape(filter_quads, [-1, 4, 2])

        rot_x = tf.stack([tf.cos(angle), -tf.sin(angle)], -1)
        rot_y = tf.stack([tf.sin(angle), tf.cos(angle)], -1)
        rot_mat = tf.stack([rot_x, rot_y], -2)
        # rot_mat_repeat = tf.stack([rot_mat, rot_mat, rot_mat, rot_mat], -2)
        rot_quads = tf.einsum('jk,lij->lik', rot_mat, temp_quads)
        rot_quads = tf.reshape(rot_quads, [-1, 8])
        rot_boxes = tf.stack(
            [tf.minimum(tf.minimum(rot_quads[..., 0], rot_quads[..., 2]), tf.minimum(rot_quads[..., 4], rot_quads[..., 6])),
            tf.minimum(tf.minimum(rot_quads[..., 1], rot_quads[..., 3]), tf.minimum(rot_quads[..., 5], rot_quads[..., 7])),
            tf.maximum(tf.maximum(rot_quads[..., 0], rot_quads[..., 2]), tf.maximum(rot_quads[..., 4], rot_quads[..., 6])),
            tf.maximum(tf.maximum(rot_quads[..., 1], rot_quads[..., 3]),
                        tf.maximum(rot_quads[..., 5], rot_quads[..., 7]))],
            axis=-1)

        nms_indices = tf.image.non_max_suppression(boxes=rot_boxes,
                                                scores=filter_score,
                                                max_output_size=max_boxes,
                                                iou_threshold=nms_thresh, name='nms_indices')
        return nms_indices

    def forward(self, inputs, is_training=False, reuse=False):
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d], 
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (13 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, route_2.get_shape().as_list() if self.use_static_shape else tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (13 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, route_1.get_shape().as_list() if self.use_static_shape else tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (13 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors, angle):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = feature_map.get_shape().as_list()[1:3] if self.use_static_shape else tf.shape(feature_map)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 13 + self.class_num])

        # split the feature_map along the last dimension
        box_centers, box_sizes, conf_logits, prob_logits, quad_points = tf.split(feature_map, [2, 2, 1, self.class_num, 8], axis=-1)
        quad_points = tf.nn.tanh(quad_points)
        quad1 = quad_points[...,0:2] * anchors
        quad2 = quad_points[...,2:4] * anchors
        quad3 = quad_points[...,4:6] * anchors
        quad4 = quad_points[...,6:8] * anchors

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        rot_mat = tf.stack([tf.stack([tf.cos(-angle*math.pi/180), -tf.sin(-angle*math.pi/180)], -1), tf.stack([tf.sin(-angle*math.pi/180), tf.cos(-angle*math.pi/180)], -1)], -2)
        quad1 = tf.einsum('njk,nwhij->nwhik', rot_mat, quad1)
        quad2 = tf.einsum('njk,nwhij->nwhik', rot_mat, quad2)
        quad3 = tf.einsum('njk,nwhij->nwhik', rot_mat, quad3)
        quad4 = tf.einsum('njk,nwhij->nwhik', rot_mat, quad4)

        quad1 = x_y_offset * ratio[::-1] + quad1
        quad2 = x_y_offset * ratio[::-1] + quad2
        quad3 = x_y_offset * ratio[::-1] + quad3
        quad4 = x_y_offset * ratio[::-1] + quad4
        quads = tf.concat([quad1, quad2, quad3, quad4], axis=-1)

        return x_y_offset, conf_logits, prob_logits, quads


    def predict(self, feature_maps, angle):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors, angle) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, conf_logits, prob_logits, quads = result
            grid_size = x_y_offset.get_shape().as_list()[:2] if self.use_static_shape else tf.shape(x_y_offset)[:2]
            quads = tf.reshape(quads, [-1, grid_size[0] * grid_size[1] * 3, 8])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])

            return conf_logits, prob_logits, quads

        boxes_list, confs_list, probs_list, quads_list = [], [], [], []
        for result in reorg_results:
            conf_logits, prob_logits, quads = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            # boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)
            quads_list.append(quads)
        
        # collect results on three scales
        confs = tf.concat(confs_list, axis=1)
        probs = tf.concat(probs_list, axis=1)
        quads = tf.concat(quads_list, axis=1)

        return confs, probs, quads
    
    def loss_layer(self, feature_map_i, y_true, anchors, angle):
        '''
        calc loss function from a certain scale
        input:
            feature_map_i: feature maps of a certain scale. shape: [N, 13, 13, 3*(5 + num_class)] etc.
            y_true: y_ture from a certain scale. shape: [N, 13, 13, 3, 5 + num_class + 1] etc.
            anchors: shape [9, 2]
        '''
        
        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_conf_logits, pred_prob_logits, pred_quad_points = self.reorg_layer(feature_map_i, anchors, angle)

        ###########
        # get mask
        ###########
        object_mask = y_true[..., 4:5]

        # the calculation of ignore mask if referred from
        # https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c#L179
        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.cast(N, tf.int32))
        def loop_body(idx, ignore_mask):
            valid_true_quads = tf.boolean_mask(y_true[idx, ..., 5 + self.class_num:13 + self.class_num], tf.cast(object_mask[idx, ..., 0], 'bool'))
            iou = self.quad_iou(pred_quad_points[idx], valid_true_quads)
            best_iou = tf.reduce_max(iou, axis=-1)
            ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
            return idx + 1, ignore_mask
        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        pred_quad1 = pred_quad_points[...,0:2]
        pred_quad2 = pred_quad_points[...,2:4]
        pred_quad3 = pred_quad_points[...,4:6]
        pred_quad4 = pred_quad_points[...,6:8]

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        true_p1 = y_true[..., 5 + self.class_num:7 + self.class_num] / ratio[::-1] - x_y_offset
        true_p2 = y_true[..., 7 + self.class_num:9 + self.class_num] / ratio[::-1] - x_y_offset
        true_p3 = y_true[..., 9 + self.class_num:11 + self.class_num] / ratio[::-1] - x_y_offset
        true_p4 = y_true[..., 11 + self.class_num:13 + self.class_num] / ratio[::-1] - x_y_offset

        pred_p1 = pred_quad1 / ratio[::-1] - x_y_offset
        pred_p2 = pred_quad2 / ratio[::-1] - x_y_offset
        pred_p3 = pred_quad3 / ratio[::-1] - x_y_offset
        pred_p4 = pred_quad4 / ratio[::-1] - x_y_offset

        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############
        # mix_up weight
        mix_w = y_true[..., -1:]
        quad_loss = tf.reduce_sum((tf.abs(true_p1 - pred_p1) + tf.abs(true_p2 - pred_p2) + tf.abs(true_p3 - pred_p3) + tf.abs(true_p4 - pred_p4))* object_mask * box_loss_scale * mix_w) / N / 4

        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss = conf_loss_pos + conf_loss_neg
        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # whether to use label smooth
        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:5+ self.class_num] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 5:5+ self.class_num]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return conf_loss, class_loss, quad_loss
    

    def quad_iou(self, pred_quad_points, valid_true_quads):
        '''
        param:
            pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
            valid_true: [V, 4]
        '''

        # [13, 13, 3, 2]
        pred_box_xy = (pred_quad_points[..., 0:2] + pred_quad_points[..., 2:4] + pred_quad_points[..., 4:6] + pred_quad_points[..., 6:8])/4
        pred_box_w = tf.maximum(tf.maximum(pred_quad_points[..., 0], pred_quad_points[..., 2]), tf.maximum(pred_quad_points[..., 4], pred_quad_points[..., 6]))\
                     -tf.minimum(tf.minimum(pred_quad_points[..., 0], pred_quad_points[..., 2]), tf.minimum(pred_quad_points[..., 4], pred_quad_points[..., 6]))
        pred_box_h = tf.maximum(tf.maximum(pred_quad_points[..., 1], pred_quad_points[..., 3]), tf.maximum(pred_quad_points[..., 5], pred_quad_points[..., 7]))\
                     -tf.minimum(tf.minimum(pred_quad_points[..., 1], pred_quad_points[..., 3]), tf.minimum(pred_quad_points[..., 5], pred_quad_points[..., 7]))
        # pred_box_w = tf.maximum(pred_quad_points[..., 4], pred_quad_points[..., 6]) - tf.minimum(pred_quad_points[..., 0], pred_quad_points[..., 2])
        # pred_box_h = tf.maximum(pred_quad_points[..., 3], pred_quad_points[..., 5]) - tf.minimum(pred_quad_points[..., 1], pred_quad_points[..., 7])
        pred_box_wh = tf.stack([pred_box_w, pred_box_h], -1)
        # print("pred_box_xy ", tf.shape(pred_box_xy))
        # print("pred_box_w ", tf.shape(pred_box_w))
        # print("pred_box_h ", tf.shape(pred_box_h))
        # print("pred_box_wh ", tf.shape(pred_box_wh))
        # pred_box_wh = pred_quad_points[..., 4:6] - pred_quad_points[..., 0:2]

        # shape: [13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # [V, 2]
        true_box_xy = (valid_true_quads[:, 0:2] + valid_true_quads[:, 2:4] + valid_true_quads[:, 4:6] + valid_true_quads[:, 6:8]) / 4
        true_box_w = tf.maximum(tf.maximum(valid_true_quads[:, 0], valid_true_quads[:, 2]),
                                tf.maximum(valid_true_quads[:, 4], valid_true_quads[:, 6])) \
                     - tf.minimum(tf.minimum(valid_true_quads[:, 0], valid_true_quads[:, 2]),
                                  tf.minimum(valid_true_quads[:, 4], valid_true_quads[:, 6]))
        true_box_h = tf.maximum(tf.maximum(valid_true_quads[:, 1], valid_true_quads[:, 3]),
                                tf.maximum(valid_true_quads[:, 5], valid_true_quads[:, 7])) \
                     - tf.minimum(tf.minimum(valid_true_quads[:, 1], valid_true_quads[:, 3]),
                                  tf.minimum(valid_true_quads[:, 5], valid_true_quads[:, 7]))
        true_box_wh = tf.stack([true_box_w, true_box_h], -1)
        # 
        # 
        # true_box_xy = valid_true_boxes[:, 0:2]
        # true_box_wh = valid_true_boxes[:, 2:4]

        # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = tf.expand_dims(true_box_area, axis=0)

        # [13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou


    def compute_loss(self, y_pred, y_true, angle):
        '''
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        '''
        loss_conf, loss_class, loss_quad = 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i], angle)
            # loss_xy += result[0]
            # loss_wh += result[1]
            loss_conf += result[0]
            loss_class += result[1]
            loss_quad += result[2]
        total_loss = loss_conf + loss_class + loss_quad
        return [total_loss, loss_conf, loss_class, loss_quad]
