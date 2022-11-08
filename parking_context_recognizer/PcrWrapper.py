import time
import warnings
import glob
import itertools
import os
import numpy as np
import tensorflow as tf2
from tensorflow.keras import backend as K

from parking_context_recognizer.utils import *
from parking_context_recognizer.models import model_mobilenetv2 as pcr_model
from parking_context_recognizer.config import *

class PCRWrapper(object):
    def __init__(self):
        # self.weight_file = weight_file
        pass

    def __del__(self):
        tf2.keras.backend.clear_session()

    def destroy(self):
        tf2.keras.backend.clear_session()

    def init_from_saved_model(self, saved_path):
        self.loaded = tf2.saved_model.load(saved_path)
        print("load model varies: {}".format(self.loaded.trainable_variables))

    def init_from_ckpt(self, weight_file, img_tensor):
        model_input = tf2.keras.layers.Input(tensor=img_tensor)
        self.model = pcr_model(model_input)
        self.model.load_weights(weight_file)
        # self.sessions = K.get_session()
        # self.graphs = tf2.get_default_graph()
        
    def run_saved_model(self, img_tensor):
        type_predict, angle_predict = self.loaded(img_tensor)
        print("pcr_model output, type_predict: {}, \nangle_predict: {}".format(type_predict, angle_predict))
        init = tf.compat.v1.global_variables_initializer()
        with tf2.compat.v1.Session("") as sess:
            sess.run(init)       
            type_predict = np.argmax(type_predict.eval(), axis=1)
            type_predict = np.ndarray.tolist(type_predict)
            # cast angle into degree
            angle_predict = angle_predict * 180. - 90.
            # itertools not avaliable on tf2
            angle_predict = angle_predict[0].eval().tolist()
            print("post-process output, type_predict: {}, \nangle_predict: {}".format(type_predict, angle_predict))
            return type_predict, angle_predict

    def run_ckpt(self, img_tensor):
        type_predict, angle_predict = self.model.predict(img_tensor, steps=1)
        # out_type = tf2.convert_to_tensor(type_predict, dtype=tf2.float32)
        # out_angle = tf2.convert_to_tensor(angle_predict, dtype=tf2.float32)
        print("pcr_model input, img_tensor: {}".format(img_tensor.shape))
        print("pcr_model output, type_predict: {}, \nangle_predict: {}".format(type_predict.shape, angle_predict.shape))
        with tf2.compat.v1.Session("") as sess:
            pass
            # writer = tf2.summary.FileWriter("saved_model/pcr_graph.txt", sess.graph)
            # writer.close()

            # param_stats = tf2.profiler.profile(graph=sess.graph, options=tf2.profiler.ProfileOptionBuilder.trainable_variables_parameter())
            # flops_stats = tf2.profiler.profile(graph=sess.graph, options=tf2.profiler.ProfileOptionBuilder.float_operation())
            # print("parameters: {}, flops: {}".format(param_stats.total_parameters, flops_stats.total_float_ops))
            # tf2.saved_model.simple_save(sess, "saved_model/pcr_192_64_simple", inputs={"image": img_tensor}, outputs={"type": out_type, "angle": out_angle})
        # serialize to saved_model
        # tf2.saved_model.save(self.model, "saved_model/pcr_192_64_tf")

        type_predict = np.argmax(type_predict, axis=1)
        type_predict = np.ndarray.tolist(type_predict)
        # cast angle into degree
        angle_predict = angle_predict * 180. - 90.
        angle_predict = list(itertools.chain.from_iterable(angle_predict))
        print("post-process output, type_predict: {}, \nangle_predict: {}".format(type_predict, angle_predict))
        return type_predict, angle_predict
