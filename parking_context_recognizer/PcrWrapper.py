import time
import warnings
import glob
import itertools
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from parking_context_recognizer.utils import *
from parking_context_recognizer.models import model_mobilenetv2 as pcr_model
from parking_context_recognizer.config import *

class PCRWrapper(object):
    def __init__(self, weight_file):
        self.weight_file = weight_file

    def __del__(self):
        tf.keras.backend.clear_session()

    def destroy(self):
        tf.keras.backend.clear_session()

    def init(self, img_tensor):
        model_input = tf.keras.layers.Input(tensor=img_tensor)
        self.model = pcr_model(model_input)
        self.model.load_weights(self.weight_file)
        # self.sessions = K.get_session()
        # self.graphs = tf.get_default_graph()
        
    def run(self, img_tensor):
        type_predict, angle_predict = self.model.predict(img_tensor, steps=1)
        # with tf.Session("") as sess:
            # sess.run(self.model)
        print("pcr_model output, type_predict: {}, \nangle_predict: {}".format(type_predict, angle_predict))
        # tf.keras.backend.clear_session()

        type_predict = np.argmax(type_predict, axis=1)
        type_predict = np.ndarray.tolist(type_predict)
        # cast angle into degree
        angle_predict = angle_predict * 180. - 90.
        angle_predict = list(itertools.chain.from_iterable(angle_predict))
        print("post-process output, type_predict: {}, \nangle_predict: {}".format(type_predict, angle_predict))
        return type_predict, angle_predict
