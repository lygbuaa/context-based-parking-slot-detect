#! /usr/bin/env python3
#-*-coding:UTF-8-*-
# pip install tensorflow-cpu==2.10
import tensorflow as tf2
import numpy as np
import os, sys

# for v1 compatible
tf1 = tf2.compat.v1
# switch to TF1 mode
tf2.compat.v1.disable_eager_execution()
# tf2.x deprecated tf1.contrib.slim, pip install --upgrade tf_slim
import tf_slim as slim

def test_variable():
    # define namespace psd
    with tf1.variable_scope("psd"):
        a1_tf1 = tf1.get_variable(name='a1_tf1', shape=[1,2], initializer=tf1.constant_initializer(1.1)) 
        a2_tf1 = tf1.Variable(tf1.random_normal(shape=[2,3], mean=0, stddev=1), name='a2_tf1')  
        a3_tf2 = tf2.Variable([7.7, 8.8])
    # run graph
    with tf1.Session() as sess:
        # TF1 variable could initialize all in once
        sess.run(tf1.global_variables_initializer())
        # TF1 variable could also initialize independently
        sess.run(a1_tf1.initializer)
        # both TF1 && TF2 vars could use sess.run() && .eval(sess) to retrieve value
        print("a1_tf1 name: {}, shape: {}, val: {}".format(a1_tf1.name, tf1.Variable.get_shape(a1_tf1), sess.run(a1_tf1)))
        print("a2_tf1 name: {}, shape: {}, val: {}".format(a2_tf1.name, tf1.Variable.get_shape(a2_tf1), a2_tf1.eval(sess)))
        print("a3_tf2 name: {}, shape: {}, val: {}".format(a3_tf2.name, tf1.Variable.get_shape(a3_tf2), sess.run(a3_tf2)))
        print("a3_tf2 name: {}, shape: {}, val: {}".format(a3_tf2.name, tf1.Variable.get_shape(a3_tf2), a3_tf2.eval(sess)))

    # print all variables automatically collected in the graph
    print("global_variables: {}".format(tf1.global_variables()))

if __name__ == '__main__':
    test_variable()