import numpy as np
import math
import time
import tensorflow as tf
import keras.backend as K

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def optimizer(learner, loss, learning_rate, momentum=0.9):
    opt = None
    if learner.lower() == "adagrad":
        opt = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8).minimize(loss)
    elif learner.lower() == "rmsprop":
        opt = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    elif learner.lower() == "adam":
        opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif learner.lower() == "gd":
        opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif learner.lower() == "momentum":
        opt = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
    else:
        raise ValueError("please select a suitable optimizer")
    return opt


def pairwise_loss(loss_function, y, margin=0.5):
    loss = None
    if loss_function.lower() == "bpr":
        loss = -tf.reduce_sum(tf.log_sigmoid(y))
    elif loss_function.lower() == "hinge":
        loss = tf.reduce_sum(tf.maximum(y + margin, 0))
    elif loss_function.lower() == "square":
        loss = tf.reduce_sum(tf.square(1 - y))
    else:
        raise Exception("please choose a suitable loss function")
    return loss


def pointwise_loss(loss_function, y, pred_y):
    loss = None
    if loss_function.lower() == "cross_entropy":
        loss = -tf.reduce_sum(y * tf.log_sigmoid(pred_y) + (1 - y) * tf.log_sigmoid(1 - pred_y))
    elif loss_function.lower() == "multinominal":
        loss = -tf.reduce_mean(tf.reduce_sum(y * tf.nn.log_softmax(pred_y)))
    elif loss_function.lower() == "square":
        loss = tf.reduce_sum(tf.square(y - pred_y))
    else:
        raise Exception("please choose a suitable loss function")
    return loss


def relaxed_ranking_loss(pred_y, interesting_weight, uninteresting_weight):
    #기존 loss와 다르게...relaxed ranking loss라서 일단 ranking loss는 CD도 하니까 참고해보기
    interesting = pred_y*interesting_weight
    uninteresting = pred_y*uninteresting_weight

    #above shape 알수없음
    above = tf.reduce_sum(interesting, 1)
    above = tf.reshape(above, [-1, 1])

    #shape 일관적으로 (?,1682)
    flipped = K.reverse(interesting, axes=1)
    expo1 = tf.math.exp(flipped)
    below1 = tf.math.cumsum(expo1, axis=1)

    #expo2는 (?,1682), below2는 알수없음
    expo2 = tf.math.exp(uninteresting)
    below2 = tf.reduce_sum(expo2, 1)
    below2 = tf.reshape(below2, [-1,1])

    base = below1 + below2
    expo = tf.math.log(base)
    below = tf.reduce_sum(expo, 1)
    below = tf.reshape(below, [-1,1])

    #RRD_loss는 텐서(모양은 알수없음)
    ab = -(above-below)
    RRD_loss = tf.reduce_sum(ab)

    return RRD_loss

#implicit version
def weighted_pointwise_loss(loss_function, y, pred_y, weight):
    if loss_function.lower() == "cross_entropy":
        loss = - tf.reduce_sum(weight * (y * tf.log(tf.sigmoid(pred_y) + 1e-10) + (1 - y) * tf.log(1 - tf.sigmoid(pred_y) + 1e-10)))
    elif loss_function.lower() == "square":
        loss = tf.reduce_sum(weight * (tf.square(y - pred_y)))
    else:
        raise Exception("please choose a suitable loss function")
    return loss


#explicit version
def exp_weighted_pointwise_loss(loss_function, y, pred_y, weight):
    if loss_function.lower() == "cross_entropy":
        loss = - tf.reduce_sum(weight * ((y/5) * tf.math.log(tf.sigmoid(pred_y) + 1e-10) + (1 - y/5) * tf.math.log(1 - tf.sigmoid(pred_y) + 1e-10)))
    elif loss_function.lower() == "square":
        loss = tf.reduce_sum(weight * (tf.square(y - pred_y)))
    else:
        raise Exception("please choose a suitable loss function")
    return loss


def activation_function(act, act_input):
    act_func = None
    if act == "sigmoid":
        act_func = tf.nn.sigmoid(act_input)
    elif act == "tanh":
        act_func = tf.nn.tanh(act_input)
    elif act == "relu":
        act_func = tf.nn.relu(act_input)
    elif act == "elu":
        act_func = tf.nn.elu(act_input)
    elif act == "identity":
        act_func = tf.identity(act_input)
    else:
        raise NotImplementedError("ERROR")
    return act_func


def kernel(kernel_type, kernel_h, dist):
    if kernel_type.lower() == 'epanechnikov':
        return (3 / 4) * max((1 - (dist / kernel_h) ** 2), 0)
    if kernel_type.lower() == 'uniform':
        return (dist < kernel_h)
    if kernel_type.lower() == 'triangular':
        return max((1 - dist / kernel_h), 0)
    if kernel_type.lower() == 'random':
        return np.random.uniform(0, 1) * (dist < kernel_h)


def dist(dist_type, a, b):
    if np.array_equal(a, b): return 0
    if dist_type.lower() == 'arccos':
        return (2 / math.pi) * np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    elif dist_type == 'cos':
        return (1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))  # 0 ~ 2
    else:
        raise NameError("Please write correct dist_type")


def getlocaltime():
    date = time.strftime('%y-%m-%d', time.localtime())
    current_time = time.strftime('%H:%M:%S', time.localtime())
