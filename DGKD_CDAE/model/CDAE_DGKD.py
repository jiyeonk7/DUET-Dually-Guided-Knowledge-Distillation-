from __future__ import absolute_import
from __future__ import division
import numpy as np
import math
import os
import tensorflow as tf
from model.AbstractRecommender import AbstractRecommender
from util import Tool
import pickle

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CDAE_DGKD(AbstractRecommender):
    def __init__(self, sess, dataset, model_conf):
        self.sess = sess
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_dict = dataset.train_dict

        self.hidden_dim = model_conf.hidden_dim
        self.batch_size = model_conf.batch_size
        self.loss_function = model_conf.loss_function
        self.learner = model_conf.learner
        self.learning_rate = model_conf.learning_rate
        self.reg = model_conf.reg
        self.act = model_conf.act
        self.symmetric = model_conf.symmetric
        self.corruption_ratio = model_conf.corruption_ratio
        self.eval_matrix = dataset.train_matrix.toarray()
        self.test_batch_size = model_conf.test_batch_size

        self.teacher_dim = model_conf.teacher_dim
        self.alpha = model_conf.alpha
        self.beta = model_conf.beta

        #path of the files containing the result of pre-trained teacher #1
        self.unintFilePath = "./data/ml1m-occf/ml1m_uninteresting/unint_90/u1_0.9.base"
        self.intFilePath = "./data/ml1m-occf/ml1m_interesting/top10/u1_0.10.base"
        
        #uninteresting items score as 1
        with open(self.unintFilePath, "rb") as unintFile:
            self.unint_teacher_matrix = np.full((self.num_users,self.num_items),1)
            self.unint_teacher_mask = np.zeros_like(self.unint_teacher_matrix, dtype=float)
            for line in unintFile:
                line = line.decode()
                u, i, r = line.split("\t")
                self.unint_teacher_mask[int(u)-1, int(i)-1] = 1

        #interesting items score as teacher1's score
        with open(dataset.filename + "_cdae_%d.p" % (self.teacher_dim), "rb") as f:
            self.int_teacher_matrix = pickle.load(f)

        with open(self.intFilePath,"rb") as intFile:
            self.int_teacher_mask = np.zeros_like(self.int_teacher_matrix, dtype=float)
            for line in intFile:
                line = line.decode()
                u,i,r = line.split("\t")
                self.int_teacher_mask[int(u)-1, int(i)-1] = 1


    def _create_placeholders(self):
        with tf.name_scope("placeholders"):
            self.user_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
            self.item_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_items])
            self.keep_prob_ph = tf.compat.v1.placeholder_with_default(1.0, shape=None)
            self.unint_teacher_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_items])
            self.unint_distill_mask_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_items])
            self.int_teacher_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_items])
            self.int_distill_mask_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_items])


    def _create_variables(self):
        with tf.name_scope("variables"):
            self.user_embeddings_var = tf.Variable(tf.random_normal([self.num_users, self.hidden_dim], stddev=0.01))
            self.weights_var = {
                'encoder': tf.Variable(tf.random_normal([self.num_items, self.hidden_dim], stddev=0.01)),
                'decoder': tf.Variable(tf.random_normal([self.hidden_dim, self.num_items], stddev=0.01)),
            }
            self.biases_var = {
                'encoder': tf.Variable(tf.random_normal([self.hidden_dim], stddev=0.01)),
                'decoder': tf.Variable(tf.random_normal([self.num_items], stddev=0.01)),
            }


    def _create_inference(self):
        with tf.name_scope("inference"):
            h = tf.nn.dropout(self.item_ph, self.keep_prob_ph)
            encoder_op = tf.matmul(h, self.weights_var['encoder']) + self.biases_var['encoder']

            user_embedding = tf.nn.embedding_lookup(self.user_embeddings_var, self.user_ph)
            encoder_op += user_embedding  
            encoder_op = Tool.activation_function(self.act, encoder_op)
            if self.symmetric:
                decoder = tf.transpose(self.weights_var['encoder'])
            else:
                decoder = self.weights_var['decoder']
            self.output = tf.matmul(encoder_op, decoder) + self.biases_var['decoder']


    def _create_loss(self):
        with tf.name_scope("loss"):
            CF_loss = Tool.exp_weighted_pointwise_loss(self.loss_function, self.item_ph, self.output, self.item_ph)  # only positive
            unint_loss = Tool.exp_weighted_pointwise_loss(self.loss_function, self.unint_teacher_ph, self.output, self.unint_distill_mask_ph)
            int_loss = Tool.exp_weighted_pointwise_loss(self.loss_function, self.int_teacher_ph, self.output, self.int_distill_mask_ph)
            KD_loss = (1-self.beta)*unint_loss + self.beta*int_loss
            reg_loss = tf.nn.l2_loss(self.weights_var['encoder']) + tf.nn.l2_loss(self.user_embeddings_var)
            if not self.symmetric:
                reg_loss += tf.nn.l2_loss(self.weights_var['decoder'])
            self.loss = (1-self.alpha)*CF_loss + self.alpha * KD_loss + 2 * self.reg * reg_loss


    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = Tool.optimizer(self.learner, self.loss, self.learning_rate)


    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()


    def train_model(self, epoch):
        perm_user_idx = np.random.permutation(self.num_users)
        num_batches = math.floor(self.num_users / self.batch_size)
        if num_batches == 0:
            num_batches, self.batch_size = 1, self.num_users

        total_loss = 0.0
        for batch_id in range(num_batches):
            batch_user_idx = perm_user_idx[batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
            # Build the batch matrix.
            batch_matrix = self.eval_matrix[batch_user_idx]
            unint_batch_teacher = self.unint_teacher_matrix[batch_user_idx]
            unint_batch_distill_mask = self.unint_teacher_mask[batch_user_idx]
            int_batch_teacher = self.int_teacher_matrix[batch_user_idx]
            int_batch_distill_mask = self.int_teacher_mask[batch_user_idx]

            # Train the model for the batch matrix.
            total_loss += self.train_model_per_batch(batch_user_idx, batch_matrix, unint_batch_teacher, unint_batch_distill_mask, int_batch_teacher, int_batch_distill_mask)

        return total_loss / self.num_users


    def train_model_per_batch(self, batch_user_idx, batch_matrix, unint_batch_teacher, unint_batch_distill_mask, int_batch_teacher, int_batch_distill_mask):
        feed_dict = {self.user_ph: batch_user_idx,
                     self.item_ph: batch_matrix,
                     self.keep_prob_ph: 1.0 - self.corruption_ratio,
                     self.unint_teacher_ph: unint_batch_teacher,
                     self.unint_distill_mask_ph: unint_batch_distill_mask,
                     self.int_teacher_ph: int_batch_teacher,
                     self.int_distill_mask_ph: int_batch_distill_mask}
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return loss


    def predict(self, dataset):
        eval_input = dataset.eval_input
        eval_output = np.zeros_like(eval_input)

        batch_size = self.test_batch_size
        num_batches = math.ceil(eval_input.shape[0] / batch_size)
        user_idx = range(eval_input.shape[0])

        for batch_id in range(num_batches):
            batch_user_idx = user_idx[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_eval_matrix = eval_input[batch_id * batch_size: (batch_id + 1) * batch_size]
            feed_dict = {self.user_ph: batch_user_idx, self.item_ph: batch_eval_matrix}
            eval_output[batch_id * batch_size: (batch_id + 1) * batch_size] = \
                self.sess.run(self.output, feed_dict=feed_dict)

        return eval_output
