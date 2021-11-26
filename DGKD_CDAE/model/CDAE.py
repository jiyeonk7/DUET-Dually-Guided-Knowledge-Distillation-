'''
Reference: Yao Wu et al. ""Collaborative Denoising Auto-Encoders for Top-N Recommender Systems" in WSDM 2016
'''

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


class CDAE(AbstractRecommender):
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
        self.negative_ratio = model_conf.negative_ratio
        self.eval_matrix = dataset.train_matrix.toarray()
        self.test_batch_size = model_conf.test_batch_size

        self.save_output = model_conf.save_output


    def random_sampling(self, ratio):
        # sample users randomly
        masking_matrix = np.zeros_like(self.eval_matrix)
        for u in range(self.num_users):
            missing_idx = np.where(self.eval_matrix[u] == 0)[0]
            sample_idx = np.random.choice(missing_idx, int(ratio * len(missing_idx)))
            masking_matrix[u, sample_idx] = 1
        return masking_matrix


    def _create_placeholders(self):
        with tf.name_scope("placeholders"):
            self.user_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
            self.item_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_items])
            self.keep_prob_ph = tf.compat.v1.placeholder_with_default(1.0, shape=None)
            self.negative_mask_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_items])


    def _create_variables(self):
        with tf.name_scope("variables"):
            self.user_embeddings_var = tf.Variable(tf.compat.v1.random_normal([self.num_users, self.hidden_dim], stddev=0.01))
            self.weights_var = {
                'encoder': tf.Variable(tf.compat.v1.random_normal([self.num_items, self.hidden_dim], stddev=0.01)),
                'decoder': tf.Variable(tf.compat.v1.random_normal([self.hidden_dim, self.num_items], stddev=0.01)),
            }
            self.biases_var = {
                'encoder': tf.Variable(tf.compat.v1.random_normal([self.hidden_dim], stddev=0.01)),
                'decoder': tf.Variable(tf.compat.v1.random_normal([self.num_items], stddev=0.01)),
            }


    def _create_inference(self):
        with tf.name_scope("inference"):
            h = tf.nn.dropout(self.item_ph, self.keep_prob_ph)
            encoder_op = tf.matmul(h, self.weights_var['encoder']) + self.biases_var['encoder']

            # Find the user embedding vector for a given user_ph.
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings_var, self.user_ph)
            encoder_op += user_embedding  # Add the user embedding for CDAE.
            encoder_op = Tool.activation_function(self.act, encoder_op)
            if self.symmetric:
                decoder = tf.transpose(self.weights_var['encoder'])
            else:
                decoder = self.weights_var['decoder']
            self.output = tf.matmul(encoder_op, decoder) + self.biases_var['decoder']


    def _create_loss(self):
        with tf.name_scope("loss"):
            loss = Tool.exp_weighted_pointwise_loss(self.loss_function, self.item_ph, self.output,self.item_ph)
            loss += Tool.exp_weighted_pointwise_loss(self.loss_function, self.item_ph, self.output,self.negative_mask_ph)
            reg_loss = tf.nn.l2_loss(self.weights_var['encoder']) + tf.nn.l2_loss(self.user_embeddings_var)
            if not self.symmetric:
                reg_loss += tf.nn.l2_loss(self.weights_var['decoder'])
            self.loss = loss + 2 * self.reg * reg_loss


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
        negative_mask_matrix = self.random_sampling(self.negative_ratio)

        total_loss = 0.0
        for batch_id in range(num_batches):
            batch_user_idx = perm_user_idx[batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
            # Build the batch matrix.
            batch_matrix = self.eval_matrix[batch_user_idx]
            batch_negative_mask = negative_mask_matrix[batch_user_idx]

            # Train the model for the batch matrix.
            total_loss += self.train_model_per_batch(batch_user_idx, batch_matrix, batch_negative_mask)

        return total_loss / self.num_users


    def train_model_per_batch(self, batch_user_idx, batch_matrix, batch_negative_mask):
        feed_dict = {self.user_ph: batch_user_idx,
                     self.item_ph: batch_matrix,
                     self.keep_prob_ph: 1.0 - self.corruption_ratio,
                     self.negative_mask_ph: batch_negative_mask}
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

        if dataset.mode == 'test' and self.save_output:
            with open(dataset.filename + "_cdae_%d.p"%(self.hidden_dim), 'wb') as f:
                pickle.dump(eval_output, f)
            print("Teacher output saved!")

        return eval_output
