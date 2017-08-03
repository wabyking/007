# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf


class RNNGenerator(object):
    def __init__(self, itm_cnt, usr_cnt, dim_hidden, n_time_step, learning_rate, grad_clip, emb_dim, lamda=0.1, initdelta=0.05):
        """
        Args:
            dim_itm_embed: (optional) Dimension of item embedding.
            dim_usr_embed: (optional) Dimension of user embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM. 
            usr_cnt: (optional) The size of all users.
            itm_cnt: (optional) The size of all items.
        """
        self.V_M = itm_cnt
        self.V_U = usr_cnt
        
        self.H = dim_hidden
        self.T = n_time_step
        
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        
        self.item_sequence = tf.placeholder(tf.float32, [None, self.T, self.V_U])        
        self.user_sequence = tf.placeholder(tf.float32, [None, self.T, self.V_M])        
        self.rating = tf.placeholder(tf.float32, [None,])
                        
        self.learning_rate = learning_rate
    
    
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.initdelta = initdelta

        with tf.variable_scope('MF'):
            self.user_embeddings = tf.Variable(
                tf.random_uniform([self.V_U, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                  dtype=tf.float32))
            self.item_embeddings = tf.Variable(
                tf.random_uniform([self.V_M, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                  dtype=tf.float32))
            self.item_bias = tf.Variable(tf.zeros([self.V_M]))

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)



    def _decode_lstm(self, h_usr, h_itm, reuse=False):
        with tf.variable_scope('rating', reuse=reuse):
            w_usr = tf.get_variable('w_usr', [self.H, self.H], initializer=self.weight_initializer)
            b_usr = tf.get_variable('b_usr', [self.H], initializer=self.const_initializer)
            w_itm = tf.get_variable('w_itm', [self.H, self.H], initializer=self.weight_initializer)
            b_itm = tf.get_variable('b_itm', [self.H], initializer=self.const_initializer)

            usr_vec = tf.matmul(h_usr, w_usr) + b_usr
            
            itm_vec = tf.matmul(h_itm, w_itm) + b_itm
                                    
            out_preds = tf.reduce_sum(tf.multiply(usr_vec, itm_vec), 1)                      
            
            return out_preds

            
    def _get_initial_lstm(self, batch_size):
        with tf.variable_scope('initial_lstm'):                        
            c_itm = tf.zeros([batch_size, self.H], tf.float32)
            h_itm = tf.zeros([batch_size, self.H], tf.float32)
            c_usr = tf.zeros([batch_size, self.H], tf.float32)
            h_usr = tf.zeros([batch_size, self.H], tf.float32)                       
            return c_itm, h_itm, c_usr, h_usr

    def _item_embedding(self, inputs, reuse=False):
        with tf.variable_scope('item_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V_U, self.H], initializer=self.emb_initializer)           
            x_flat = tf.reshape(inputs, [-1, self.V_U]) #(N * T, U)       
            x = tf.matmul(x_flat, w) #(N * T, H)
            x = tf.reshape(x, [-1, self.T, self.H]) #(N, T, H)
            return x
        
    def _user_embedding(self, inputs, reuse=False):
        with tf.variable_scope('user_embedding', reuse=reuse):
           w = tf.get_variable('w', [self.V_M, self.H], initializer=self.emb_initializer)           
           x_flat = tf.reshape(inputs, [-1, self.V_M]) #(N * T, M)       
           x = tf.matmul(x_flat, w) #(N * T, H)
           x = tf.reshape(x, [-1, self.T, self.H]) #(N, T, H)
           return x
        
    def build_pretrain(self):
        
        batch_size = tf.shape(self.item_sequence)[0]
                       
        c_itm, h_itm, c_usr, h_usr = self._get_initial_lstm(batch_size)
        x_itm = self._item_embedding(inputs=self.item_sequence)
        x_usr = self._user_embedding(inputs=self.user_sequence)    

        itm_lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.H)
        usr_lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.H)
        
        for t in range(self.T):
            with tf.variable_scope('itm_lstm', reuse=(t!=0)):
                _, (c_itm, h_itm) = itm_lstm_cell(inputs=x_itm[:,t,:], state=[c_itm, h_itm])            
            with tf.variable_scope('usr-lstm', reuse=(t!=0)):
                _, (c_usr, h_usr) = usr_lstm_cell(inputs=x_usr[:,t,:], state=[c_usr, h_usr])
         
        
        self.pre_logits_RNN = self._decode_lstm(h_usr, h_itm, reuse=False) 
        
        self.loss_RNN = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating,logits=self.pre_logits_RNN)
        
        self.pre_logits_MF = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        
        self.loss_MF = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating,
                                                                logits=self.pre_logits_MF) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias)
        )
                
        self.joint_loss = self.loss_RNN + self.loss_MF
        
#        loss = tf.reduce_mean(tf.square(self.predictions - self.rating))
        
        self.pretrain_loss = self.joint_loss / tf.to_float(batch_size)
        
        self.predictions = self.pre_logits_RNN + self.pre_logits_MF
        
        pretrain_opt = tf.train.AdamOptimizer(self.learning_rate)            
        grads = tf.gradients(self.pretrain_loss, tf.trainable_variables())
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        self.pretrain_updates = pretrain_opt.apply_gradients(grads_and_vars=grads_and_vars)
        
#        self.i_prob = tf.gather(
#            tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]),
#            self.i)
#        self.loss_list = tf.transpose(tf.pack(loss_list), (1, 0))#(N,T)           
#        self.rewards = tf.placeholder(tf.float32, shape=[None])     
#        
#        self.pg_loss = tf.reduce_sum(tf.reduce_sum(self.loss_list, 1) * self.rewards) / tf.to_float(batch_size)
#        pg_opt = tf.train.AdamOptimizer(self.learning_rate)            
#        pg_grads = tf.gradients(self.pg_loss, tf.trainable_variables())
#        pg_grads_and_vars = list(zip(pg_grads, tf.trainable_variables()))
#        self.pg_updates = pg_opt.apply_gradients(grads_and_vars=pg_grads_and_vars)                                
        
    def prediction(self, sess, user_sequence, item_sequence, u, i):
        outputs = sess.run([self.predictions], feed_dict = {self.user_sequence: user_sequence, 
                        self.item_sequence: item_sequence, self.u: u, self.i: i})  
        return outputs
           
    def pretrain_step(self, sess, user_sequence, item_sequence, rating, u, i):        
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict = {self.user_sequence: user_sequence, 
                        self.item_sequence: item_sequence, self.rating: rating, self.u: u, self.i: i})
        return outputs

#    def unsupervised_train_step(self, sess, features_batch, sampled_captions_batch, rewards):
#        outputs = sess.run([self.pg_updates, self.pretrain_loss, self.loss_list, self.pg_loss], feed_dict = {self.features: features_batch, 
#                        self.captions: sampled_captions_batch, self.rewards: rewards})
#                        
#        return outputs
