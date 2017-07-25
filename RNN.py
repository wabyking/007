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
    def __init__(self, itm_cnt, usr_cnt, dim_hidden, n_time_step, learning_rate, grad_clip):
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

        loss = 0.0
#        loss_list = []
        itm_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        usr_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        
        for t in range(self.T):
            with tf.variable_scope('itm_lstm', reuse=(t!=0)):
                _, (c_itm, h_itm) = itm_lstm_cell(inputs=x_itm[:,t,:], state=[c_itm, h_itm])            
            with tf.variable_scope('usr-lstm', reuse=(t!=0)):
                _, (c_usr, h_usr) = usr_lstm_cell(inputs=x_usr[:,t,:], state=[c_usr, h_usr])
         
        
        self.predictions = self._decode_lstm(h_usr, h_itm, reuse=False)  
        
#                
        loss = tf.reduce_mean(tf.square(self.predictions - self.rating))
        
        self.pretrain_loss = loss / tf.to_float(batch_size)
#        
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
        
    def prediction(self, sess, user_sequence, item_sequence):
        outputs = sess.run([self.predictions], feed_dict = {self.user_sequence: user_sequence, 
                        self.item_sequence: item_sequence})  
        return outputs
           
    def pretrain_step(self, sess, user_sequence, item_sequence, rating):        
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict = {self.user_sequence: user_sequence, 
                        self.item_sequence: item_sequence, self.rating: rating})
        return outputs

#    def unsupervised_train_step(self, sess, features_batch, sampled_captions_batch, rewards):
#        outputs = sess.run([self.pg_updates, self.pretrain_loss, self.loss_list, self.pg_loss], feed_dict = {self.features: features_batch, 
#                        self.captions: sampled_captions_batch, self.rewards: rewards})
#                        
#        return outputs
