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
import pickle
import numpy as np

class RNNGenerator(object):
    def __init__(self, itm_cnt, usr_cnt, dim_hidden, n_time_step, learning_rate, grad_clip, emb_dim, lamda=0.2, initdelta=0.05,MF_paras=None,model_type="rnn",use_sparse_tensor=False):
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
        self.param=MF_paras
        self.H = dim_hidden
        self.T = n_time_step
        
        self.MF_paras=MF_paras
        self.grad_clip = grad_clip

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        
        if use_sparse_tensor:
            self.item_sequence = tf.placeholder(tf.float32, [None, self.T, self.V_U])        
            self.user_sequence = tf.placeholder(tf.float32, [None, self.T, self.V_M])

            self.user_indices = tf.placeholder(tf.int64)
            self.user_shape = tf.placeholder(tf.int64)
            self.user_values = tf.placeholder(tf.float64)
            user_sparse_tensor = tf.SparseTensor(user_indices, user_shape, user_values)
            self.user_sequence = tf.sparse_tensor_to_dense(user_sparse_tensor)

            self.item_indices = tf.placeholder(tf.int64)
            self.item_shape = tf.placeholder(tf.int64)
            self.item_values = tf.placeholder(tf.float64)
            item_sparse_tensor = tf.SparseTensor(item_indices, item_shape, item_values)
            self.item_sequence = tf.sparse_tensor_to_dense(item_sparse_tensor)
            
        else:
            self.item_sequence = tf.placeholder(tf.float32, [None, self.T, self.V_U])        
            self.user_sequence = tf.placeholder(tf.float32, [None, self.T, self.V_M])   

        self.rating = tf.placeholder(tf.float32, [None,])
                        
        self.learning_rate = learning_rate
    
    
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.initdelta = initdelta
        
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)

        self.paras_rnn=[]
        self.model_type=model_type
    def _init_MF(self):
        with tf.variable_scope('MF'):
            if self.MF_paras is None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.V_U, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.V_M, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.V_M]))            
                self.user_bias = tf.Variable(tf.zeros([self.V_U])) 
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.user_bias = tf.Variable(self.param[2])           
                self.item_bias = tf.Variable(self.param[3])     

            self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
            self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
            self.i_bias = tf.gather(self.item_bias, self.i)
            self.u_bias = tf.gather(self.user_bias, self.u)

            self.paras_mf=[self.user_embeddings,self.item_embeddings,self.user_bias,self.item_bias]

    def _decode_lstm(self, h_usr, h_itm, reuse=False):
        if False:
            with tf.variable_scope('rating', reuse=reuse):
                w_usr = tf.get_variable('w_usr', [self.H, self.H], initializer=self.weight_initializer)
                b_usr = tf.get_variable('b_usr', [self.H], initializer=self.const_initializer)
                w_itm = tf.get_variable('w_itm', [self.H, self.H], initializer=self.weight_initializer)
                b_itm = tf.get_variable('b_itm', [self.H], initializer=self.const_initializer)

                usr_vec = tf.matmul(h_usr, w_usr) + b_usr
                
                itm_vec = tf.matmul(h_itm, w_itm) + b_itm
                                        
                out_preds = tf.reduce_sum(tf.multiply(usr_vec, itm_vec), 1)                      
                self.paras_rnn.extend([w_usr,b_usr,w_itm,b_itm])
                return out_preds
        else:
            out_preds = tf.reduce_sum(tf.multiply(h_usr, h_itm), 1) 
            print("do not use a fully-connectted layer by the outputing vector of LSTM")  
            return out_preds
            
    def _get_initial_lstm(self, batch_size):
        with tf.variable_scope('initial_lstm'):                        
            c_itm = tf.zeros([batch_size, self.H], tf.float32)
            h_itm = tf.zeros([batch_size, self.H], tf.float32)
            c_usr = tf.zeros([batch_size, self.H], tf.float32)
            h_usr = tf.zeros([batch_size, self.H], tf.float32) 
            # self.paras_rnn.extend([c_itm, h_itm, c_usr, h_usr])   # these variable should be trainable or not                     
            return c_itm, h_itm, c_usr, h_usr

    def _item_embedding(self, inputs, reuse=False):
        with tf.variable_scope('item_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V_U, self.H], initializer=self.emb_initializer)           
            x_flat = tf.reshape(inputs, [-1, self.V_U]) #(N * T, U)       
            x = tf.matmul(x_flat, w) #(N * T, H)
            x = tf.reshape(x, [-1, self.T, self.H]) #(N, T, H)
            self.paras_rnn.extend([w])
            return x
        
    def _user_embedding(self, inputs, reuse=False):
        with tf.variable_scope('user_embedding', reuse=reuse):
           w = tf.get_variable('w', [self.V_M, self.H], initializer=self.emb_initializer)           
           x_flat = tf.reshape(inputs, [-1, self.V_M]) #(N * T, M)       
           x = tf.matmul(x_flat, w) #(N * T, H)
           x = tf.reshape(x, [-1, self.T, self.H]) #(N, T, H)
           self.paras_rnn.extend([w])
           return x
        
    def build_pretrain(self):
        
        batch_size = tf.shape(self.item_sequence)[0]
                       
        c_itm, h_itm, c_usr, h_usr = self._get_initial_lstm(batch_size)
        x_itm = self._item_embedding(inputs=self.item_sequence)
        x_usr = self._user_embedding(inputs=self.user_sequence)    

        itm_lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.H)
        usr_lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.H)
        
        self._init_MF()
        
        for t in range(self.T):
            with tf.variable_scope('itm_lstm', reuse=(t!=0)):
                _, (c_itm, h_itm) = itm_lstm_cell(inputs=x_itm[:,t,:], state=[c_itm, h_itm])            
            with tf.variable_scope('usr-lstm', reuse=(t!=0)):
                _, (c_usr, h_usr) = usr_lstm_cell(inputs=x_usr[:,t,:], state=[c_usr, h_usr])
         
        
        MF_Regularizer = self.lamda * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.u_bias) +tf.nn.l2_loss(self.i_bias))
        RNN_Regularizer = tf.reduce_sum([tf.nn.l2_loss(para) for para in self.paras_rnn])
        
        self.pre_logits_RNN = self._decode_lstm(h_usr, h_itm, reuse=False)         
        self.loss_RNN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating, logits=self.pre_logits_RNN)) + RNN_Regularizer
        self.pre_logits_MF = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias  +self.u_bias       
        self.loss_MF = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating, logits=self.pre_logits_MF)) + MF_Regularizer
        
        self.pre_joint_logits = self.pre_logits_MF + self.pre_logits_RNN
        self.joint_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating, logits=self.pre_joint_logits)) 
        
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer   

        optimizer = self.optimizer(learning_rate=self.learning_rate)
        if self.model_type == 'joint':
            grads = tf.gradients(self.joint_loss, tf.trainable_variables())
        elif self.model_type == 'rnn':
            grads = tf.gradients(self.loss_RNN, tf.trainable_variables())
        elif self.model_type == 'mf':
            grads = tf.gradients(self.loss_MF, tf.trainable_variables())
            
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        self.pretrain_updates = optimizer.apply_gradients(grads_and_vars=grads_and_vars)            
            
        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias +self.u_bias


        self.reward = tf.placeholder(tf.float32)
        self.pg_loss = - tf.reduce_mean(tf.log( tf.sigmoid(self.pre_joint_logits)) * self.reward) + MF_Regularizer + RNN_Regularizer             
        
        pg_grads = tf.gradients(self.pg_loss, tf.trainable_variables())               
        pg_grads_and_vars = list(zip(pg_grads, tf.trainable_variables()))
        self.pg_updates = optimizer.apply_gradients(grads_and_vars=pg_grads_and_vars)                     
        
    def prediction(self, sess, user_sequence, item_sequence, u, i,sparse=False):
        if sparse:
            user_sequence,item_sequence=[ii.toarray() for ii in user_sequence],[ii.toarray() for ii in item_sequence]
        outputs = sess.run(self.pre_joint_logits, feed_dict = {self.user_sequence: user_sequence, 
                        self.item_sequence: item_sequence, self.u: u, self.i: i})  
        return outputs
    
    def predictionItems(self, sess, u):
        outputs = sess.run(self.all_logits, feed_dict = {self.u: u})  
        return outputs
             

    def getRewards(self,sess, samples,sparse=False):
        u_seq,i_seq = [[ sample[i].toarray()  for sample in samples ]  for i in range(2)]
        u,i = [[ sample[i]  for sample in samples ]  for i in range(2,4)]
        
        reward_logits = self.prediction(sess,u_seq,i_seq,u,i)        
        return 2 * (tf.sigmoid(reward_logits) - 0.5)    


    def saveMFModel(self, sess, filename):
        self.paras_mf = [self.user_embeddings,self.item_embeddings,self.user_bias,self.item_bias]
        param = sess.run(self.paras_mf)
        pickle.dump(param, open(filename, 'wb'))


    