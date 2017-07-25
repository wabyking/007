#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:00:22 2017

@author: nlp
"""
import numpy as np
import tensorflow as tf
import time
from dataHelper import DataHelper
import pickle
from RNN import RNNGenerator as G_A
#batch_itm_sequence = np.array([[[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],
#        [[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],
#        [[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],
#        [[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],
#        [[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],
#        [[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],],dtype=np.float32)
#
#batch_usr_sequence = np.array([[[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]],
#        [[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]],
#        [[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]],
#        [[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]],
#        [[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]],
#        [[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]]],dtype=np.float32)
#
#batch_rating = np.array([5,2,3,1,5,6])

def setupFlag(): 
    
    flags = tf.app.flags
    flags.DEFINE_string("dataset", "moviesLen-100k", "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("test_file_name", "ua.test", "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("train_file_name", "ua.base", "Comma-separated list of hostname:port pairs")
    flags.DEFINE_integer("batch_size", 32, "Batch size of data while training")
    flags.DEFINE_integer("user_delta", 7, "Batch size of data while training")
    flags.DEFINE_integer("item_delta", 7, "Batch size of data while training")  # TODO :  user_delta could not equals to item_delta
    flags.DEFINE_integer("item_windows_size", 10, "Batch size of data while training")
    flags.DEFINE_integer("learning_rate", 10e-3, "learning rate")
    flags.DEFINE_integer("dim_embed", 40, "Dimensions of input embeddings")
    flags.DEFINE_integer("dim_hidden", 40, "Dimensions of hidden size")
    flags.DEFINE_integer("n_epochs_pretrain", 40, "Epochs in the training phrase")    
    flags.DEFINE_boolean("TestAccuracy", True, "Test accuracy")

	# FLAGS = flags.FLAGS
	# # FLAGS.workernum=4
    return flags.FLAGS


FLAGS = setupFlag()
helper = DataHelper(FLAGS)

model = G_A(itm_cnt = helper.i_cnt, 
            usr_cnt = helper.u_cnt, 
            dim_hidden = FLAGS.dim_embed,
            n_time_step = FLAGS.item_windows_size, 
            learning_rate = FLAGS.learning_rate, 
            grad_clip = 5.0)

model.build_pretrain()
tf.get_variable_scope().reuse_variables()

print "Batch size: %d" % FLAGS.batch_size
print "The size of all users: %d" % helper.u_cnt
print "The size of all items: %d" % helper.i_cnt  

sess = tf.InteractiveSession()    
tf.initialize_all_variables().run()
saver = tf.train.Saver(max_to_keep=40)  
  
for e in range(FLAGS.n_epochs_pretrain):
    curr_loss = 0  
    n_iters_per_epoch = 0
    start_t = time.time()
    
    for batch_u_seq, batch_i_seq, batch_rating in helper.getBatch():        
        _,l = model.pretrain_step(sess, batch_u_seq, batch_i_seq, batch_rating)
        curr_loss += l
        n_iters_per_epoch += 1
        print l
        
    if FLAGS.TestAccuracy:
        print "Elapsed time: ", time.time() - start_t
        print "Current epoch loss: ", curr_loss / n_iters_per_epoch
    