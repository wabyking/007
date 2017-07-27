#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:00:22 2017

@author: nlp
"""
import config
import numpy as np
import tensorflow as tf
import time
from RNN import RNNGenerator as G_A
from dataHelper import DataHelper
import math
batch_itm_sequence = np.array([[[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],
        [[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],
        [[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],
        [[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],
        [[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],
        [[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2],[0,5,4,2,1,2,3,5,7,2]],],dtype=np.float32)

batch_usr_sequence = np.array([[[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]],
        [[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]],
        [[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]],
        [[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]],
        [[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]],
        [[0,5,4,2,1],[0,5,4,2,1],[0,5,4,2,1]]],dtype=np.float32)

batch_rating = np.array([5,2,3,1,5,6])

batch_size = 6
itm_cnt = 5
usr_cnt = 10

learning_rate = 10e-3
n_epochs_pretrain = 1

dim_embed=40
dim_hidden=40
n_time_step=3

FLAGS=config.getTestFlag()
helper=DataHelper(FLAGS)  

model = G_A(itm_cnt = helper.i_cnt, 
                         usr_cnt = helper.u_cnt, 
                         dim_hidden = dim_hidden, 
                         n_time_step = FLAGS.item_windows_size, 
                         learning_rate = learning_rate, 
                         grad_clip = 5.0)

model.build_pretrain()
tf.get_variable_scope().reuse_variables()
#model.build_sampler(max_len=20)

n_examples = len(batch_itm_sequence)
n_iters_per_epoch = int(np.ceil(float(n_examples)/batch_size))

print( "The number of epoch: %d" %n_epochs_pretrain)
print( "Data size: %d" % n_examples)
print( "Batch size: %d" % batch_size)
print( "Iterations per epoch: %d" % n_iters_per_epoch)

sess = tf.InteractiveSession()    
tf.global_variables_initializer().run()
saver = tf.train.Saver(max_to_keep=40)  



for e in range(FLAGS.n_epochs):
    curr_loss = 0  
    start_t = time.time()
    for x,y,z in helper.getBatch():
        _,l = model.pretrain_step(sess, x, y, z)
        curr_loss += l
        # print( "Elapsed time: ", time.time() - start_t)
        print( "Current epoch loss: ", l)


    results=np.array([])
    for x,y,z in helper.getBatch(flag="test"):
        predicted = model.prediction(sess, x, y)
        error=(np.array(predicted)-np.array(z))
        se= np.square(error)
        results=np.append(results,se)
    mse=np.mean(results)
    print ("rmse = %.6f" % math.sqrt(mse))


