import pandas as pd 
import os
import datetime
import numpy as np 
import pickle
import config
from tools import log_time_delta
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix,csr_matrix
import math
from config import Singleton
import sklearn
import tensorflow as tf
from RNN import RNNGenerator as G_A
from dataHelper  import DataHelper

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

           
pool=Pool(cpu_count())
u_seqss,i_seqss,ratingss=helper.getBatch_prepare(pool,mode="test", epoches_size=1)
#
#pos_index = [i for i,r in enumerate(ratingss) if r>3.99]
#neg_index = [i for i,r in enumerate(ratingss) if r<=3.99]
#
#pos_batches = [(u_seqss[i],i_seqss[i],1) for i in pos_index]
#neg_batches = [(u_seqss[i],i_seqss[i],0) for i in neg_index]
#                
#if mode=="train" and shuffle:                
#    batches = pos_batches +  neg_batches
#    batches = sklearn.utils.shuffle(batches)    
#    
#n_batches= int(len(batches)/ self.conf.batch_size)
#for i in range(0,n_batches):
#    batch = batches[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
#    u_seqs=pool.map(sparse2dense, [ii[0] for ii in batch])
#    i_seqs=pool.map(sparse2dense, [ii[1] for ii in batch])
#    ratings=[ii[2] for ii in batch]
#    yield u_seqs,i_seqs,ratings


             
if __name__== "__main__":
    from multiprocessing import  freeze_support
    from sklearn.metrics import roc_auc_score
    freeze_support()
    flagFactory=Singleton()
    FLAGS=flagFactory.getInstance()
    helper=DataHelper(FLAGS)	            
    
    model = G_A(itm_cnt = 1682, 
             usr_cnt = 943, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate = FLAGS.learning_rate, 
             grad_clip = 5.0,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = 0.1 / FLAGS.batch_size,
             initdelta = 0.05)

    model.build_pretrain()
    tf.get_variable_scope().reuse_variables()
    
    
    sess = tf.InteractiveSession()    
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=40)  
    
    for e in range(10):
        curr_loss = 0  
        start_t = time.time()
        for x,y,z in helper.prepare():
            _,l = model.pretrain_step(sess, x, y, z)    
        
#        results=np.array([]) 
        y_true = np.array([]) 
        y_scores = np.array([]) 
        for x,y,z in helper.prepare(mode="test"):
            predicted = model.prediction(sess, x, y)
            y_true = np.append(y_true,z)  
            y_scores = np.append(y_scores,predicted)  
#            error=(np.array(predicted)-np.array(z))
#            se= np.square(error)
#            results=np.append(results,se)      
        print ("auc = %.6f" % roc_auc_score(y_true, np.array(y_scores)))           
#        mse=np.mean(results)
#        print ("rmse = %.6f" % math.sqrt(mse))
        print( "Elapsed time: ", time.time() - start_t)
        