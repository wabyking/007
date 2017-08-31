import pandas as pd 
import os
import datetime
import numpy as np 
import pickle

from tools import log_time_delta
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix,csr_matrix
import math
from config import Singleton
import sklearn
import tensorflow as tf
from Discrimiator import Dis
from Generator import Gen
from tqdm import tqdm
from config import Singleton
from dataHelper import DataHelper
FLAGS=Singleton().get_andy_flag()
helper=DataHelper(FLAGS)
#from oldMF import DIS
import time
import random

os.environ['CUDA_VISIBLE_DEVICES'] = ''


g1 = tf.Graph()
g2 = tf.Graph()
sess1 = tf.InteractiveSession(graph=g1)        
sess2 = tf.InteractiveSession(graph=g2)

paras=None


with g1.as_default():
#    tf.get_variable_scope().reuse_variables()
    gen = Gen(itm_cnt = helper.i_cnt, 
             usr_cnt = helper.u_cnt, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate =  0.001, 
             grad_clip = 0.2,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = FLAGS.lamda,
             initdelta = 0.05,
             MF_paras=paras,
             model_type="joint",
             update_rule = 'sgd',
             use_sparse_tensor=FLAGS.sparse_tensor
             )
    gen.build_pretrain()
    init1=tf.global_variables_initializer()
    saver1 = tf.train.Saver(max_to_keep=50)
    sess1.run(init1)
#    checkpoint_filepath= "model/Gen_gan/joint-25-0.18548-0.19452.ckpt"
#    saver1.restore(sess1,checkpoint_filepath)
    
with g2.as_default():
#    tf.get_variable_scope().reuse_variables()
    dis = Dis(itm_cnt = helper.i_cnt, 
             usr_cnt = helper.u_cnt, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate = 0.0005, 
             grad_clip = 0.2,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = FLAGS.lamda,
             initdelta = 0.05,
             MF_paras=paras,
             model_type="joint",
             update_rule = 'sgd',
             use_sparse_tensor=FLAGS.sparse_tensor
             )
    dis.build_pretrain()
    init2=tf.global_variables_initializer()
    saver2 = tf.train.Saver(max_to_keep=50)
    sess2.run(init2)                
#    checkpoint_filepath= "model/Dis/joint-25-0.26000-0.28933.ckpt"
    checkpoint_filepath= "model/netflix_three_month/Dis/joint-25-0.16839-0.18968.ckpt"
    # checkpoint_filepath= "model/Dis/joint-25-0.26067-0.29000.ckpt"
#    checkpoint_filepath= "model/Dis/joint-50-0.25467-0.23867.ckpt"
    saver2.restore(sess2,checkpoint_filepath)


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

#df = helper.data
# pick 100 test users to evaluate metrics and 600 users' training data.
def main(checkpoint_dir="model/"):
    
    scores=(helper.evaluateMultiProcess(sess1, gen))
    scores=(helper.evaluateMultiProcess(sess2, dis))
    print(scores)
    if FLAGS.model_type=="mf":
        best_p5=scores[1]
    else:
        best_p5=scores[1][1]
    print("orginal p5 score %.5f" % best_p5)
    
    for e in range(20):
        if e>0:    
            for g_epoch in range(20):    
                rewardes,pg_losses=[],[] 
                for user in tqdm(helper.test_users):  #df["uid"].unique() 
                    sample_lambda,samples = 0.5,[]
                    pos = helper.user_item_pos_rating_time_dict.get(user,{})   
                    all_prob = softmax(gen.predictionItems(sess1,user))
                    pn = (1 - sample_lambda) * all_prob
                    pn[list(pos.keys())] += sample_lambda * 1.0 / len(pos)
                    sample_items = np.random.choice(np.arange(helper.i_cnt), 2 * 32, p=pn)#len(pos)
                    for item in sample_items:
                        if item in list(pos.keys()):
                            pos_itm, t = item, pos[item]
                            u_seqs,i_seqs = helper.getSeqInTime(user,pos_itm,t)
                            samples.append((u_seqs,i_seqs,user,pos_itm))
                        else:
                            neg_itm, t = item, 0
                            u_seqs,i_seqs = helper.getSeqInTime(user,neg_itm,t)
                            samples.append((u_seqs,i_seqs,user,neg_itm))
                            
                    u_seq_pos,i_seq_pos = [[ s[j].toarray() for s in samples ]  for j in range(2)]
                    u_pos,i_pos = [[ s[j]  for s in samples ]  for j in range(2,4)]
                    
                    reward = dis.prediction(sess2,u_seq_pos,i_seq_pos,u_pos,i_pos,sparse=False)
                    reward = (reward-np.mean(reward))/np.std(reward)
#                    reward = reward * all_prob[sample_items] / pn[sample_items]                    
                    pg_loss = gen.unsupervised_train_step(sess1, u_seq_pos,i_seq_pos,u_pos,i_pos, reward)
                    pg_losses.append(pg_loss)
                    rewardes.append(np.sum(reward))
                print("pg loss : %.5f reward : %.5f "%(np.mean(np.array(pg_losses)),np.sum(np.array(rewardes))))                
                scores = helper.evaluateMultiProcess(sess1, gen)    
                print (scores)
                            
        for d_epoch in range(3):  
            rnn_losses,mf_losses,joint_losses=[],[],[]

            for user in tqdm(helper.test_users):                                                    
                samples = []
                all_prob = softmax(gen.predictionItems(sess1,user))
                pos_dict = helper.user_item_pos_rating_time_dict.get(user,{})     
                                                
                pos = [list(pos_dict.keys())[i] for i in np.random.choice(len(pos_dict),32)]
                neg = np.random.choice(np.arange(helper.i_cnt), size=32, p=all_prob)
                for i in range(len(pos)):
                    pos_itm, t = pos[i],pos_dict[pos[i]]
                    u_seqs,i_seqs = helper.getSeqInTime(user,pos_itm,t)
                    samples.append((u_seqs,i_seqs,user,pos_itm,1.))
                    
                    neg_itm, t = neg[i],0
                    u_seqs,i_seqs = helper.getSeqInTime(user,neg_itm,t)   
                    samples.append((u_seqs,i_seqs,user,neg_itm,0.))

                u_seq,i_seq = [[ s[j].toarray()  for s in samples ]  for j in range(2)]
                u,i = [[ s[j]  for s in samples ]  for j in range(2,4)]
                ratings = [ s[4]  for s in samples ] 
                
                _,loss_mf,loss_rnn,joint_loss,rnn,mf = dis.pretrain_step(sess2,ratings, u, i,u_seq,i_seq)                    
                rnn_losses.append(loss_rnn)
                mf_losses.append(loss_mf)
                joint_losses.append(joint_loss)
            print("rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(np.mean(np.array(loss_rnn)),np.mean(np.array(loss_mf)),np.mean(np.array(joint_loss))))            
            scores = helper.evaluateMultiProcess(sess2, dis)    
            print (scores)
        
if __name__== "__main__":
    main()
