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
from dataHelper import FLAGS,helper
from oldMF import DIS
import time


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# load mf model
if os.path.exists(FLAGS.pretrained_model) and FLAGS.pretrained:
	print("use MF model"+FLAGS.pretrained_model)
	paras= pickle.load(open(FLAGS.pretrained_model,"rb"))
else:
	print("fail to load empty ")
	paras=None

g1 = tf.Graph()
g2 = tf.Graph()
sess1 = tf.InteractiveSession(graph=g1)        
sess2 = tf.InteractiveSession(graph=g2)


with g1.as_default():
    gen = G_A(itm_cnt = helper.i_cnt, 
             usr_cnt = helper.u_cnt, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate = FLAGS.learning_rate, 
             grad_clip = 0.2,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = FLAGS.lamda,
             initdelta = 0.05,
             MF_paras=paras,
             model_type=FLAGS.model_type
             )
    gen.build_pretrain()
    init1=tf.global_variables_initializer()
    saver1 = tf.train.Saver(max_to_keep=50)

    
with g2.as_default():
    dis = G_A(itm_cnt = helper.i_cnt, 
             usr_cnt = helper.u_cnt, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate = FLAGS.learning_rate, 
             grad_clip = 0.2,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = FLAGS.lamda,
             initdelta = 0.05,
             MF_paras=paras,
             model_type=FLAGS.model_type
             )
    dis.build_pretrain()
    init2=tf.global_variables_initializer()
    saver2 = tf.train.Saver(max_to_keep=50)

for sess,model,init,saver in zip([sess1,sess2],[gen,dis],[init1,init2],[saver1,saver2]):
    sess.run(init)
    

    checkpoint_filepath= "model/joint-25-0.28000.ckpt"
    saver.restore(sess,checkpoint_filepath)
    scores=helper.evaluateMultiProcess(sess,model)
    if FLAGS.model_type=="mf":
        best_p5=scores[1]
    else:
        best_p5=scores[1][1]
    print(scores)


def main(checkpoint_dir="model/"):
    global best_p5
    for epoch in range(1000):

        if epoch > 0:
            for d_epoch in range(10):

                # for i,(uid,itemid,rating) in enumerate(helper.getBatch4MF()):
                rnn_losses,mf_losses,joint_losses=[],[],[]
                for i,(u_seqs,i_seqs,rating,uid,itemid) in enumerate(helper.getBatchFromSamples(dns=True,sess=sess1,model=gen,fresh=False)):

                    _,loss_mf,loss_rnn,joint_loss= dis.pretrain_step(sess2,  rating, uid, itemid,u_seqs,i_seqs)
                    rnn_losses.append(loss_rnn)
                    mf_losses.append(loss_mf)
                    joint_losses.append(joint_loss)

                print(" rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(np.mean(np.array(rnn_losses)),np.mean(np.array(mf_losses)),np.mean(np.array(joint_losses))) )
                scores= (helper.evaluateMultiProcess(sess2,dis))
                # print(helper.evaluateRMSE(sess,model))
                print(scores)
                if FLAGS.model_type=="mf":
                    curentt_p5_score=scores[1]
                else:
                    curentt_p5_score=scores[1][1]

                if curentt_p5_score >best_p5:        	
                    best_p5=curentt_p5_score
                    saver.save(sess, checkpoint_dir + '%s-%d-%.5f.ckpt'% (FLAGS.model_type,FLAGS.re_rank_list_length,best_p5))
                    mf_model = '%s-%d-%.5f.pkl'% (FLAGS.model_type,FLAGS.re_rank_list_length,best_p5)
                    # model.saveMFModel(sess,mf_model)
                    print(best_p5)

                # Train G
        for g_epoch in range(50):  # 50
            for user in helper.data["uid"].unique(): 

#                pos_items_time_dict=helper.user_item_pos_rating_time_dict.get(user,{})
#                if len(pos_items_time_dict)==0:                   # todo  not do this
#                    continue
                all_rating = gen.predictionItems(sess1,user)                           # todo delete the pos ones
                exp_rating = np.exp(np.array(all_rating) *helper.conf.temperature)
                prob = exp_rating / np.sum(exp_rating)

                neg = np.random.choice(np.arange(helper.i_cnt), size=len(helper.conf.gan_k), p=prob)
                samples=[]
                for  neg_item_id in neg:                # gan_k guding

                    u_seqss,i_seqss= helper.getSeqOverAlltime(user,neg_item_id)
                    predicted = gen.prediction(sess1,u_seqss,i_seqss, [user]*len(u_seqss),[neg_item_id]*len(u_seqss))
                    index=np.argmax(predicted)
                    samples.append((u_seqss[index],i_seqss[index],user,neg_item_id ))
                
                rewards= dis.getRewards(sess2, samples)


                gen.gan_feadback(sess1,samples, rewards)

if __name__== "__main__":
	main()