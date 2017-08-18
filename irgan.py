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
from Discrimiator import Dis
from Generator import Gen

from dataHelper import FLAGS,helper
#from oldMF import DIS
import time


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# load mf model
if os.path.exists(FLAGS.pretrained_model) and FLAGS.pretrained:
	print("use MF model"+FLAGS.pretrained_model)
	paras= pickle.load(open(FLAGS.pretrained_model,"rb"))
else:
	print("fail to load empty ")
	paras=None

#g1 = tf.Graph()
#g2 = tf.Graph()
#sess1 = tf.InteractiveSession(graph=g1)        
#sess2 = tf.InteractiveSession(graph=g2)


gen = Gen(itm_cnt = helper.i_cnt, 
         usr_cnt = helper.u_cnt, 
         dim_hidden = FLAGS.rnn_embedding_dim, 
         n_time_step = FLAGS.item_windows_size, 
         learning_rate = FLAGS.learning_rate, 
         grad_clip = 0.2,
         emb_dim = FLAGS.mf_embedding_dim,
         lamda = FLAGS.lamda,
         initdelta = 0.05,
         MF_paras=paras,
         model_type=FLAGS.model_type,
         update_rule = 'adam'
         )
gen.build_pretrain()

dis = Dis(itm_cnt = helper.i_cnt, 
             usr_cnt = helper.u_cnt, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate = FLAGS.learning_rate, 
             grad_clip = 0.2,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = FLAGS.lamda,
             initdelta = 0.05,
             MF_paras=paras,
             model_type=FLAGS.model_type,
             update_rule = 'adam'
             )
dis.build_pretrain()

sess = tf.InteractiveSession()        
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=50)

sess.run(init)

#    checkpoint_filepath= "model/joint-25-0.24133.ckpt"
#    saver.restore(sess,checkpoint_filepath)
scores=helper.evaluateMultiProcess(sess, dis)
if FLAGS.model_type=="mf":
    best_p5=scores[1]
else:
    best_p5=scores[1][1]
print(scores)

df = helper.train

def main(checkpoint_dir="model/"):
    global best_p5
    for epoch in range(1000):
        if epoch > 0:
            for d_epoch in range(helper.conf.d_epoch_size):

                # for i,(uid,itemid,rating) in enumerate(helper.getBatch4MF()):
                rnn_losses,mf_losses,joint_losses=[],[],[]
                for i,(u_seqs,i_seqs,rating,uid,itemid) in enumerate(helper.getBatchFromSamples(dns=True,sess=sess,model=gen,fresh=False)):

                    _,loss_mf,loss_rnn,joint_loss,_ = dis.pretrain_step(sess,  rating, uid, itemid,u_seqs,i_seqs)
                    rnn_losses.append(loss_rnn)
                    mf_losses.append(loss_mf)
                    joint_losses.append(joint_loss)

                print(" rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(np.mean(np.array(rnn_losses)),np.mean(np.array(mf_losses)),np.mean(np.array(joint_losses))) )
                scores= (helper.evaluateMultiProcess(sess,dis))
                # print(helper.evaluateRMSE(sess,model))
                print(scores)
                if FLAGS.model_type=="mf":
                    curentt_p5_score=scores[1]
                else:
                    curentt_p5_score=scores[1][1]

                if curentt_p5_score >best_p5:        	
                    best_p5=curentt_p5_score
                    saver.save(sess, checkpoint_dir + '%s-%d-%.5f.ckpt'% (FLAGS.model_type,FLAGS.re_rank_list_length,best_p5))
#                    mf_model = '%s-%d-%.5f.pkl'% (FLAGS.model_type,FLAGS.re_rank_list_length,best_p5)
                    # model.saveMFModel(sess,mf_model)
#                    print(best_p5)


        for g_epoch in range(helper.conf.g_epoch_size):

            user = df["uid"][0]
            for user in df["uid"].unique():

#                pos_items_time_dict=helper.user_item_pos_rating_time_dict.get(user,{})
#                if len(pos_items_time_dict)==0:                   # todo  not do this
#                    continue
#               Generate movies candidates with MF
                all_rating = dis.predictionItems(sess,user)                           # todo delete the pos ones
                exp_rating = np.exp(np.array(all_rating) * helper.conf.temperature)
                prob = exp_rating / np.sum(exp_rating)
                
                sampled_items = np.random.choice(np.arange(helper.i_cnt), size=helper.conf.gan_k, p=prob)
                                
                samples=[]
                for item in sampled_items:     
                    #update G with unlabeled data                                                
                    #update G with labeled data         
                    u_seqs,i_seqs = helper.getSeqInTime(user,item,0)                   
                    labeled_row = df.loc[(df.uid==user) & (df.itemid==item)]                                                         
                    samples.append((u_seqs,i_seqs,user,item, (1 if len(labeled_row)>0 else 0), 
                                    int(labeled_row.rating if len(labeled_row)>0 else 0)))
                        
#                    if helper.conf.lastone:                                                                                  
#                    else:
#                        u_seqss,i_seqss= helper.getSeqOverAlltime(user,item)
#                        predicted = gen.prediction(sess,u_seqss,i_seqss, [user]*len(u_seqss),[item]*len(u_seqss),sparse=True)
#                        index=np.argmax(predicted)
#                        samples.append((u_seqss[index],i_seqss[index],user,item ))
                        
                rewards = dis.getRewards(sess,gen, samples, sparse=True)
                
                gen.unsupervised_train_step(sess,samples, rewards)
                
#                print("gan loss %.5f"%gen.unsupervised_train_step(sess1,samples, rewards))
                
            scores = (helper.evaluateMultiProcess(sess,gen))
            print(scores)            

if __name__== "__main__":
	main()