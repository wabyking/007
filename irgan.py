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
	print("Fineutune the discrimiator with pretrained MF named " + FLAGS.pretrained_model)
	paras= pickle.load(open(FLAGS.pretrained_model,"rb"))
else:
	print("Fail to load pretrained MF model ")
	paras=None

#g1 = tf.Graph()
#g2 = tf.Graph()
#sess1 = tf.InteractiveSession(graph=g1)
#sess2 = tf.InteractiveSession(graph=g2)

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
             update_rule = 'sgd'
             )
dis.build_pretrain()

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
         update_rule = 'sgd'
         )
gen.build_pretrain()

sess = tf.InteractiveSession()    
saver = tf.train.Saver(max_to_keep=40) 
tf.global_variables_initializer().run()

#checkpoint_filepath= "model/joint_g_d/joint-25-0.20000.ckpt"
checkpoint_filepath= "model/joint-25-0.21533.ckpt"
saver.restore(sess,checkpoint_filepath)

#scores=helper.evaluateMultiProcess(sess, dis)

print(helper.evaluateMultiProcess(sess, dis))
print(helper.evaluateMultiProcess(sess, gen))
#[[ 0.24111111  0.23333333  0.212       0.25956044  0.26670492  0.3048191 ]
# [ 0.19444444  0.21533333  0.215       0.21885524  0.27984188  0.41407132]]
#[[ 0.02333333  0.026       0.025       0.03447644  0.05083666  0.07598918]
# [ 0.02444444  0.02666667  0.02566667  0.04551901  0.06735523  0.10214565]]

#if FLAGS.model_type=="mf":
#    best_p5=scores[1]
#else:
#    best_p5=scores[1][1]
#print(scores)

#[[ 0.20888889  0.18466667  0.157       0.24437542  0.24719717  0.28737471]
# [ 0.14444444  0.14533333  0.14033333  0.17172352  0.21879874  0.32662828]]
#[[ 0.21111111  0.18466667  0.16        0.23470304  0.23428751  0.27704266]
# [ 0.21111111  0.208       0.18033333  0.25352937  0.2968801   0.38834072]]

#[[ 0.06222222  0.058       0.047       0.08622233  0.10302443  0.13400635]
# [ 0.04333333  0.02933333  0.02166667  0.0707058   0.07538342  0.09211432]]
#[[ 0.21111111  0.18666667  0.161       0.23548052  0.23594752  0.27933239]
# [ 0.22666667  0.21066667  0.18166667  0.27080088  0.30230874  0.39421854]]

#rnn loss : 0.69211 mf loss : 0.69347  : joint loss 122.25066
#pg loss : 120.65841 reward : -0.98438 

df = helper.test

uid_cnt = len(df["uid"].unique())


#rnn loss : 0.68943 mf loss : 0.69388  : joint loss 124.76549
#pg loss : 123.84924 reward : 0.00130 

#rnn loss : 0.69394 mf loss : 0.69320 : joint loss 118.36721
#pg loss : 113.73331 reward : 0.00014 

def main(checkpoint_dir="model/"):
    for e in range(100):
        
        rnn_losses,mf_losses,joint_losses=[],[],[]
         
#        for i,  (u_seqs_pos,i_seqs_pos,ratings_pos,userids_pos,itemids_pos, 
#             u_seqs_neg,i_seqs_neg,ratings_neg,userids_neg,itemids_neg) in enumerate(helper.getBatchFromDNS(dns=True,sess=sess,model=gen,fresh=False)):       
        for i, (u_seqs,i_seqs,ratings,userids,itemids) in enumerate(helper.getBatchFromDNS(dns=True,sess=sess,model=gen,fresh=False)):           
#            batchGenerator = helper.getBatchFromDNS(dns=True,sess=sess,model=gen,fresh=False)
#            (u_seqs,i_seqs,ratings,userids,itemids) = batchGenerator.next()
    #        (u_seqs_pos,i_seqs_pos,ratings_pos,userids_pos,itemids_pos, 
    #         u_seqs_neg,i_seqs_neg,ratings_neg,userids_neg,itemids_neg) = batchGenerator.next()
    
            for g_epoch in range(1):            
    #            user = df["uid"].unique()[np.random.randint(uid_cnt, size=1)[0]]
    ##            for user in df["uid"].unique():
    #            all_rating = dis.predictionItems(sess,user)                           # todo delete the pos ones
    #            exp_rating = np.exp(np.array(all_rating) * helper.conf.temperature)
    #            prob = exp_rating / np.sum(exp_rating)                
    #            sampled_items = np.random.choice(np.arange(helper.i_cnt), size=128, p=prob)
            for g_epoch in range(10):   
                rewardes,pg_losses=[],[]     
                for user in df["uid"].unique():            
                    # generate pesudo labels for the given user
                    all_rating = gen.predictionItems(sess,user)                           # todo delete the pos ones            
                    exp_rating = np.exp(np.array(all_rating) * helper.conf.temperature)
                    prob = exp_rating / np.sum(exp_rating)
                   
#                    negative_items_sampled = np.random.choice(np.arange(helper.i_cnt), size=10, p=prob)
                    negative_items_sampled = np.argsort(prob)[::-1][:10]
                    
                    negative_samples = []
                    for item in negative_items_sampled:                                                
                        u_seqs,i_seqs = helper.getSeqInTime(user,item,0)   
                        negative_samples.append((u_seqs,i_seqs,user,item ))                        
                    
                    u_seq_neg,i_seq_neg = [[ s[j].toarray()  for s in negative_samples ]  for j in range(2)]
                    u_neg,i_neg = [[ s[j]  for s in negative_samples ]  for j in range(2,4)]
                    unlabeled_rewards = dis.prediction(sess,u_seq_neg,i_seq_neg,u_neg,i_neg)
                    unlabeled_rewards = [2* (sigmoid(v)-0.5) for v in unlabeled_rewards]
                    
                    pg_loss = gen.unsupervised_train_step(sess, u_seq_neg,i_seq_neg,u_neg,i_neg, unlabeled_rewards)
                    pg_losses.append(pg_loss)
                    rewardes.append(unlabeled_rewards)                    
                    
                print("pg loss : %.5f reward : %.5f "%(np.mean(np.array(pg_losses)),np.mean(np.array(rewardes))))
    #            samples=[]
    #            for item in sampled_items:           
    #                u_seqs,i_seqs = helper.getSeqInTime(user,item,0)                   
    #                labeled_row = df.loc[(df.uid==user) & (df.itemid==item)]                                                       
    #                samples.append((u_seqs,i_seqs,user,item, (1 if len(labeled_row)>0 else 0), 
    #                                int(labeled_row.rating if len(labeled_row)>0 else 0)))
    #            rewards = dis.getRewards(sess,gen, samples, sparse=True)           
    
    #            labeled_rewards = np.zeros(len(samples))
    #            return 2 * (self.sigmoid(unlabeled_rewards) - 0.5)
                    
                
#            _,loss_mf,loss_rnn,joint_loss,rnn,mf = dis.pretrain_step(sess,ratings_pos + ratings_neg, 
#                                                                     userids_pos + userids_neg, 
#                                                                     itemids_pos + itemids_neg,
#                                                                      u_seqs_pos + u_seqs_neg,
#                                                                      i_seqs_pos + i_seqs_neg)        
#            rnn_losses.append(loss_rnn)
#            mf_losses.append(loss_mf)
#            joint_losses.append(joint_loss)

    #                    if helper.conf.lastone:                                                                                  
    #                    else:
    #                        u_seqss,i_seqss= helper.getSeqOverAlltime(user,item)
    #                        predicted = gen.prediction(sess,u_seqss,i_seqss, [user]*len(u_seqss),[item]*len(u_seqss),sparse=True)
    #                        index=np.argmax(predicted)
    #                        samples.append((u_seqss[index],i_seqss[index],user,item ))
#        print("epoch : %d rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(e,np.mean(np.array(loss_rnn)),np.mean(np.array(loss_mf)),np.mean(np.array(joint_loss))))
        print("pg loss : %.5f reward : %.5f "%(np.mean(np.array(pg_losses)),np.mean(np.array(rewardes))))
        
#        print(helper.evaluateMultiProcess(sess, dis))
        print(helper.evaluateMultiProcess(sess, gen))

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
if __name__== "__main__":
	main()