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

from dataHelper import FLAGS,helper
#from oldMF import DIS
import time
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


g1 = tf.Graph()
g2 = tf.Graph()
sess1 = tf.InteractiveSession(graph=g1)        
sess2 = tf.InteractiveSession(graph=g2)

paras=None
with g1.as_default():
    gen = Gen(itm_cnt = helper.i_cnt, 
             usr_cnt = helper.u_cnt, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate =  FLAGS.learning_rate, 
             grad_clip = 0.2,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = FLAGS.lamda,
             initdelta = 0.05,
             MF_paras=paras,
             model_type=FLAGS.model_type,
             update_rule = 'sgd',
             use_sparse_tensor=FLAGS.sparse_tensor
             )
    gen.build_pretrain()
    init1=tf.global_variables_initializer()
    saver1 = tf.train.Saver(max_to_keep=50)
    sess1.run(init1)
#    checkpoint_filepath= "model/joint-25-0.25933.ckpt"
#    saver1.restore(sess1,checkpoint_filepath)
    
with g2.as_default():
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
             update_rule = 'sgd',
             use_sparse_tensor=FLAGS.sparse_tensor
             )
    dis.build_pretrain()
    init2=tf.global_variables_initializer()
    saver2 = tf.train.Saver(max_to_keep=50)
    sess2.run(init2)
    checkpoint_filepath= "model/Dis/joint-25-0.26067-0.28800.ckpt"
    saver2.restore(sess2,checkpoint_filepath)


print(helper.evaluateMultiProcess(sess2, dis))
print(helper.evaluateMultiProcess(sess1, gen))

setting =False

df = helper.test
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def main(checkpoint_dir="model/"):


    for e in range(20):
            
        for g_epoch in range(100):    

            rewardes,pg_losses=[],[] 
            for user in df["uid"].unique():            
                # generate pesudo labels for the given user
                all_rating = gen.predictionItems(sess1,user)                           # todo delete the pos ones            

                exp_rating = np.exp(np.array(all_rating) * 50)
                prob = exp_rating / np.sum(exp_rating)
#                    sorted(prob,reverse=True)[:10]
                negative_items_sampled = np.random.choice(np.arange(helper.i_cnt), size=32, p=prob)
#                negative_items_sampled = np.argsort(prob)[::-1][:32]                                        
                
                negative_samples = []
                unlabeled_rewards=[]
                u_seq_neg=[]
                i_seq_neg = []
                u_neg=[]
                i_neg=[]
                if setting==True: 
                    for item in negative_items_sampled:                                                
                        u_seqs,i_seqs = helper.getSeqInTime(user,item,0)   
                        negative_samples.append((u_seqs,i_seqs,user,item ))                        
                                                        
                    u_seq_neg,i_seq_neg = [[ s[j].toarray()  for s in negative_samples ]  for j in range(2)]
                    u_neg,i_neg = [[ s[j]  for s in negative_samples ]  for j in range(2,4)]
                    unlabeled_rewards =  np.array(dis.prediction(sess2,u_seq_neg,i_seq_neg,u_neg,i_neg,sparse=False))
                    unlabeled_rewards = list((unlabeled_rewards-np.mean(unlabeled_rewards))/np.std(unlabeled_rewards))
                    #unlabeled_rewards = [2* (sigmoid(v)-0.5) for v in unlabeled_rewards]

                positive_samples=[]
                labeled_rewards=[]
                u_seq_pos=[]
                i_seq_pos = []
                u_pos=[]
                i_pos=[]
                if setting==False:
                    pos_items_time_dict = helper.user_item_pos_rating_time_dict.get(user,{})   
                    if len(pos_items_time_dict)==0:
                        continue
                    for ind in np.random.randint(len(pos_items_time_dict), size=32):                        
                        positive_item,t = list(pos_items_time_dict.items())[ind]                    
                        u_seqs,i_seqs = helper.getSeqInTime(user,positive_item,t)
                        positive_samples.append((u_seqs,i_seqs,user,positive_item))
                  
                    u_seq_pos,i_seq_pos = [[ s[j].toarray()  for s in positive_samples ]  for j in range(2)]
                    u_pos,i_pos = [[ s[j]  for s in positive_samples ]  for j in range(2,4)]
                    labeled_rewards = dis.prediction(sess2,u_seq_pos,i_seq_pos,u_pos,i_pos,sparse=False)
    #                labeled_rewards = [2* (sigmoid(v)-0.5) + 0.1 for v in labeled_rewards]
                    labeled_rewards=list((labeled_rewards-np.mean(labeled_rewards))/np.std(labeled_rewards)) 
#                    pg_loss = gen.unsupervised_train_step(sess, u_seq_neg,i_seq_neg,u_neg,i_neg, unlabeled_rewards)
                pg_loss = gen.unsupervised_train_step(sess1, u_seq_neg + u_seq_pos,
                                                      i_seq_neg + i_seq_pos,
                                                      u_neg + u_pos,i_neg + i_pos, unlabeled_rewards + labeled_rewards)
                pg_losses.append(pg_loss)
                rewardes.extend(unlabeled_rewards)     
#            with open("test_lr.txt", "a") as myfile:
#                myfile.write("pg loss : %.5f reward : %.5f "%(np.mean(np.array(pg_losses)),np.sum(np.array(rewardes)))+"\n")                                   
            if g_epoch % 3 == 0:
                print("pg loss : %.5f reward : %.5f "%(np.mean(np.array(pg_losses)),np.sum(np.array(rewardes))))
            g = helper.evaluateMultiProcess(sess1, gen)    
            print (g)
#            for d_epoch in range(2): 
#                rnn_losses,mf_losses,joint_losses=[],[],[]
#                positive_samples = []
#                negative_samples = []                    
#                for user in df["uid"].unique():
#                    pos_items_time_dict = helper.user_item_pos_rating_time_dict.get(user,{})   # null 
#                    if len(pos_items_time_dict)==0:
#                        continue
#        
#                    # generate pesudo labels for the given user
#                    all_rating = gen.predictionItems(sess,user)
#                    exp_rating = np.exp(np.array(all_rating) * helper.conf.temperature)
#                    prob = exp_rating / np.sum(exp_rating)
#                    
##                    negative_items_argmax = np.argsort(prob)[::-1][:16]    
#                    negative_items_sampled = np.random.choice(np.arange(helper.i_cnt), size=32, p=prob)                                
#                    for item in negative_items_sampled:                    
#                        # the pesudo labels are regarded as high-quality negative labels but at the beginning, the pesudo labels are very low-quality ones.
#                        u_seqs,i_seqs = helper.getSeqInTime(user,item,0)   
#                        negative_samples.append((u_seqs,i_seqs,user,item,0 ))
#                        
#                        # sample positive examples in a random manner
#                        positive_item,t = list(pos_items_time_dict.items())[np.random.randint(len(pos_items_time_dict), size=1)[0]]
#                        u_seqs,i_seqs = helper.getSeqInTime(user,positive_item,t)
#                        positive_samples.append((u_seqs,i_seqs,user,positive_item,1))
#                    samples = negative_samples + positive_samples
#                    
#                    random.shuffle(samples)
#                    
#                    u_seq,i_seq = [[ s[j].toarray()  for s in samples ]  for j in range(2)]
#                    u,i = [[ s[j]  for s in samples ]  for j in range(2,4)]
#                    ratings = [ s[4]  for s in samples ]    
#                    _,loss_mf,loss_rnn,joint_loss,rnn,mf = dis.pretrain_step(sess,ratings, u, i,u_seq,i_seq)
#                    rnn_losses.append(loss_rnn)
#                    mf_losses.append(loss_mf)
#                    joint_losses.append(joint_loss)
#                with open("test1.txt", "a") as myfile:
#                    myfile.write("rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(np.mean(np.array(loss_rnn)),np.mean(np.array(loss_mf)),np.mean(np.array(joint_loss)))+"\n")
#                print("rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(np.mean(np.array(loss_rnn)),np.mean(np.array(loss_mf)),np.mean(np.array(joint_loss))))        
#            
#            d = helper.evaluateMultiProcess(sess, dis)
#            g = helper.evaluateMultiProcess(sess, gen)
#            with open("test1.txt", "a") as myfile:
#                myfile.write("\n".join(str(elem) for elem in d))
#                myfile.write("\n")
#                myfile.write("\n".join(str(elem) for elem in g))
#            print(d)
#            print(g)
        for d_epoch in range(3):  
            rnn_losses,mf_losses,joint_losses=[],[],[]
            user_item_neg_rating_time_dict = lambda group:{item:t for i,(item,t)  in group[group.rating<=2][["itemid","user_granularity"]].iterrows()}
            user_item_neg_rating_time_dict = helper.train.groupby("uid").apply(user_item_neg_rating_time_dict).to_dict()
        
            for user in df["uid"].unique():                        
                all_rating = gen.predictionItems(sess1,user)                           # todo delete the pos ones            
                exp_rating = np.exp(np.array(all_rating) * 50)
                prob = exp_rating / np.sum(exp_rating)
                   
                pesudo_positive_items = np.argsort(prob)[::-1][:32]
                
                pesudo_positive_samples = []                  
                for item in pesudo_positive_items:                                                
                    u_seqs,i_seqs = helper.getSeqInTime(user,item,0)   
                    pesudo_positive_samples.append((u_seqs,i_seqs,user,item,1))                        
                
                negative_samples = []
                neg_items_time_dict = user_item_neg_rating_time_dict.get(user,{})   
                if len(neg_items_time_dict)==0:
                    continue                
                for ind in np.random.randint(len(neg_items_time_dict), size=32):                        
                    negative_item,t = list(neg_items_time_dict.items())[ind]                    
                    u_seqs,i_seqs = helper.getSeqInTime(user,negative_item,t)
                    negative_samples.append((u_seqs,i_seqs,user,negative_item,0))
                
                samples = pesudo_positive_samples + negative_samples
                u_seq,i_seq = [[ s[j].toarray()  for s in samples ]  for j in range(2)]
                u,i = [[ s[j]  for s in samples ]  for j in range(2,4)]
                ratings = [ s[4]  for s in samples ] 
                _,loss_mf,loss_rnn,joint_loss,rnn,mf = dis.pretrain_step(sess1,ratings, u, i,u_seq,i_seq)                    
                rnn_losses.append(loss_rnn)
                mf_losses.append(loss_mf)
                joint_losses.append(joint_loss)
            if d_epoch % 3 == 0:
                print("rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(np.mean(np.array(loss_rnn)),np.mean(np.array(loss_mf)),np.mean(np.array(joint_loss))))            

        d = helper.evaluateMultiProcess(sess2, dis)
        
#        with open("test_lr.txt", "a") as myfile:
#            myfile.write("\n".join(str(elem) for elem in d))
#            myfile.write("\n")
#            myfile.write("\n".join(str(elem) for elem in g))
        print(d)
        print(g)
#            for i, (u_seqs,i_seqs,ratings,userids,itemids) in enumerate(helper.getBatchFromDNS(dns=True,sess=sess,model=gen,fresh=False)):
#                _,loss_mf,loss_rnn,joint_loss,rnn,mf = dis.pretrain_step(sess,ratings, userids, itemids,u_seqs,i_seqs)                    
#                rnn_losses.append(loss_rnn)
#                mf_losses.append(loss_mf)
#                joint_losses.append(joint_loss)
#            print("rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(np.mean(np.array(loss_rnn)),np.mean(np.array(loss_mf)),np.mean(np.array(joint_loss))))        
#            

#                    if helper.conf.lastone:                                                                                  
#                    else:
#                        u_seqss,i_seqss= helper.getSeqOverAlltime(user,item)
#                        predicted = gen.prediction(sess,u_seqss,i_seqss, [user]*len(u_seqss),[item]*len(u_seqss),sparse=True)
#                        index=np.argmax(predicted)
#                        samples.append((u_seqss[index],i_seqss[index],user,item ))
#            samples=[]
#            for item in sampled_items:           
#                u_seqs,i_seqs = helper.getSeqInTime(user,item,0)                   
#                labeled_row = df.loc[(df.uid==user) & (df.itemid==item)]                                                       
#                samples.append((u_seqs,i_seqs,user,item, (1 if len(labeled_row)>0 else 0), 
#                                int(labeled_row.rating if len(labeled_row)>0 else 0)))
#            rewards = dis.getRewards(sess,gen, samples, sparse=True)           

#            labeled_rewards = np.zeros(len(samples))
#            return 2 * (self.sigmoid(unlabeled_rewards) - 0.5)        


if __name__== "__main__":
    main()
