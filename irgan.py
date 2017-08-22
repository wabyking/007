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
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
             learning_rate = 0.01, 
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
         learning_rate = 0.01, 
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

checkpoint_filepath= "model/joint-25-0.27733.ckpt"
saver.restore(sess,checkpoint_filepath)

# checkpoint_filepath= "model/joint-25-0.21533.ckpt"
# saver.restore(sess,checkpoint_filepath)


# #scores=helper.evaluateMultiProcess(sess, dis)


#print(helper.evaluateMultiProcess(sess, dis))
#print(helper.evaluateMultiProcess(sess, gen))

#D[[ 0.23        0.20866667  0.21466667  0.2522202   0.24813455  0.3131026 ]
# [ 0.30666667  0.27733333  0.224       0.43249112  0.46714382  0.54298359]]
#G[[ 0.21222222  0.21133333  0.20633333  0.23679337  0.24926554  0.30595089]
# [ 0.31        0.26666667  0.22133333  0.44481547  0.46434018  0.54110146]]

#[[ 0.24666667  0.22733333  0.20333333  0.26392857  0.26178889  0.28930757]
# [ 0.20222222  0.224       0.21833333  0.21664232  0.27487287  0.40599028]]

#[[ 0.26111111  0.23333333  0.21        0.27076758  0.26552905  0.30000889]
# [ 0.16333333  0.154       0.16266667  0.18576331  0.20232514  0.29628877]]


# print(helper.evaluateMultiProcess(sess, dis))

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



#rnn loss : 0.68943 mf loss : 0.69388  : joint loss 124.76549
#pg loss : 123.84924 reward : 0.00130 

#rnn loss : 0.69394 mf loss : 0.69320 : joint loss 118.36721
#pg loss : 113.73331 reward : 0.00014 

df = helper.test
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def main(checkpoint_dir="model/"):
#        for i,  (u_seqs_pos,i_seqs_pos,ratings_pos,userids_pos,itemids_pos, 
#             u_seqs_neg,i_seqs_neg,ratings_neg,userids_neg,itemids_neg) in enumerate(helper.getBatchFromDNS(dns=True,sess=sess,model=gen,fresh=False)):                   
#            batchGenerator = helper.getBatchFromDNS(dns=True,sess=sess,model=gen,fresh=False)
#            (u_seqs,i_seqs,ratings,userids,itemids) = batchGenerator.next()
    #        (u_seqs_pos,i_seqs_pos,ratings_pos,userids_pos,itemids_pos, 
    #         u_seqs_neg,i_seqs_neg,ratings_neg,userids_neg,itemids_neg) = batchGenerator.next()
    

    #            user = df["uid"].unique()[np.random.randint(uid_cnt, size=1)[0]]
    ##            for user in df["uid"].unique():
    #            all_rating = dis.predictionItems(sess,user)                           # todo delete the pos ones
    #            exp_rating = np.exp(np.array(all_rating) * helper.conf.temperature)
    #            prob = exp_rating / np.sum(exp_rating)                
    #            sampled_items = np.random.choice(np.arange(helper.i_cnt), size=128, p=prob)

    for e in range(10):
            
        for g_epoch in range(20):    
            rewardes,pg_losses=[],[] 
            for user in df["uid"].unique():            
                # generate pesudo labels for the given user
                all_rating = gen.predictionItems(sess,user)                           # todo delete the pos ones            
                exp_rating = np.exp(np.array(all_rating) * 20)
                prob = exp_rating / np.sum(exp_rating)
#                    sorted(prob,reverse=True)[:10]
#                    negative_items_sampled = np.random.choice(np.arange(helper.i_cnt), size=32, p=prob)
                negative_items_sampled = np.argsort(prob)[::-1][:32]                                        
                
                negative_samples = []                  
                for item in negative_items_sampled:                                                
                    u_seqs,i_seqs = helper.getSeqInTime(user,item,0)   
                    negative_samples.append((u_seqs,i_seqs,user,item ))                        
                                                    
                u_seq_neg,i_seq_neg = [[ s[j].toarray()  for s in negative_samples ]  for j in range(2)]
                u_neg,i_neg = [[ s[j]  for s in negative_samples ]  for j in range(2,4)]
                unlabeled_rewards = dis.prediction(sess,u_seq_neg,i_seq_neg,u_neg,i_neg)
                unlabeled_rewards = [2* (sigmoid(v)-0.5) for v in unlabeled_rewards]

                positive_samples = []
                pos_items_time_dict = helper.user_item_pos_rating_time_dict.get(user,{})   
                if len(pos_items_time_dict)==0:
                    continue
                for ind in np.random.randint(len(pos_items_time_dict), size=32):                        
                    positive_item,t = pos_items_time_dict.items()[ind]                    
                    u_seqs,i_seqs = helper.getSeqInTime(user,positive_item,t)
                    positive_samples.append((u_seqs,i_seqs,user,positive_item))
              
                u_seq_pos,i_seq_pos = [[ s[j].toarray()  for s in positive_samples ]  for j in range(2)]
                u_pos,i_pos = [[ s[j]  for s in positive_samples ]  for j in range(2,4)]
                labeled_rewards = dis.prediction(sess,u_seq_pos,i_seq_pos,u_pos,i_pos)
                labeled_rewards = [2* (sigmoid(v)-0.5) + 0.1 for v in labeled_rewards]

                pg_loss = gen.unsupervised_train_step(sess, u_seq_neg + u_seq_pos,
                                                      i_seq_neg + i_seq_pos,
                                                      u_neg + u_pos,i_neg + i_pos, unlabeled_rewards + labeled_rewards)
                pg_losses.append(pg_loss)
                rewardes.append(unlabeled_rewards)     
            with open("test_lr.txt", "a") as myfile:
                myfile.write("pg loss : %.5f reward : %.5f "%(np.mean(np.array(pg_losses)),np.sum(np.array(rewardes)))+"\n")                                   
            print("pg loss : %.5f reward : %.5f "%(np.mean(np.array(pg_losses)),np.sum(np.array(rewardes))))
            
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
#                        positive_item,t = pos_items_time_dict.items()[np.random.randint(len(pos_items_time_dict), size=1)[0]]
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
        
        rnn_losses,mf_losses,joint_losses=[],[],[]
        user_item_neg_rating_time_dict = lambda group:{item:t for i,(item,t)  in group[group.rating<=2][["itemid","user_granularity"]].iterrows()}
        user_item_neg_rating_time_dict = helper.train.groupby("uid").apply(user_item_neg_rating_time_dict).to_dict()
    
        for user in df["uid"].unique():                        
            all_rating = gen.predictionItems(sess,user)                           # todo delete the pos ones            
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
                negative_item,t = neg_items_time_dict.items()[ind]                    
                u_seqs,i_seqs = helper.getSeqInTime(user,negative_item,t)
                negative_samples.append((u_seqs,i_seqs,user,negative_item,0))
            
            samples = pesudo_positive_samples + negative_samples
            u_seq,i_seq = [[ s[j].toarray()  for s in samples ]  for j in range(2)]
            u,i = [[ s[j]  for s in samples ]  for j in range(2,4)]
            ratings = [ s[4]  for s in samples ] 
            _,loss_mf,loss_rnn,joint_loss,rnn,mf = dis.pretrain_step(sess,ratings, u, i,u_seq,i_seq)                    
            rnn_losses.append(loss_rnn)
            mf_losses.append(loss_mf)
            joint_losses.append(joint_loss)
        print("rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(np.mean(np.array(loss_rnn)),np.mean(np.array(loss_mf)),np.mean(np.array(joint_loss))))            

        d = helper.evaluateMultiProcess(sess, dis)
        g = helper.evaluateMultiProcess(sess, gen)
        with open("test_lr.txt", "a") as myfile:
            myfile.write("\n".join(str(elem) for elem in d))
            myfile.write("\n")
            myfile.write("\n".join(str(elem) for elem in g))
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