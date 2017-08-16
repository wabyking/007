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


exit()
def main(checkpoint_dir="model/"):
    global best_p5
    for epoch in range(1000):

        # for i,(uid,itemid,rating) in enumerate(helper.getBatch4MF()):
        rnn_losses=[]
        mf_losses=[]
        joint_losses=[]
        for i,(u_seqs,i_seqs,rating,uid,itemid) in enumerate(helper.prepare(rating_flag=False)):

            # feed_dict={discriminator.u: uid, discriminator.i: itemid,discriminator.label: rating}
            # _, model_loss,l2_loss,pre_logits = sess.run([discriminator.d_updates,discriminator.point_loss,discriminator.l2_loss,discriminator.pre_logits],feed_dict=feed_dict)    

            # _,l,pre_logits_MF = model.pretrain_step(sess, (np.array(rating)>3.99).astype("int32"), uid, itemid)
            # print(u_seqs,i_seqs,rating,uid,itemid)
            _,loss_mf,loss_rnn,joint_loss= model.pretrain_step(sess,  rating, uid, itemid,u_seqs,i_seqs)
            rnn_losses.append(loss_rnn)
            mf_losses.append(loss_mf)
            joint_losses.append(joint_loss)

            # print( "MF loss: %.5f  RNN loss : %.5f"%(loss_mf,loss_rnn))
            # print(l)
            # print(sess.run(model.user_embeddings)[0])
            # print(  "loss %f  logits %s" %(l,str(pre_logits_MF)))
        # start=time.time()
        # print("%d epoch" % epoch)
        # print(sess.run(model.user_embeddings))
        # print(sess.run(model.item_embeddings))
        # print(sess.run(model.item_bias))
        # print(sess.run(model.user_bias))
       	# print(i)
        print(" rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(np.mean(np.array(rnn_losses)),np.mean(np.array(mf_losses)),np.mean(np.array(joint_losses))) )
        scores= (helper.evaluateMultiProcess(sess,model))
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
            model.saveMFModel(sess,mf_model)
            print(best_p5)

        # print("non multiprocess evalution have spent %f s"%(time.time()-start) )


             
if __name__== "__main__":
	main()