import pandas as pd 
import os
import numpy as np 
import pickle
from config import Singleton
import tensorflow as tf
from Discrimiator import Dis
from Generator import Gen
import time
from dataHelper import FLAGS,helper


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

if os.path.exists(FLAGS.pretrained_model) and FLAGS.pretrained:
	print("Fineutune the discrimiator with pretrained MF named " + FLAGS.pretrained_model)
	paras= pickle.load(open(FLAGS.pretrained_model,"rb"))
else:
	print("Fail to load pretrained MF model ")
	paras=None

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
             model_type=FLAGS.model_type,
             update_rule = 'sgd',
             use_sparse_tensor=FLAGS.sparse_tensor,
             pairwise=FLAGS.pairwise
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
         update_rule = 'sgd',
         use_sparse_tensor=FLAGS.sparse_tensor
         )
gen.build_pretrain()

sess = tf.InteractiveSession()    
saver = tf.train.Saver(max_to_keep=40) 
tf.global_variables_initializer().run()

# checkpoint_dir="model/"
 
# # saver.save(sess, checkpoint_dir + 'model.ckpt')


# ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
# if ckpt and ckpt.model_checkpoint_path: 
#     print("load model") 
#     saver.restore(sess, ckpt.model_checkpoint_path)
# print(sess.run(model.user_embeddings)[0])
# print(sess.run(model.user_embeddings)[0])

# model.restoreModel("mf.model",save_type="mf")


# checkpoint_filepath= "model/joint-25-0.27733.ckpt"
# saver.restore(sess,checkpoint_filepath)

# model.saveModel(sess,"rnn.model",save_type="rnn")

# print(helper.evaluateMultiProcess(sess, dis))
# print(helper.evaluateMultiProcess(sess, gen))
#scores=helper.evaluateMultiProcess(sess,dis)
#if FLAGS.model_type=="mf":
#    best_p5=scores[1]
#else:
#    best_p5=scores[1][1]
#print(scores)

#global best_p5

def main():
    checkpoint_dir="model/"
    best_p5 = 0
    for epoch in range(1000):


        rnn_losses_d, mf_losses_d, joint_losses_d = [],[],[]
        for i,(u_seqs,i_seqs,rating,uid,itemid) in enumerate(helper.getBatchFromSamples_point(dns=FLAGS.dns,sess=sess,model=None,fresh=False)):


            _,loss_mf_g,loss_rnn_g,joint_loss_g = gen.pretrain_step(sess, rating, uid, itemid, u_seqs, i_seqs)
            _,loss_mf_d,loss_rnn_d,joint_loss_d,rnn,mf = dis.pretrain_step(sess, rating, uid, itemid, u_seqs, i_seqs)            
            

            rnn_losses_d.append(loss_rnn_d)
            mf_losses_d.append(loss_mf_d)
            joint_losses_d.append(joint_loss_d)

        print(" rnn loss : %.5f mf loss : %.5f  : joint loss %.5f" %
              (np.mean(np.array(rnn_losses_d)),np.mean(np.array(mf_losses_d)),np.mean(np.array(joint_losses_d))) )


        scores = (helper.evaluateMultiProcess(sess, dis))
        print(scores)
        scores = (helper.evaluateMultiProcess(sess, gen))
        print(scores)



#        # print(helper.evaluateRMSE(sess,model))        
        if FLAGS.model_type == "mf":
           curentt_p5_score = scores[1]
        else:
           curentt_p5_score = scores[1][1]
        saver.save(sess, checkpoint_dir + '%s-%d-%.5f.ckpt'% (FLAGS.model_type,FLAGS.re_rank_list_length,curentt_p5_score))
#        if curentt_p5_score > best_p5:        	
#            best_p5 = curentt_p5_score
        
#            helper.create_dirs("model/mf")
#            mf_model = 'model/mf/%s-%d-%.5f.pkl'% (FLAGS.model_type,FLAGS.re_rank_list_length,best_p5)
#            dis.saveMFModel(sess,mf_model)
#            print(best_p5)

        # print("non multiprocess evalution have spent %f s"%(time.time()-start) )
def  pairtrain():
    print (helper.evaluateMultiProcess(sess, dis))
    joint_losses=[]
    start=time.time() 
    for epoch in range(500):
        for i,((user,u_seqs,item,i_seqs,item_neg,i_seqs_neg)) in enumerate(helper.getBatchFromSamples_pair(dns=FLAGS.dns,sess=sess,model=dis,fresh=False)):

            _,joint_loss = dis.pretrain_step_pair(sess, user,u_seqs,item,i_seqs,item_neg,i_seqs_neg)     

            joint_losses.append(joint_loss) 
        print("mean loss = %.5f"% np.mean(joint_loss))
        scores = (helper.evaluateMultiProcess(sess, dis))
            # print(helper.evaluateRMSE(sess,model))
        print(scores)

        
def analysisData():
	datas=[]
	for i,(uids,itemids,ratings) in enumerate(helper.getBatch4MF()):
		for uid,itemid,rating in zip(uids,itemids,ratings):
			line="%d\t%d\t%d" %(uid,itemid,rating)
			datas.append(line)
	with open("a.txt","w") as f:
		f.write("\n".join(datas))
	datas=[]

	for i,(_,_,ratings,uids,itemids) in enumerate(helper.prepare()):
		for uid,itemid,rating in zip(uids,itemids,ratings):
			line="%d\t%d\t%d" %(uid,itemid,rating)
			datas.append(line)

	with open("b.txt","w") as f:
		f.write("\n".join(datas))
             
if __name__== "__main__":
    if helper.conf.pairwise:
        pairtrain()
    else:
        main()