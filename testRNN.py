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



g1 = tf.Graph()
g2 = tf.Graph()
sess1 = tf.InteractiveSession(graph=g1)        
sess2 = tf.InteractiveSession(graph=g2)


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


for sess_,model,init,saver in zip([sess1,sess2],[gen,dis],[init1,init2],[saver1,saver2]):
    sess_.run(init)
    

#    checkpoint_filepath= "model/joint-25-0.28000.ckpt"
#    saver.restore(sess,checkpoint_filepath)
    

    
#sess = tf.InteractiveSession()    
#saver = tf.train.Saver(max_to_keep=40) 
#tf.global_variables_initializer().run()

# checkpoint_dir="model/"
 
# # saver.save(sess, checkpoint_dir + 'model.ckpt')


# ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
# if ckpt and ckpt.model_checkpoint_path: 
#     print("load model") 
#     saver.restore(sess, ckpt.model_checkpoint_path)
# print(sess.run(model.user_embeddings)[0])
# print(sess.run(model.user_embeddings)[0])

# model.restoreModel("mf.model",save_type="mf")


#checkpoint_filepath= "model/joint-25-0.27733.ckpt"
#saver.restore(sess,checkpoint_filepath)

# model.saveModel(sess,"rnn.model",save_type="rnn")

# print(helper.evaluateMultiProcess(sess, dis))
# print(helper.evaluateMultiProcess(sess, gen))
#scores=helper.evaluateMultiProcess(sess,dis)



def main(sess,model):
    
    scores=helper.evaluateMultiProcess(sess,model)
    if FLAGS.model_type=="mf":
        best_p5=scores[1]
    else:
        best_p5=scores[1][1]
    print(scores)
    
    best_p5 = 0
    print ("training " +("Dis" if isinstance(model ,Dis) else "Gen") +" model")
    for epoch in range(100):

        rnn_losses_g, mf_losses_g, joint_losses_g = [],[],[]

        for i,(u_seqs,i_seqs,rating,uid,itemid) in enumerate(helper.getBatchFromSamples_point(dns=FLAGS.dns,sess=sess,model=None,fresh=False)):
   
            _,loss_mf_g,loss_rnn_g,joint_loss_g ,mf,rnn= model.pretrain_step(sess, rating, uid, itemid, u_seqs, i_seqs)            
           
            
            rnn_losses_g.append(loss_rnn_g)
            mf_losses_g.append(loss_mf_g)
            joint_losses_g.append(joint_loss_g)

            
        print(" rnn loss : %.5f mf loss : %.5f  : joint loss %.5f" %
              (np.mean(np.array(rnn_losses_g)),np.mean(np.array(mf_losses_g)),np.mean(np.array(joint_losses_g))) )


        scores = (helper.evaluateMultiProcess(sess, model))
        print(scores)


       
        if FLAGS.model_type == "mf":
            curentt_p5_score = scores[1]
        else:
            curentt_p5_score = scores[1][1]

        if curentt_p5_score > best_p5:        	
            best_p5 = curentt_p5_score
            print("best p5 score %5.f"% best_p5)
            checkpoint_dir=    "model/" +("Dis" if isinstance(model ,Dis) else "Gen") +"/"
            helper.create_dirs(checkpoint_dir)
            
            saver.save(sess, checkpoint_dir + '%s-%d-%.5f-%.5f.ckpt'% (FLAGS.model_type,FLAGS.re_rank_list_length,scores[0][1], scores[1][1]))

#            helper.create_dirs("model/mf")
#            mf_model = 'model/mf/%s-%d-%.5f.pkl'% (FLAGS.model_type,FLAGS.re_rank_list_length,best_p5)
#            dis.saveMFModel(sess,mf_model)
#            print(best_p5)

        # print("non multiprocess evalution have spent %f s"%(time.time()-start) )
def  pairtrain():
    print (helper.evaluateMultiProcess(sess, dis))
    joint_losses=[]

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
#        main(sess1,gen)
        main(sess2,dis)
