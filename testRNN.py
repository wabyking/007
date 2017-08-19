import pandas as pd 
import os
import numpy as np 
import pickle
from config import Singleton
import tensorflow as tf
from Discrimiator import Dis
from Generator import Gen

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



# waby = DIS(helper.i_cnt, helper.u_cnt, FLAGS.mf_embedding_dim, lamda=0.1, param=None, initdelta=0.05, learning_rate=0.001)

# tf.get_variable_scope().reuse_variables()    

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

# checkpoint_filepath= "model/joint-25-0.28000.ckpt"
# saver.restore(sess,checkpoint_filepath)
# model.saveModel(sess,"rnn.model",save_type="rnn")

#scores=helper.evaluateMultiProcess(sess,dis)
#if FLAGS.model_type=="mf":
#    best_p5=scores[1]
#else:
#    best_p5=scores[1][1]
#print(scores)

#global best_p5

def main(checkpoint_dir="model/"):
    best_p5 = 0
    for epoch in range(1000):
        rnn_losses=[]
        mf_losses=[]
        joint_losses=[]
        for i,(u_seqs,i_seqs,rating,uid,itemid) in enumerate(helper.getBatchFromSamples(dns=FLAGS.dns,sess=sess,model=dis,fresh=False)):


            # feed_dict={discriminator.u: uid, discriminator.i: itemid,discriminator.label: rating}
            # _, model_loss,l2_loss,pre_logits = sess.run([discriminator.d_updates,discriminator.point_loss,discriminator.l2_loss,discriminator.pre_logits],feed_dict=feed_dict)    

            # _,l,pre_logits_MF = model.pretrain_step(sess, (np.array(rating)>3.99).astype("int32"), uid, itemid)
            # print(u_seqs,i_seqs,rating,uid,itemid)
   
            _,loss_mf,loss_rnn,joint_loss = dis.pretrain_step(sess, rating, uid, itemid, u_seqs, i_seqs)
            rnn_losses.append(loss_rnn)
            mf_losses.append(loss_mf)
            joint_losses.append(joint_loss)                        

        # print(sess.run(model.user_embeddings))
        # print(sess.run(model.item_embeddings))
        # print(sess.run(model.item_bias))
        # print(sess.run(model.user_bias))
        print(" rnn loss : %.5f mf loss : %.5f  : joint loss %.5f" %(np.mean(np.array(rnn_losses)),np.mean(np.array(mf_losses)),np.mean(np.array(joint_losses))) )
        scores = (helper.evaluateMultiProcess(sess, dis))
        # print(helper.evaluateRMSE(sess,model))
        print(scores)

        if FLAGS.model_type == "mf":
            curentt_p5_score = scores[1]
        else:
            curentt_p5_score = scores[1][1]

        if curentt_p5_score > best_p5:        	
            best_p5 = curentt_p5_score
            saver.save(sess, checkpoint_dir + '%s-%d-%.5f.ckpt'% (FLAGS.model_type,FLAGS.re_rank_list_length,best_p5))
#            helper.create_dirs("model/mf")
#            mf_model = 'model/mf/%s-%d-%.5f.pkl'% (FLAGS.model_type,FLAGS.re_rank_list_length,best_p5)
#            dis.saveMFModel(sess,mf_model)
#            print(best_p5)

        # print("non multiprocess evalution have spent %f s"%(time.time()-start) )


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
	main()