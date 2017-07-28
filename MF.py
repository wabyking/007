import tensorflow as tf
import pickle as cPickle
import numpy as np
import config as conf
from dataHelper import DataHelper
import math
FLAGS=conf.getTestFlag()
helper=DataHelper(FLAGS)



class DIS():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.0001,loss_type="point",eta=0):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []
        self.loss_type=loss_type
        self.eta=eta
        with tf.variable_scope('discriminator'):
            if self.param == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
                self.user_bias = tf.Variable(tf.zeros([self.userNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(self.param[2])
                self.user_bias = tf.Variable(self.param[3])

        self.d_params = [self.user_embeddings, self.item_embeddings, self.item_bias,self.user_bias]

        # placeholder definition
        self.u = tf.placeholder(tf.int32,name="user_id")
        self.i = tf.placeholder(tf.int32,name="item_id_or_pos_id")
        
        self.label = tf.placeholder(tf.float32,name="label")

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
  
        self.i_bias = tf.gather(self.item_bias, self.i)
        self.u_bias = tf.gather(self.user_bias, self.u)
        
        # pair-wise if needed       
        self.j = tf.placeholder(tf.int32,name="neg_id")
        self.j_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.j)
        self.j_bias = tf.gather(self.item_bias, self.j)
        with tf.name_scope("pair-wise"):           
            self.y = tf.sigmoid(
                self.i_bias - self.j_bias + tf.reduce_sum(
                    tf.multiply(self.u_embedding, self.i_embedding - self.j_embedding), 1)
            )
            self.pair_loss = -tf.reduce_mean(tf.log(self.y)) 
        
        with tf.name_scope("point-wise"):
            self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias +self.u_bias
            # self.point_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,logits=self.pre_logits) )
            self.point_loss = tf.reduce_sum(tf.square(self.label-self.pre_logits) )
        
        if self.loss_type=="pair":
            self.loss= self.pair_loss
            print ("pair loss")
        elif self.loss_type=="point":
            self.loss= self.point_loss
            print ("point loss")
        else:
            self.pre_logits_pos = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
            self.pre_logits_neg = tf.reduce_sum(tf.multiply(self.u_embedding, self.j_embedding), 1) + self.j_bias
            self.batch_size=tf.shape(self.pre_logits_neg)[0]

            self.pre_logits= tf.concat([self.pre_logits_pos,self.pre_logits_neg],0)

            self.fusion_label=tf.concat([tf.ones([self.batch_size]),tf.zeros([self.batch_size])],0)
            self.V=self.fusion_label
            self.fusion_point_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.fusion_label ,logits=self.pre_logits) 
            print ("fusion loss")
            # self.loss= self.point_loss
            
            # self.point_loss_neg = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,logits=self.pre_logits)

            # self.point_loss_pos = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,logits=self.pre_logits)

            # self.loss=self.point_loss
            self.loss= self.eta* self.fusion_point_loss  + (1- self.eta)* self.pair_loss

        self.l2_loss=self.lamda * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias))
        self.pre_loss=self.loss+self.l2_loss

        # self.global_step = tf.get_variable('global_step_in_class', [], initializer=tf.constant_initializer(0), trainable=False)
        # trainer = tf.train.AdamOptimizer(self.conf.learning_rate, self.conf.momentum) 
        # gradients = trainer.compute_gradients(self.pre_loss)
        # clipped_gradients = [(tf.clip_by_value(_[0], -self.conf.grad_clip, self.conf.grad_clip), _[1]) for _ in gradients]
        # self.d_updates = trainer.apply_gradients(clipped_gradients,global_step= self.global_step)

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)

        # self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),
        #                                    1) + self.i_bias
        # self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)

        # # for test stage, self.u: [batch_size]
        # self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
        #                             transpose_b=True) + self.item_bias

        # self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        # self.NLL = -tf.reduce_mean(tf.log(
        #     tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))
        # )
        # # for dns sample
        # self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias

    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'wb'))


np.random.seed(70)
param = None
loss_type="point"
discriminator = DIS(helper.i_cnt, helper.u_cnt, helper.conf.mf_embedding_dim, lamda=0.2, param=param, initdelta=0.05, learning_rate=0.01,loss_type=loss_type)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver=tf.train.Saver()
  


def createModel( checkpoint_dir="model/"):
    best_score = 10

    with  tf.Session(config=config) as sess:   
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(5):
            for i,(uid,itemid,rating) in enumerate(helper.getBatch4MF()):

                feed_dict={discriminator.u: uid, discriminator.i: itemid,discriminator.label: rating}
                _, model_loss,l2_loss,pre_logits = sess.run([discriminator.d_updates,discriminator.point_loss,discriminator.l2_loss,discriminator.pre_logits],feed_dict=feed_dict)    

            train_performance=helper.testModel(sess,discriminator,flag="train")
            test_performance=helper.testModel(sess,discriminator,flag="test")

            print ("train set rmse = %.6f  test rmse = %.6f" % (train_performance,test_performance))


            if best_score> test_performance:

                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1) 
                best_score=test_performance


def testModel(checkpoint_dir="model/"):


    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path)  

        print ("rmse = %.6f" % helper.testModel(sess,discriminator,flag="test"))


def main():
    createModel()
    testModel()
if __name__ == '__main__':
    main()
