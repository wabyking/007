import tensorflow as tf
import pickle as cPickle


class DIS():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []
       

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
        
      
        with tf.name_scope("point-wise"):
            self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias +self.u_bias
            self.point_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,logits=self.pre_logits) )
        
        
        self.loss=self.point_loss
        self.l2_loss=self.lamda * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias))
        self.pre_loss=self.point_loss+self.l2_loss


        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)

        self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),
                                           1) + self.i_bias
        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.NLL = -tf.reduce_mean(tf.log(
            tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))
        )
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias

    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'wb'))
    def predictionItems(self, sess, u):
        outputs = sess.run(self.all_logits, feed_dict = {self.u: u})  
        return outputs
         
    def pretrain_step(self, sess,  rating, u, i): 

        outputs = sess.run([self.d_updates, self.loss,self.pre_logits], feed_dict = {self.label: rating, self.u: u, self.i: i})
        return outputs

    def pretrain_step_mf(self, sess,  rating, u): 

        outputs = sess.run([self.d_updates, self.loss], feed_dict = {self.label: rating, self.u: u, self.i: i})
        return outputs
    def prediction(self, sess, u, i , user_sequence=None, item_sequence=None):
        # if user_sequence is None or item_sequence is None:
        outputs = sess.run([self.pre_logits], feed_dict = { self.u: u, self.i: i})  
        return outputs



        