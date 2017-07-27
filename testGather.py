import tensorflow as tf 
import numpy as np

x_data=np.linspace(1,100,100)
x= tf.Variable(x_data)
y= tf.placeholder(tf.int32, shape=[3,5])
z = tf.gather(x,y)
y_data=np.linspace(1,15,15).reshape((3,5))
print (y_data) 
result=tf.reduce_sum(tf.one_hot([1,3,5], depth=10, on_value=None, off_value=None, axis=None, dtype=None, name=None),0 )
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	print(sess.run(result,feed_dict={y:y_data}))