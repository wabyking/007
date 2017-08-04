class Singleton(object):
	__instance=None
	def __init__(self):
		pass
	def getInstance(self):
		if Singleton.__instance is None:
			# Singleton.__instance=object.__new__(cls,*args,**kwd)
			Singleton.__instance=self.getTestFlag()
		return Singleton.__instance
	def getTestFlag(self):
		import tensorflow as tf
		flags = tf.app.flags

 
		flags.DEFINE_string("dataset", "moviesLen-100k", "Comma-separated list of hostname:port pairs")

		# flags.DEFINE_string("test_file_name", "ua.test", "Comma-separated list of hostname:port pairs")
		flags.DEFINE_string("train_file_name", "ratings.csv", "Comma-separated list of hostname:port pairs")
		flags.DEFINE_string("work_dir", "online_model", "Comma-separated list of hostname:port pairs")
		flags.DEFINE_integer("export_version", "80", "Comma-separated list of hostname:port pairs")

		flags.DEFINE_integer("batch_size", 32, "Batch size of data while training")
		flags.DEFINE_integer("user_delta", 7, "Batch size of data while training")
		flags.DEFINE_integer("item_delta", 7, "Batch size of data while training")  # TODO :  user_delta could not equals to item_delta

		flags.DEFINE_integer("item_windows_size", 20, "Batch size of data while training")
		flags.DEFINE_integer("user_windows_size", 20, "Batch size of data while training")
		flags.DEFINE_integer("n_epochs", 10, "Batch size of data while training")
		flags.DEFINE_integer("test_granularity_count", 2, "Batch size of data while training")
		flags.DEFINE_integer("mf_embedding_dim", 100, "Batch size of data while training")
		flags.DEFINE_integer("rnn_embedding_dim", 100, "Batch size of data while training")        
		flags.DEFINE_float("learning_rate", 0.01, "Batch size of data while training")
		flags.DEFINE_float("grad_clip", 5, "Batch size of data while training")

		flags.DEFINE_float("momentum", 1, "Batch size of data while training")

		flags.DEFINE_integer("threshold", 300, "Erase the users if the number of rating less than threshold")
		flags.DEFINE_boolean("TestAccuracy", True, "Test accuracy")

		train_split_data={"moviesLen-100k": "1998-04-05" , "netflix-6-mouth": "2005-12-01" }
		flags.FLAGS.split_data=train_split_data.get(flags.FLAGS.dataset,0)
		# flags.DEFINE_string("split_data", "ratings.csv", "Comma-separated list of hostname:port pairs")
		FLAGS= flags.FLAGS
		if FLAGS.dataset =="moviesLen-100k":
			FLAGS.threshold=0
		# # FLAGS.workernum=4
		return FLAGS