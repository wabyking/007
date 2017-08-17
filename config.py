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

 
		flags.DEFINE_string("dataset", "moviesLen_100k", "Comma-separated list of hostname:port pairs")
		flags.DEFINE_string("model_type", "joint", "Comma-separated list of hostname:port pairs")
		flags.DEFINE_string("pretrained_model", "mf0.12.pkl", "Comma-separated list of hostname:port pairs")
		# flags.DEFINE_string("model_type", "mf", "Comma-separated list of hostname:port pairs")
		# flags.DEFINE_string("model_type", "joint", "Comma-separated list of hostname:port pairs")

		# flags.DEFINE_string("test_file_name", "ua.test", "Comma-separated list of hostname:port pairs")
		flags.DEFINE_string("train_file_name", "ratings.csv", "Comma-separated list of hostname:port pairs")
		flags.DEFINE_string("work_dir", "online_model", "Comma-separated list of hostname:port pairs")
		flags.DEFINE_integer("export_version", "80", "Comma-separated list of hostname:port pairs")
		flags.DEFINE_string("moviesLen_100k_split_data", "1998-03-08", "Comma-separated list of hostname:port pairs")
		flags.DEFINE_string("netflix_6_mouth_split_data", "2005-12-01", "Comma-separated list of hostname:port pairs")

		flags.DEFINE_integer("batch_size", 128, "Batch size of data while training")
		flags.DEFINE_integer("gan_k", 128, " ")
		flags.DEFINE_integer("user_delta", 7, "Batch size of data while training")
		flags.DEFINE_integer("item_delta", 7, "Batch size of data while training")  # TODO :  user_delta could not equals to item_delta
		flags.DEFINE_integer("re_rank_list_length", 25, "Batch size of data while training")
		flags.DEFINE_integer("item_windows_size", 4, "Batch size of data while training")
		flags.DEFINE_integer("user_windows_size", 4, "Batch size of data while training")
		flags.DEFINE_integer("n_epochs", 10, "Batch size of data while training")
		flags.DEFINE_integer("test_granularity_count", 2, "Batch size of data while training")
		flags.DEFINE_integer("mf_embedding_dim", 10, "Batch size of data while training")
		flags.DEFINE_integer("rnn_embedding_dim", 10, "Batch size of data while training")        
		flags.DEFINE_float("learning_rate", 0.01, "Batch size of data while training")
		flags.DEFINE_float("grad_clip", 0.1, "Batch size of data while training")
		flags.DEFINE_float("lamda", 0.1, "Batch size of data while training")
		flags.DEFINE_float("temperature", 1, "Batch size of data while training")

		flags.DEFINE_float("momentum", 1, "Batch size of data while training")

		flags.DEFINE_integer("threshold", 300, "Erase the users if the number of rating less than threshold")
		flags.DEFINE_boolean("TestAccuracy", True, "Test accuracy")
		flags.DEFINE_boolean("pretrained", False, "Test accuracy")
		flags.DEFINE_boolean("is_sparse", True, "Test accuracy")
		flags.DEFINE_boolean("rating_flag", False, "Test accuracy")
		flags.DEFINE_boolean("dns", False, "Test accuracy")
		FLAGS= flags.FLAGS
		train_split_data={"moviesLen_100k": FLAGS.moviesLen_100k_split_data , "netflix_6_mouth": FLAGS.netflix_6_mouth_split_data }
		FLAGS.split_data=train_split_data.get(flags.FLAGS.dataset,0)
		# flags.DEFINE_string("split_data", "ratings.csv", "Comma-separated list of hostname:port pairs")

		if FLAGS.dataset =="moviesLen_100k":
			FLAGS.threshold=0
		# # FLAGS.workernum=4
		return FLAGS