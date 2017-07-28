
def getTestFlag():
	import tensorflow as tf
	flags = tf.app.flags
	flags.DEFINE_string("dataset", "moviesLen-100k", "Comma-separated list of hostname:port pairs")
	# flags.DEFINE_string("test_file_name", "ua.test", "Comma-separated list of hostname:port pairs")
	flags.DEFINE_string("train_file_name", "ratings.csv", "Comma-separated list of hostname:port pairs")

	flags.DEFINE_integer("batch_size", 32, "Batch size of data while training")
	flags.DEFINE_integer("user_delta", 7, "Batch size of data while training")
	flags.DEFINE_integer("item_delta", 7, "Batch size of data while training")  # TODO :  user_delta could not equals to item_delta
	flags.DEFINE_integer("item_windows_size", 10, "Batch size of data while training")
	flags.DEFINE_integer("user_windows_size", 10, "Batch size of data while training")
	flags.DEFINE_integer("n_epochs", 10, "Batch size of data while training")
	flags.DEFINE_integer("test_granularity_count", 2, "Batch size of data while training")
	flags.DEFINE_integer("mf_embedding_dim", 100, "Batch size of data while training")
	flags.DEFINE_float("learning_rate", 0.1, "Batch size of data while training")
	flags.DEFINE_float("grad_clip", 0.1, "Batch size of data while training")
	flags.DEFINE_float("momentum", 1, "Batch size of data while training")
	
	flags.DEFINE_boolean("TestAccuracy", True, "Test accuracy")

	train_split_data={"moviesLen-100k": "1998-04-05" , "netflix-6-mouth": "2005-12-01" }
	flags.FLAGS.split_data=train_split_data[flags.FLAGS.dataset]
	# flags.DEFINE_string("split_data", "ratings.csv", "Comma-separated list of hostname:port pairs")
	# FLAGS = flags.FLAGS
	# # FLAGS.workernum=4
	return flags.FLAGS