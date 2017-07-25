import pandas as pd 
import os
import datetime
import numpy as np 
import pickle

class DataHelper():
	def __init__(self,conf):
		self.conf=conf
		self.train = self.loadData("train")
		self.test = self.loadData("test")
		self.u_cnt= self.train["uid"].max()+1
		self.i_cnt= self.train["itemid"].max()+1   # index starts with one instead of zero

	def create_dirs(dirname):
		if not os.path.exists(dirname):
			os.makedirs(dirname)

	def loadData(self, data_type='train'):
		data_dir="data\\"+self.conf.dataset
		if data_type=="train":
			filename = os.path.join(data_dir, self.conf.train_file_name)
		
		elif data_type=="test":
		 	filename = os.path.join(data_dir, self.conf.test_file_name)
		else:
			print("no such data type")
			exit(0)
		df=pd.read_csv(filename,sep="\t", names=["uid","itemid","rating","timestamp"])

		stamp2date = lambda stamp :datetime.datetime.fromtimestamp(stamp)
		df["date"]= df["timestamp"].apply(stamp2date)
		min_day=df["date"].min()
		df["days"] = (pd.to_datetime(df["date"]) -pd.datetime(1970,1,1)).dt.days
		df["days"]=df["days"] -df["days"].min()
		# df = df[ df.date.str >"1997-09" & df.date < "1998-04"]
		return df

	def user_windows_apply(self,group,user_dict):
		uid=(int(group["uid"].mode()))
		# user_dict[uid]= len(group["days"].unique())
		user_dict.setdefault(uid,{})
		for user_granularity in group["user_granularity"]:
			# print (group[group.user_granularity==user_granularity])
			user_dict[uid][user_granularity]= group[group.user_granularity==user_granularity][["itemid","rating"]]
		return len(group["user_granularity"].unique())
	def item_windows_apply(self,group,item_dict):
		itemid=(int(group["itemid"].mode()))
		# user_dict[uid]= len(group["days"].unique())
		item_dict.setdefault(itemid,{})
		for item_granularity in group["item_granularity"]:
			# print (group[group.user_granularity==user_granularity])
			item_dict[itemid][item_granularity]= group[group.item_granularity==item_granularity][["uid","rating"]]
			# print (item_dict[itemid][item_granularity])
		return len(group["item_granularity"].unique())

	def getUserVector(self,user_set):
		u_seq=[0]*(self.i_cnt)
		
		if not user_set is None:
			for index,row in user_set.iterrows():
				u_seq[row["itemid"]]=row["rating"]
		return u_seq

	def getItemVector(self,item_set):
		i_seq=[0]*(self.u_cnt)
		if not item_set is None:
			for index,row in item_set.iterrows():
				i_seq[row["uid"]]=row["rating"]
		return i_seq

		
	def getBatch(self,dict_pickle_file="user_item_dict.pkl", samples_pickle_file="samples.pkl",shuffle= False,flag="train"):
		if os.path.exists(samples_pickle_file):
			samples=pickle.load(open(samples_pickle_file, 'rb'))
			print("samples load over")
		else:
			df= self.train.copy()

			df["user_granularity"] = df["days"] // self.conf.user_delta   # //means floor div
			df["item_granularity"] = df["days"] // self.conf.item_delta   # //means floor div

			if os.path.exists(dict_pickle_file):
				user_dict,item_dict= pickle.load(open(dict_pickle_file, 'rb'))
			else:
				user_dict,item_dict={},{}
				user_windows = df.groupby("uid").apply(self.user_windows_apply,user_dict=user_dict)
				item_windows = df.groupby("itemid").apply(self.item_windows_apply,item_dict=item_dict)
				pickle.dump([user_dict,item_dict], open(dict_pickle_file, 'wb'),protocol=2)

			samples=[]
			# print (len(df[df["user_granularity"]>=self.conf.user_windows_size ]))
			if flag=="train":
				start=self.conf.user_windows_size
				end=df["user_granularity"].max()+1 - self.conf.test_granularity_count
			else:
				start=df["user_granularity"].max()+1 - self.conf.test_granularity_count
				end=df["user_granularity"].max()+1
			for t in range(start,end): # because  item_windows_size== user_windows_size and user_delta ==item_delta
				print(t)
				for index,row in df[df.user_granularity==t].iterrows():
					userid =row["uid"]
					itemid =row["itemid"]
					rating =row["rating"]
					item_seqs,user_seqs=[],[]
					for pre_t in range(t-self.conf.user_windows_size ,t):	
						# print(user_dict[userid].get(pre_t,None))	
						user_seqs.append(user_dict[userid].get(pre_t,None))
						item_seqs.append(item_dict[itemid].get(pre_t,None))
					samples.append((user_seqs,item_seqs,rating))
	
			# print( np.array(batch_item_seqs).shape)
			# print( np.array(batch_user_seqs).shape)
			pickle.dump(samples, open("samples.pkl", 'wb'),protocol=2)
		
		if flag=="train" and shuffle:
			samples = np.random.permutation(samples)
		print(len(samples))
		n_batches= int(len(samples)/ self.conf.batch_size)
		for i in range(0,n_batches):
			batch = samples[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
			u_seqs= [[self.getUserVector(u_seq) for u_seq in pairs[0] ] for pairs in batch]
			i_seqs= [[self.getItemVector(i_seq) for i_seq in pairs[1] ]  for pairs in batch]
			ratings=[pair[2] for pair in batch]
			yield u_seqs,i_seqs,ratings
		batch= samples[-1*self.conf.batch_size:] 
		u_seqs= [[self.getUserVector(u_seq) for u_seq in pairs[0] ] for pairs in batch]
		i_seqs= [[self.getItemVector(i_seq) for i_seq in pairs[1] ]  for pairs in batch]
		ratings=[pair[2] for pair in batch]
		yield u_seqs,i_seqs,ratings

				



def getTestFlag():
	import tensorflow as tf
	flags = tf.app.flags
	flags.DEFINE_string("dataset", "moviesLen-100k", "Comma-separated list of hostname:port pairs")
	flags.DEFINE_string("test_file_name", "ua.test", "Comma-separated list of hostname:port pairs")
	flags.DEFINE_string("train_file_name", "ua.base", "Comma-separated list of hostname:port pairs")

	flags.DEFINE_integer("batch_size", 32, "Batch size of data while training")
	flags.DEFINE_integer("user_delta", 7, "Batch size of data while training")
	flags.DEFINE_integer("item_delta", 7, "Batch size of data while training")  # TODO :  user_delta could not equals to item_delta
	flags.DEFINE_integer("item_windows_size", 10, "Batch size of data while training")
	flags.DEFINE_integer("user_windows_size", 10, "Batch size of data while training")
	
	flags.DEFINE_boolean("TestAccuracy", True, "Test accuracy")

	# FLAGS = flags.FLAGS
	# # FLAGS.workernum=4
	return flags.FLAGS

def main():

	FLAGS=getTestFlag()
	helper=DataHelper(FLAGS)
	print (helper.u_cnt)
	print (helper.i_cnt)
	i=0
	for x,y,z in helper.getBatch():
		# print(np.array(x).shape)
		# print(np.array(y).shape)
		# print(np.array(z).shape)
		print(i)
		i+=1


if __name__=="__main__":
	main()
