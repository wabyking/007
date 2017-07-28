import pandas as pd 
import os,math
import datetime
import numpy as np 
import pickle
import config
from tools import log_time_delta
class DataHelper():
	def __init__(self,conf):
		self.conf=conf
		self.train = self.loadData("train")

		# self.test = self.loadData("test")
		self.u_cnt= self.train["uid"].max()+1
		self.i_cnt= self.train["itemid"].max()+1   # index starts with one instead of zero

	def create_dirs(dirname):
		if not os.path.exists(dirname):
			os.makedirs(dirname)

	@log_time_delta
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

		if self.conf.dataset == "moviesLen-100k":
			stamp2date = lambda stamp :datetime.datetime.fromtimestamp(stamp)
			df["date"]= df["timestamp"].apply(stamp2date)
		else:
			df["date"]= df["timestamp"]

		# df["days"] = (pd.to_datetime(df["date"]) -pd.datetime(1970,1,1)).dt.days
		# df["days"]=df["days"] -df["days"].min()
		y,m,d =	(int(i) for i in self.conf.split_data.split("-"))
		df["days"] = (pd.to_datetime(df["date"]) -pd.datetime(y,m,d )).dt.days

		# df = df[ df.date.str >"1997-09" & df.date < "1998-04"]

		df["user_granularity"] = df["days"] // self.conf.user_delta   # //means floor div
		df["item_granularity"] = df["days"] // self.conf.item_delta   # //means floor div
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

		
	def getBatch(self,shuffle= True,flag="train"):
		samples_pickle_file="tmp/samples_"+self.conf.dataset+"_"+flag+".pkl"
		dict_pickle_file="tmp/user_item_"+self.conf.dataset+".pkl"
		if os.path.exists(samples_pickle_file):
			samples=pickle.load(open(samples_pickle_file, 'rb'))
			print("samples load over")
		else:
			df= self.train.copy()		

			if os.path.exists(dict_pickle_file):
				user_dict,item_dict= pickle.load(open(dict_pickle_file, 'rb'))
			else:
				user_dict,item_dict={},{}
				user_windows = df.groupby("uid").apply(self.user_windows_apply,user_dict=user_dict)
				item_windows = df.groupby("itemid").apply(self.item_windows_apply,item_dict=item_dict)
				pickle.dump([user_dict,item_dict], open(dict_pickle_file, 'wb'),protocol=2)

			samples=[]
			# print (len(df[df["user_granularity"]>=self.conf.user_windows_size ]))
			print( "train size %d "% len(df[(df.user_granularity > df.user_granularity.min() + self.conf.user_windows_size) & (df.user_granularity <0 ) ] ))
			print( "train size %d "% len(df[df.user_granularity >= 0 ] ))
			# exit()
			if flag=="train":
				start=df["user_granularity"].min()+self.conf.user_windows_size
				end=0
			else:
				start=0
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
			pickle.dump(samples, open(samples_pickle_file, 'wb'),protocol=2)

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
	def getBatch4MF(self,flag="train",shuffle=True):
		np.random.seed(0)
		train_flag= np.random.random(len(self.train))>0.2
		if flag=="train":
			df=self.train[train_flag]
			if shuffle ==True:
				df=df.iloc[np.random.permutation(len(df))]
		else:
			df=self.train[~train_flag]

		n_batches= int(len(df)/ self.conf.batch_size)
		for i in range(0,n_batches):
			batch = df[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
			yield batch["uid"],batch["itemid"],batch["rating"]
		batch= df[-1*self.conf.batch_size:] 
		yield batch["uid"],batch["itemid"],batch["rating"]


	def getBatch4MF_fineTune(self,flag="train",shuffle=True):
		np.random.seed(0)
		train_flag= np.random.random(len(self.train))>0.2
		if flag=="train":
			df=self.train[train_flag]
			if shuffle ==True:
				df=df.iloc[np.random.permutation(len(df))]
		else:
			df=self.train[~train_flag]
		samples=pd.DataFrame()
		for i in range(1,6):
			samples=pd.concat([samples, df[df.rating==i].reset_index()[:4800]])
		df=samples
		n_batches= int(len(df)/ self.conf.batch_size)
		for i in range(0,n_batches):
			batch = df[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
			yield batch["uid"],batch["itemid"],batch["rating"]
		batch= df[-1*self.conf.batch_size:] 

	def testModel(self,sess,discriminator,flag="train"):
		results=np.array([])
		for uid,itemid,rating in self.getBatch4MF(flag=flag):
		    feed_dict={discriminator.u: uid, discriminator.i: itemid,discriminator.label: rating}
		    predicted = sess.run(discriminator.pre_logits,feed_dict=feed_dict)
		    error=(np.array(predicted)-np.array(rating))
		    se= np.square(error)
		    results=np.append(results,se)
		# print (sess.run(discriminator.user_bias)[:10])
		mse=np.mean(results)
		return math.sqrt(mse)
		

def main():

	FLAGS=config.getTestFlag()
	helper=DataHelper(FLAGS)
	# print(train[train.user_granularity==train.user_granularity.max()-2] )  
	print ((helper.train.groupby("rating").count()) / len(helper.train) )

	print (helper.u_cnt)
	print (helper.i_cnt)
	i=0
	for x,y,z in helper.getBatch4MF_fineTune():
		# print(np.array(x).shape)
		# print(np.array(y).shape)
		# print(np.array(z).shape)
		print(i)
		i+=1

if __name__ == '__main__':
	main()