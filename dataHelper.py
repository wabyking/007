import pandas as pd 
import os
import datetime
import numpy as np 
import pickle
import config
from tools import log_time_delta

class DataHelper():
    def __init__(self,conf):
        self.conf=conf
        
        self.dataset_pkl = "tmp/"+self.conf.dataset+".pkl"        
        if os.path.exists(self.dataset_pkl):
            self.train = pickle.load(open(self.dataset_pkl, 'rb'))
        else:
            self.train = self.loadData()            
        
        self.u_cnt= self.train["uid"].max()+1
        self.i_cnt= self.train["itemid"].max()+1   # index starts with one instead of zero
        
        print( "The number of users: %d" % self.u_cnt)
        print( "The number of items: %d" % self.i_cnt)
        
        self.train_pkl = "tmp/samples_"+self.conf.dataset+"_train.pkl"
        self.test_pkl = "tmp/samples_"+self.conf.dataset+"_test.pkl"
        self.dict_pkl = "tmp/user_item_"+self.conf.dataset+".pkl"
        self.df= self.train.copy()
                
        if os.path.exists(self.dict_pkl):
            self.user_dict,self.item_dict= pickle.load(open(self.dict_pkl, 'rb'))
        else:            
            self.user_dict,self.item_dict={},{}
            user_windows = self.df.groupby("uid").apply(self.user_windows_apply,user_dict=self.user_dict)
            item_windows = self.df.groupby("itemid").apply(self.item_windows_apply,item_dict=self.item_dict)
            pickle.dump([self.user_dict,self.item_dict], open(self.dict_pkl, 'wb'),protocol=2)
        
        print( "The number of user dict: %d" % len(self.user_dict))
        print( "The number of item dict: %d" % len(self.item_dict))
        
        if os.path.exists(self.train_pkl):
            self.trainset = pickle.load(open(self.train_pkl, 'rb'))
            self.testset = pickle.load(open(self.test_pkl, 'rb'))            
        else:
            self.testset = self.generateSamples(mode="test")
            self.trainset = self.generateSamples(mode="train")
            

        n_examples_train = len(self.trainset)
        n_examples_test = len(self.testset)
        n_train_batch= int(len(self.trainset)/ self.conf.batch_size)
        n_test_batch= int(len(self.testset)/ self.conf.batch_size)
        print( "The number of epoch: %d" %self.conf.n_epochs)
        print( "Train data size: %d" % n_examples_train)
        print( "Test data size: %d" % n_examples_test)
        print( "Batch size: %d" % self.conf.batch_size)
        print( "Iterations per epoch in train data: %d" % n_train_batch)        
        print( "Iterations in test data: %d" % n_test_batch)
        
        np.random.seed(1)
        self.trainset = np.random.permutation(self.trainset)            
            
    def create_dirs(dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    @log_time_delta
    def loadData(self):
        data_dir="data/"+self.conf.dataset
        filename = os.path.join(data_dir, self.conf.train_file_name)
       
        df=pd.read_csv(filename,sep="\t", names=["uid","itemid","rating","timestamp"])

        if self.conf.dataset == "moviesLen-100k":
            stamp2date = lambda stamp :datetime.datetime.fromtimestamp(stamp)
            df["date"]= df["timestamp"].apply(stamp2date)
        else:
            df["date"]= df["timestamp"]

        # df["days"] = (pd.to_datetime(df["date"]) -pd.datetime(1970,1,1)).dt.days
        # df["days"]=df["days"] -df["days"].min()
        y,m,d =    (int(i) for i in self.conf.split_data.split("-"))
        df["days"] = (pd.to_datetime(df["date"]) -pd.datetime(y,m,d )).dt.days

        # df = df[ df.date.str >"1997-09" & df.date < "1998-04"]

        df["user_granularity"] = df["days"] // self.conf.user_delta   # //means floor div
        df["item_granularity"] = df["days"] // self.conf.item_delta   # //means floor div
        
        
        
        if self.conf.threshold > 0:        
            counts_df = pd.DataFrame(df.groupby('uid').size().rename('counts'))
            users = set(counts_df[counts_df.counts>self.conf.threshold].index)
            df = df[df.uid.isin(users)]
        
        
        df['u_original'] = df['uid'].astype('category')
        df['i_original'] = df['itemid'].astype('category')
        df['uid'] = df['u_original'].cat.codes
        df['itemid'] = df['i_original'].cat.codes
        df = df.drop('u_original', 1)
        df = df.drop('i_original', 1)

        pickle.dump(df, open(self.dataset_pkl, 'wb'),protocol=2)
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
        
    def generateSamples(self, mode="train"):
        
        df = self.df
        samples=[]
        # print (len(df[df["user_granularity"]>=self.conf.user_windows_size ]))
#        print( "train size %d "% len(df[(df.user_granularity > df.user_granularity.min() + self.conf.user_windows_size) & (df.user_granularity <0 ) ] ))
#        print( "train size %d "% len(df[df.user_granularity >= 0 ] ))
        # exit()
        if mode=="train":
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
                rating =row["rating"] # get the r_ijt
                item_seqs,user_seqs=[],[]
                for pre_t in range(t-self.conf.user_windows_size ,t):    
                    user_seqs.append(self.user_dict[userid].get(pre_t,None))
                    item_seqs.append(self.item_dict[itemid].get(pre_t,None))
                
                if mode=="train":                
                    null_user_seqs = len([e for e in user_seqs if e is None])
                    if null_user_seqs < self.conf.user_windows_size: # Append new examples when the user have rated at least 1 in recent 140 days.
                        samples.append((user_seqs,item_seqs,rating))    
                else:
                    samples.append((user_seqs,item_seqs,rating))    
                    
        if mode == "train":
            pickle.dump(samples, open(self.train_pkl, 'wb'),protocol=2)
        else:
            pickle.dump(samples, open(self.test_pkl, 'wb'),protocol=2)
        return samples
          
    def getBatch(self,mode="train"):        
        if mode == "train":
            samples = self.trainset
        else:
            samples = self.testset
        n_batches= int(len(samples)/ self.conf.batch_size)
        for i in range(0,n_batches):
            batch = samples[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
            u_seqs= np.array([[self.getUserVector(u_seq) for u_seq in pairs[0] ] for pairs in batch])
            i_seqs= np.array([[self.getItemVector(i_seq) for i_seq in pairs[1] ]  for pairs in batch])
            ratings=np.array([pair[2] for pair in batch])
            yield u_seqs,i_seqs,ratings
#        batch= samples[-1*self.conf.batch_size:]
#        u_seqs= [[self.getUserVector(u_seq) for u_seq in pairs[0] ] for pairs in batch]
#        i_seqs= [[self.getItemVector(i_seq) for i_seq in pairs[1] ]  for pairs in batch]
#        ratings=[pair[2] for pair in batch]
#        yield u_seqs,i_seqs,ratings

def main():
    a=[1,2,3]
    a = np.random.permutation(a)
    
    start_t = time.time() 
    trainset = pickle.load(open("tmp/samples_netflix-6-mouth_train.pkl", 'rb'))
    print( "Elapsed time: ", time.time() - start_t)
    testset = pickle.load(open(self.test_pkl, 'rb'))
    
    FLAGS=config.getTestFlag()
    helper=DataHelper(FLAGS)
    # print(train[train.user_granularity==train.user_granularity.max()-2] )  

    print (helper.u_cnt)
    print (helper.i_cnt)
    i=0
    import time
    start_t = time.time() 
    for x,y,z in helper.getBatch():
        # print(np.array(x).shape)
        # print(np.array(y).shape)
        # print(np.array(z).shape)
        print( "Elapsed time: ", time.time() - start_t)
        start_t = time.time()      
        i+=1
    i=0
if __name__ == '__main__':
    main()
    
#df= helper.train.copy()    
#
#df['u_original'] = df['uid'].astype('category')
#df['i_original'] = df['itemid'].astype('category')
#df['uid'] = df['u_original'].cat.codes
#df['itemid'] = df['i_original'].cat.codes
#df = df.drop('u_original', 1)
#df = df.drop('i_original', 1)
#
#counts_df = pd.DataFrame(df.groupby('uid').size().rename('counts'))
#users = set(counts_df[counts_df.counts>100].index)
#df = df[df.uid.isin(users)]

    
    