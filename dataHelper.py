import pandas as pd 
import os
import datetime
import numpy as np 
import pickle
import config
from tools import log_time_delta
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix,csr_matrix
import math
from config import Singleton
import sklearn

class DataHelper():
    def __init__(self,conf,mode="run"):
        self.conf=conf
        self.data = self.loadData()
        self.u_cnt= self.data ["uid"].max()+1
        self.i_cnt= self.data ["itemid"].max()+1                    
        
        # print( "The number of users: %d" % self.u_cnt)
        # print( "The number of items: %d" % self.i_cnt) 
        print ("..")      
            
    def create_dirs(self,dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    @log_time_delta
    def loadData(self):
        dataset_pkl = "tmp/"+self.conf.dataset +".pkl"
        if os.path.exists(dataset_pkl):
            return pickle.load(open(dataset_pkl, 'rb'))
        data_dir="data/%s"% self.conf.dataset 
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
        
        pickle.dump(df, open(dataset_pkl, 'wb'),protocol=2)
                        
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
          
    def getBatch_prepare(self,pool,mode="train", epoches_size=1,shuffle=True):  
        pickle_name = "tmp/samples_"+self.conf.dataset+"_"+mode+".pkl"
        if os.path.exists(pickle_name):
            print ("load %s over"% pickle_name )
            u_seqss,i_seqss,ratingss=pickle.load(open(pickle_name, 'rb'))
            return u_seqss,i_seqss,ratingss
        else:
            dict_pkl = "tmp/user_item_"+self.conf.dataset+".pkl"

            if os.path.exists(dict_pkl):
                user_dict,item_dict= pickle.load(open(dict_pkl, 'rb'))
            else:            
                user_dict,item_dict={},{}
                user_windows = self.data.groupby("uid").apply(self.user_windows_apply,user_dict=user_dict)
                item_windows = self.data.groupby("itemid").apply(self.item_windows_apply,item_dict=item_dict)
                pickle.dump([user_dict,item_dict], open(dict_pkl, 'wb'),protocol=2)
            print ("dict load over")
            
            df = self.data
            samples=[]
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
                        user_seqs.append(user_dict[userid].get(pre_t,None))
                        item_seqs.append(item_dict[itemid].get(pre_t,None))
                    
                    if mode=="train": #if :                
                        null_user_seqs = len([e for e in user_seqs if e is None])
                        if null_user_seqs < self.conf.user_windows_size: # Append new examples when the user have rated at least 1 in recent 140 days.
                            samples.append((user_seqs,item_seqs,rating))    
                    else:
                        samples.append((user_seqs,item_seqs,rating))          
            
        u_seqss, i_seqss, ratingss=[],[],[]         
        start=time.time()
        
        print("shuffle time spent %f"% (time.time()-start))
        n_batches= int(len(samples)/ self.conf.batch_size)
        print("%d batch"% n_batches)

                
        for i in range(0,n_batches):
            start=time.time()

            batch = samples[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
            #u_seqs= np.array([[self.getUserVector(u_seq) for u_seq in pairs[0] ] for pairs in batch])
            #i_seqs= np.array([[self.getItemVector(i_seq) for i_seq in pairs[1] ]  for pairs in batch])
            u_seqs= pool.map(getUserVector1,[pairs[0] for pairs in batch])
            i_seqs= pool.map(getItemVector1,[pairs[1] for pairs in batch])

            ratings=[pair[2] for pair in batch]
            u_seqss.extend(u_seqs)
            i_seqss.extend(i_seqs)
            ratingss.extend(ratings)
            
            # if i %10==0:
            #     print("processed %d lines"%i)
            # print("spent %f"% (time.time()-start))
        pickle.dump([u_seqss,i_seqss,ratingss], open(pickle_name, 'wb'),protocol=2)
        return u_seqss,i_seqss,ratingss
        
        # if mode=="train" and shuffle:
        #     u_seqss,i_seqss,ratings = sklearn.utils.shuffle(zip(u_seqss,i_seqss,ratings))

        # n_batches= int(len(df)/ self.conf.batch_size)
        # for i in range(0,n_batches):
        #     batch = df[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
        #     yield batch["uid"],batch["itemid"],batch["rating"]
        # batch= df[-1*self.conf.batch_size:] 
        # yield batch["uid"],batch["itemid"],batch["rating"]


        # pickle.dump([u_seqss,i_seqss,ratings], open(pickle_name, 'wb'),protocol=2) 

    def getUserVector(self,user_sets):
       u_seqs=[]
       for user_set in user_sets:
           u_seq=[0]*(i_cnt)
       
           if not user_set is None:
               for index,row in user_set.iterrows():
                   u_seq[row["itemid"]]=row["rating"]
           u_seqs.append(u_seq)
       return np.array(u_seqs)
    
    
    def getItemVector(self,item_sets):
       i_seqs=[]
       for item_set in item_sets:
           i_seq=[0]*(u_cnt)
           if not item_set is None:
               for index,row in item_set.iterrows():
                   i_seq[row["uid"]]=row["rating"]
           i_seqs.append(i_seq)
       return np.array(i_seqs)
    def getBatch4MF(self,flag="train",shuffle=True):
        np.random.seed(0)
        train_flag= np.random.random(len(self.data))>0.2
        if flag=="train":
            df=self.data[train_flag]
            if shuffle ==True:
                df=df.iloc[np.random.permutation(len(df))]
                print ("shuffle over")
        else:
            df=self.data[~train_flag]

        n_batches= int(len(df)/ self.conf.batch_size)
        for i in range(0,n_batches):
            batch = df[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
            yield batch["uid"],batch["itemid"],batch["rating"]
        batch= df[-1*self.conf.batch_size:] 
        yield batch["uid"],batch["itemid"],batch["rating"]

    def testModel(self,sess,discriminator,flag="test"):
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
    
    def prepare(self,shuffle=True,mode="train"):
        pool=Pool(cpu_count())
        u_seqss,i_seqss,ratingss=self.getBatch_prepare(pool,mode=mode, epoches_size=1)
        
        pos_index = [i for i,r in enumerate(ratingss) if r>3.99]
        neg_index = [i for i,r in enumerate(ratingss) if r<=3.99]        
        pos_batches = [(u_seqss[i],i_seqss[i],1) for i in pos_index]
        neg_batches = [(u_seqss[i],i_seqss[i],0) for i in neg_index]            
        batches = pos_batches +  neg_batches
            
        if mode=="train" and shuffle:             
            batches = sklearn.utils.shuffle(batches)   

        #batches=[(x,y,z) for x,y,z in zip(u_seqss,i_seqss,ratingss)]
       
        n_batches= int(len(batches)/ self.conf.batch_size)
        for i in range(0,n_batches):
            batch = batches[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
            u_seqs=pool.map(sparse2dense, [ii[0] for ii in batch])
            i_seqs=pool.map(sparse2dense, [ii[1] for ii in batch])
            ratings=[ii[2] for ii in batch]
            yield u_seqs,i_seqs,ratings


def sparse2dense(sparse):
    return sparse.toarray()



flagFactory=Singleton()
FLAGS=flagFactory.getInstance()
helper=DataHelper(FLAGS)

def getItemVector1(item_sets):
    rows=[]
    cols=[]
    datas=[]
    for index_i,item_set in enumerate(item_sets):
        if not item_set is None:
            for index_j,row in item_set.iterrows():
                rows.append(index_i)
                cols.append(row["uid"])
                datas.append(row["rating"])
    result=csr_matrix((datas, (rows, cols)), shape=(helper.conf.user_windows_size, helper.u_cnt))
    return result

def getUserVector1(user_sets):
    rows=[]
    cols=[]
    datas=[]
    for index_i,user_set in enumerate(user_sets):           
        if not user_set is None:
            for index,row in user_set.iterrows():
                rows.append(index_i)
                cols.append(row["itemid"])
                datas.append(row["rating"])
    return csr_matrix((datas, (rows, cols)), shape=(helper.conf.user_windows_size, helper.i_cnt))



def main():

    start_t = time.time() 
    start_t = time.time() 
    i = 0

    pool=Pool(cpu_count())
    for x,y,z in helper.getBatch(pool,mode="train", epoches_size=1):                   
        print(x[0].toarray().shape)
        print(y[0].toarray().shape)
        print(np.array(z).shape)
        print( "Elapsed time: ", time.time() - start_t)
        outPickle="batches/train/%d"%i
        pickle.dump((x,y,z), open(outPickle, 'wb'),protocol=2)
        start_t = time.time()     
        i+=1   


        
#if __name__ == '__main__':
#    for x,y,z in helper.prepare():
#        print(np.array(x).shape)
