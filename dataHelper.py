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

        # print(self.data)
        self.train= self.data[self.data.days<0]
        self.test= self.data[self.data.days>=0]


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

            u_seqss,i_seqss,ratingss,useridss,itemidss=pickle.load(open(pickle_name, 'rb'))
            return u_seqss,i_seqss,ratingss,useridss,itemidss

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

                            samples.append((user_seqs,item_seqs,rating,userid,itemid))    
                    else:
                        samples.append((user_seqs,item_seqs,rating,userid,itemid))          
            
        u_seqss, i_seqss, ratingss,useridss,itemidss=[],[],[],[],[]         

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

            userids=[pair[3] for pair in batch]
            itemids=[pair[4] for pair in batch]
            u_seqss.extend(u_seqs)
            i_seqss.extend(i_seqs)
            ratingss.extend(ratings)
            useridss.extend(userids)
            itemidss.extend(itemids)
            # if i %10==0:
            #     print("processed %d lines"%i)
            # print("spent %f"% (time.time()-start))
        pickle.dump([u_seqss,i_seqss,ratingss,useridss,itemidss], open(pickle_name, 'wb'),protocol=2)
        return u_seqss,i_seqss,ratingss,useridss,itemidss

        
        # if mode=="train" and shuffle:
        #     u_seqss,i_seqss,ratings = sklearn.utils.shuffle(zip(u_seqss,i_seqss,ratings))

        # n_batches= int(len(df)/ self.conf.batch_size)
        # for i in range(0,n_batches):
        #     batch = df[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
        #     yield batch["uid"],batch["itemid"],batch["rating"]
        # batch= df[-1*self.conf.batch_size:] 
        # yield batch["uid"],batch["itemid"],batch["rating"]


        # pickle.dump([u_seqss,i_seqss,ratings], open(pickle_name, 'wb'),protocol=2) 

#    def prepare(self,shuffle=True,mode="train"):
#        i=0
#        pool=Pool(cpu_count())
#        u_seqss,i_seqss,ratingss,useridss,itemidss=self.getBatch_prepare(pool,mode=mode, epoches_size=1)
#        batches=[(x,y,z,w,v) for x,y,z,w,v in zip(u_seqss,i_seqss,ratingss,useridss,itemidss)]
#        if mode=="train" and shuffle:      
#            batches =sklearn.utils.shuffle(batches)
#
#        n_batches= int(len(batches)/ self.conf.batch_size)
#        for i in range(0,n_batches):
#            batch = batches[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
#            u_seqs=pool.map(sparse2dense, [ii[0] for ii in batch])
#            i_seqs=pool.map(sparse2dense, [ii[1] for ii in batch])
#            ratings=[int(ii[2]>3.99)  for ii in batch]
#            userids=[ii[3] for ii in batch]
#            itemids=[ii[4] for ii in batch]
#            yield u_seqs,i_seqs,ratings,userids,itemids


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
            feed_dict={discriminator.u: uid, discriminator.i: itemid}
            predicted = sess.run(discriminator.pre_logits,feed_dict=feed_dict)
            error=(np.array(predicted)-np.array(rating))
            se= np.square(error)
            results=np.append(results,se)
        # print (sess.run(discriminator.user_bias)[:10])
        mse=np.mean(results)

        return math.sqrt(mse)
    
    def prepare(self,shuffle=True,mode="train"):
#        pool=Pool(cpu_count())
#        u_seqs,i_seqs,ratings,userids,itemids=self.getBatch_prepare(pool,mode=mode, epoches_size=1)
#        
#        pos_index = [i for i,r in enumerate(ratingss) if r>3.99]
#        neg_index = [i for i,r in enumerate(ratingss) if r<=3.99]        
#        pos_batches = [(u_seqss[i],i_seqss[i],1) for i in pos_index]
#        neg_batches = [(u_seqss[i],i_seqss[i],0) for i in neg_index]            
#        batches = pos_batches +  neg_batches
#                
        pool=Pool(cpu_count())
        u_seqss,i_seqss,ratingss,useridss,itemidss=self.getBatch_prepare(pool,mode=mode, epoches_size=1)
        batches=[(x,y,z,w,v) for x,y,z,w,v in zip(u_seqss,i_seqss,ratingss,useridss,itemidss)]
        if mode=="train" and shuffle:      
            batches =sklearn.utils.shuffle(batches)

        n_batches= int(len(batches)/ self.conf.batch_size)
        for i in range(0,n_batches):
            batch = batches[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
            u_seqs=pool.map(sparse2dense, [ii[0] for ii in batch])
            i_seqs=pool.map(sparse2dense, [ii[1] for ii in batch])
            ratings=[int(ii[2]>3.99) for ii in batch]
            userids=[ii[3] for ii in batch]
            itemids=[ii[4] for ii in batch]
            yield u_seqs,i_seqs,ratings,userids,itemids


    def getBatchUser(self,users):
        n_batches= int(len(df)/ self.conf.batch_size)
        for i in range(0,n_batches):
            yield df[i*self.conf.batch_size:(i+1) * self.conf.batch_size]             
        yield batch[-1*(n_batches% self.conf.batch_size):]

    def evaluate(self,sess,model):
        users=set(self.data["uid"].unique())
        items=set(self.data["itemid"].unique())
        get_pos_items=lambda group: set(group[group.rating>3.99]["itemid"].tolist())
        pos_items=self.train.groupby("uid").apply(get_pos_items)
        print(pos_items.to_dict())

        # print(user_pos[942])
        users= self.data["uid"].unique().tolist()


        # for user_batch in self.getBatchUser():
        for user_id in users:
            # all_rating= sess.run(mfmodel.all_rating,feed_dict={mfmodel.u: user_id})  #[user_id]
            all_rating= np.random.random( len(items)+1)  #[user_id]

            candiate_index = items - pos_items.get(user_id, set())
            scores =[ (index,all_rating[index]) for index in candiate_index ]
            sortedScores = sorted(scores ,key= lambda x:x[1], reverse = True )

            rarank_index= ([ii[0] for ii in sortedScores[:10]])
            feed_u_seq,feed_i_seq=self.getFeedingData(user_id, rerank_index)
            # feed_dict={model.u: }
            scores = sess.run(rnn_MF_model.all_rating,feed_dict=feed_dict) 

            exit() 


        exit()
        for x,y,z,u,i in helper.prepare(mode=test):
            print(np.array(x).shape)
            print(u)
            print(i)



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



if __name__ == '__main__':
    # for x,y,z in helper.prepare():
    #     print(np.array(x).shape)
    helper.evaluate(None,None)


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

    

#def getUserVector(user_sets):
#    u_seqs=[]
#    for user_set in user_sets:
#        u_seq=[0]*(i_cnt)
#    
#        if not user_set is None:
#            for index,row in user_set.iterrows():
#                u_seq[row["itemid"]]=row["rating"]
#        u_seqs.append(u_seq)
#    return np.array(u_seqs)
#
#
#def getItemVector(item_sets):
#    i_seqs=[]
#    for item_set in item_sets:
#        i_seq=[0]*(u_cnt)
#        if not item_set is None:
#            for index,row in item_set.iterrows():
#                i_seq[row["uid"]]=row["rating"]
#        i_seqs.append(i_seq)
#    return np.array(i_seqs)


# from scipy.sparse import csr_matrix,csr_matrix

# FLAGS=config.getTestFlag()
# start_t = time.time() 
# helper=DataHelper(FLAGS,mode="run")
# print( "Elapsed time: ", time.time() - start_t)
# windows_size=helper.conf.user_windows_size
# u_cnt = helper.u_cnt
# u_cnt = helper.u_cnt
# batch_size=helper.conf.batch_size
# i_cnt = helper.i_cnt


# def getItemVector1(item_sets):
#     rows=[]
#     cols=[]
#     datas=[]
#     for index_i,item_set in enumerate(item_sets):
#         if not item_set is None:
#             for index_j,row in item_set.iterrows():
#                 rows.append(index_i)
#                 cols.append(row["uid"])
#                 datas.append(row["rating"])
#     result=csr_matrix((datas, (rows, cols)), shape=(helper.conf.user_windows_size, helper.u_cnt))
#     return result

# def getUserVector1(user_sets,helper):
#     rows=[]
#     cols=[]
#     datas=[]
#     for index_i,user_set in enumerate(user_sets):           
#         if not user_set is None:
#             for index,row in user_set.iterrows():
#                 rows.append(index_i)
#                 cols.append(row["itemid"])
#                 datas.append(row["rating"])

#     return csr_matrix((datas, (rows, cols)), shape=(helper.conf.user_windows_size, helper.i_cnt))


def haddlePair(batch,pool):     
     u_seqs= pool.map(getUserVector1,([pairs[0] for pairs in batch],helper))
     i_seqs= pool.map(getItemVector1,([pairs[1] for pairs in batch],helper))
     ratings=np.array([pair[2] for pair in batch])
     return u_seqs,i_seqs,ratings
 

#def haddlePair1(batch,pool):
#    index_i=[]
#    index_j=[]
#    index_k=[]
#    data=[]
#    for i,pair in enumerate(batch):
#        for j,user_set in enumerate(pair[0]):
#            if not user_set is None:
#                for _,row in user_set.iterrows():
#                    index_i.append(i*windows_size+j)
#                    index_j.append(j)
#                    index_k.append(row["itemid"])
#                    data.append(row["rating"])
#    start=time.time()
#    u_seqs=csr_matrix((data, (index_i,index_k)), shape=(batch_size*windows_size, i_cnt)).toarray()
#    print ("martrix building %f"%(time.time()-start) )
#    index_i=[]
#    index_j=[]
#    index_k=[]
#    data=[]
#    start=time.time()
#    count=0
#    for i,pair in enumerate(batch):
#        for j,ui_set in enumerate(pair[1]):
#            if not ui_set is None:
#                for _,row in ui_set.iterrows():
#                    index_i.append(i*windows_size+j)
#                    index_j.append(j)
#                    index_k.append(row["uid"])
#                    data.append(row["rating"])
#                    print (count)
#                    count+=1
#    print ("for loop %f"%(time.time()-start) )
#    i_seqs=csr_matrix((data, (index_i,index_k)), shape=(batch_size*windows_size, u_cnt)).toarray()
#    return u_seqs,i_seqs,[pair[2] for pair in batch] 
# def getBatch(pool,mode="train"):        
#     if mode == "train":
#         samples = helper.trainset
#     else:
#         samples = helper.testset    
#     n_batches= int(len(samples)/ helper.conf.batch_size)
#     for i in range(0,5):
#         batch = samples[i*helper.conf.batch_size:(i+1) * helper.conf.batch_size]
#         u_seqs,i_seqs,ratings = haddlePair(batch,pool)
#         yield u_seqs,i_seqs,ratings 
        
# def main1():

    
#     pool=Pool(cpu_count())
#     start_t = time.time() 
#     i = 0
#     for x,y,z in getBatch(pool,mode="train"):                   
#         #print(x[0].toarray().shape)
#         #print(y[0].toarray().shape)
#         #print(np.array(z).shape)
#         print( "Elapsed time: ", time.time() - start_t)
#         outPickle="batches/train/%d"%i
#         pickle.dump((x,y,z), open(outPickle, 'wb'),protocol=2)
#         start_t = time.time()     
#         i+=1        
    
#     start_t = time.time() 
#     i = 0
#     for x,y,z in getBatch(pool,mode="test"):                   
#         #print(x[0].toarray().shape)
#         #print(y[0].toarray().shape)
#         #print(np.array(z).shape)
#         print( "Elapsed time: ", time.time() - start_t)
#         outPickle="batches/test/%d"%i
#         pickle.dump((x,y,z), open(outPickle, 'wb'),protocol=2)
#         start_t = time.time()   
#         i+=1
#     exit()
        
#     for i in range(10):
#         start_t = time.time()
#         x,y,z = pickle.load(open("batches/train/%d"%i, 'rb'))
#         xx= [item.toarray() for item in x]
#         yy= [item.toarray() for item in y]        
#         print( "Elapsed time: ", time.time() - start_t)


# def sqrt(x):
#     print(x)

# self.u_cnt= self.data["uid"].max()+1
# self.i_cnt= self.data["itemid"].max()+1   
         

# self.u_cnt= self.data["uid"].max()+1
# self.i_cnt= self.data["itemid"].max()+1   # index starts with one instead of zero

# print( "The number of users: %d" % self.u_cnt)
# print( "The number of items: %d" % self.i_cnt)
 # self.df= self.data.copy()
                
        # if os.path.exists(self.dict_pkl):
        #     self.user_dict,self.item_dict= pickle.load(open(self.dict_pkl, 'rb'))
        # else:            
        #     self.user_dict,self.item_dict={},{}
        #     user_windows = self.df.groupby("uid").apply(self.user_windows_apply,user_dict=self.user_dict)
        #     item_windows = self.df.groupby("itemid").apply(self.item_windows_apply,item_dict=self.item_dict)
        #     pickle.dump([self.user_dict,self.item_dict], open(self.dict_pkl, 'wb'),protocol=2)
        
        # print( "The number of user dict: %d" % len(self.user_dict))
        # print( "The number of item dict: %d" % len(self.item_dict))
        
        # if os.path.exists(self.train_pkl):
        #     self.trainset = pickle.load(open(self.train_pkl, 'rb'))
        #     self.testset = pickle.load(open(self.test_pkl, 'rb'))            
        # else:
        #     self.testset = self.generateSamples(mode="test")
        #     self.trainset = self.generateSamples(mode="train")
        # # else:
        # #     self.train = pickle.load(open(self.dataset_pkl, 'rb'))
        # #     self.trainset = pickle.load(open(self.train_pkl, 'rb'))
        # #     self.testset = pickle.load(open(self.test_pkl, 'rb'))                        
        # self.u_cnt= self.train["uid"].max()+1
        # self.i_cnt= self.train["itemid"].max()+1   # index starts with one instead of zero
            
        # n_examples_train = len(self.trainset)
        # n_examples_test = len(self.testset)
        # n_train_batch= int(len(self.trainset)/ self.conf.batch_size)
        # n_test_batch= int(len(self.testset)/ self.conf.batch_size)
        # print( "The number of epoch: %d" %self.conf.n_epochs)
        # print( "Train data size: %d" % n_examples_train)
        # print( "Test data size: %d" % n_examples_test)
        # print( "Batch size: %d" % self.conf.batch_size)
        # print( "Iterations per epoch in train data: %d" % n_train_batch)        
        # print( "Iterations in test data: %d" % n_test_batch)
        
        # np.random.seed(1)
        # ind = np.random.permutation(len(self.trainset))  
        # self.trainset = [self.trainset[i] for i in ind]    

