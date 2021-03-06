import pandas as pd 
import os
import datetime
import numpy as np 
import pickle
import config
from tools import log_time_delta
import time
from multiprocessing import Pool
from multiprocessing import freeze_support
# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
import math
from config import Singleton
import sklearn
import itertools
import tensorflow as tf
import random

mp=False

class DataHelper():
    def __init__(self,conf,mode="run"):
        self.conf=conf
        print(self.conf.split_data)
        self.data = self.loadData()
        df = self.data

        self.train= self.data[self.data.days<0]
        self.test= self.data[self.data.days>=0]
        x=(len(df[df["user_granularity"]<=(df["user_granularity"].min()+self.conf.user_windows_size) ]))
        y=(len(df[(df["user_granularity"]>(df["user_granularity"].min()+self.conf.user_windows_size)) & (df["user_granularity"]<0)]))
        z=(len(df[ (df["user_granularity"]>=0)]))

        # print(x,y,z)

        self.u_cnt= self.data ["uid"].max()+1  #  how?
        

        self.i_cnt= self.data ["itemid"].max()+1

        
        self.user_dict,self.item_dict=self.getdicts()

        self.users=set(self.data["uid"].unique())
        self.test_users=set(self.test["uid"].unique())
        self.items=set(self.data["itemid"].unique())
        self.shared_users=set(self.train["uid"].unique()) & set(self.test["uid"].unique())
        # train_users=self.train["uid"].unique()
        # test_users=self.test["uid"].unique()
        # interaction_users = set(train_users) & set(test_users)
        # print(len(train_users))
        # print(len(test_users))
        # print(len(interaction_users))

        get_pos_items=lambda group: set(group[group.rating>3.99]["itemid"].tolist())
        self.pos_items=self.train.groupby("uid").apply(get_pos_items)
        # print(pos_items.to_dict())
                
        user_item_pos_rating_time_dict= lambda group:{item:time for i,(item,time)  in group[group.rating>3.99][["itemid","user_granularity"]].iterrows()}
        self.user_item_pos_rating_time_dict=self.train.groupby("uid").apply(user_item_pos_rating_time_dict).to_dict()
       
        
        self.test_pos_items=self.test.groupby("uid").apply(get_pos_items).to_dict()


       
            
    def create_dirs(self,dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # @log_time_delta
    def loadData(self):
        dataset_pkl = "tmp/"+self.conf.dataset +"_"+self.conf.split_data+".pkl"
        if os.path.exists(dataset_pkl):
            return pickle.load(open(dataset_pkl, 'rb'))
        data_dir="data/%s"% self.conf.dataset 
        filename = os.path.join(data_dir, self.conf.train_file_name)
       
        df=pd.read_csv(filename,sep="\t", names=["uid","itemid","rating","timestamp"])

        df=df.sort_values(["uid","itemid"])

        print(df ["uid"].max()+1)
        if self.conf.dataset == "moviesLen_100k":
            stamp2date = lambda stamp :datetime.datetime.fromtimestamp(stamp)
            df["date"]= df["timestamp"].apply(stamp2date)
        else:
            df["date"]= df["timestamp"]

        # df["days"] = (pd.to_datetime(df["date"]) -pd.datetime(1970,1,1)).dt.days
        # df["days"]=df["days"] -df["days"].min()
        y,m,d = (int(i) for i in self.conf.split_data.split("-"))

        df["days"] = (pd.to_datetime(df["date"]) -pd.datetime(y,m,d )).dt.days

        
        # df = df[ df.date.str >"1997-09" & df.date < "1998-04"]

        df["item_granularity"] = df["days"] // self.conf.item_delta   # //means floor div
        df["user_granularity"] = df["days"] // self.conf.user_delta   # //means floor div
        
        

        if self.conf.threshold > 0: # remove the users while the rating of them is lower than threshold
            counts_df = pd.DataFrame(df.groupby('uid').size().rename('counts'))
            users = set(counts_df[counts_df.counts>self.conf.threshold].index)
            df = df[df.uid.isin(users)]

#       re-arrange the user and item index from zero     
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
            
    # @log_time_delta
    def getdicts(self):
        dict_pkl = "tmp/user_item_"+self.conf.dataset+".pkl"
        if os.path.exists(dict_pkl):
            # start=time.time()
            import gc
            gc.disable()
            user_dict,item_dict= pickle.load(open(dict_pkl, 'rb'))
            gc.enable()
            # print( "Elapsed time: ", time.time() - start)
        else:            
            user_dict,item_dict={},{}
            user_windows = self.data.groupby("uid").apply(self.user_windows_apply,user_dict=user_dict)
            item_windows = self.data.groupby("itemid").apply(self.item_windows_apply,item_dict=item_dict)
            pickle.dump([user_dict,item_dict], open(dict_pkl, 'wb'),protocol=2)
            # pickle.dump([user_dict,item_dict], open(dict_pkl, 'wb'))
        # print ("dict load over")
        return user_dict,item_dict
    
    def prepare_uniform(self,pool=None,mode="train", epoches_size=1,shuffle=True,fresh=False):  
        # pickle_name = "tmp/samples_"+self.conf.dataset+"_" +mode+".pkl"
             
        df = self.data
        samples=[]

        if mode=="train":
            start=df["user_granularity"].min()+self.conf.user_windows_size
            end=0
        else:
            start=0
            end=df["user_granularity"].max()+1

        for t in range(start,end): # because item_windows_size== user_windows_size and user_delta ==item_delta
            print(t)
            for index,row in df[df.user_granularity==t].iterrows():
                userid =row["uid"]
                itemid =row["itemid"]
                rating =row["rating"] # get the r_ijt
                item_seqs,user_seqs=[],[]
                for pre_t in range(t-self.conf.user_windows_size ,t):  
                    if self.conf.is_sparse:                        
                        user_seqs.append( self.user_dict[userid].get(pre_t,None))
                        item_seqs.append( self.item_dict[itemid].get(pre_t,None))
                    else:
                        user_seqs.append( self.user_dict[userid].get(pre_t,None))
                        item_seqs.append( self.item_dict[itemid].get(pre_t,None))

                if self.conf.is_sparse:
                    user_seqs,item_seqs=getUserVector1(user_seqs),getItemVector1(item_seqs)#user_seqs: T*|M|
                else:
                    user_seqs,item_seqs=self.getUserVector(user_seqs),self.getItemVector(item_seqs)
                
#                if mode=="train": #if :                
#                    null_user_seqs = len([e for e in user_seqs if e is None])
#                    if null_user_seqs < self.conf.user_windows_size: # Append new examples when the user have rated at least 1 in recent 140 days.
#                        samples.append((user_seqs,item_seqs,rating,userid,itemid))    
#                else:
                samples.append((user_seqs,item_seqs,rating,userid,itemid))          
        
        return samples


    def getSeqOverAlltime(self,userid, itemid):   #does labeled data also do this? 

        u_seqs,i_seqs=[],[]
        for t in range(self.data["user_granularity"].min(),0):
    
            u_seqs.append(self.user_dict[userid].get(t,None))
            i_seqs.append(self.item_dict[itemid].get(t,None))

        u_seqss,i_seqss=[],[]
        for t in range( self.data["user_granularity"].min() ,0- self.conf.user_windows_size):
            u_seqss.append( u_seqs[t:t+self.conf.user_windows_size])
            i_seqss.append( i_seqs[t:t+self.conf.user_windows_size])     

        if self.conf.is_sparse:
            return [i for i in map(getUserVector1, u_seqss)],[i for i in map(getItemVector1, i_seqss)]
        else:              
            return [i for i in map(self.getUserVector, u_seqss)],[i for i in map(self.getItemVector, i_seqss)]
        

    def getSeqInTime(self,userid,itemid,t):
        u_seqs,i_seqs=[],[]
        for t in range(t-self.conf.user_windows_size,t):
            u_seqs.append(self.user_dict[userid].get(t,None))
            i_seqs.append(self.item_dict[itemid].get(t,None))
        if self.conf.is_sparse:
            return getUserVector1(u_seqs),getItemVector1(i_seqs)
        else:

            return self.getUserVector(u_seqs),self.getItemVector(i_seqs)

    def prepare_pair(self,pool=None,mode="train", epoches_size=1,sess=None,model=None):          
        df = self.data
        samples=[]           
        for user in df["uid"].unique():
            pos_items_time_dict=self.user_item_pos_rating_time_dict.get(user,{})   # null 
            if len(pos_items_time_dict)==0:
                continue
            # generate pesudo labels for the given user
            if self.conf.dns:
                all_rating = model.predictionItems(sess,user)                           # todo delete the pos ones            
                exp_rating = np.exp(np.array(all_rating) *self.conf.temperature)
                prob = exp_rating / np.sum(exp_rating)
                negative_items_sampled = np.random.choice(np.arange(self.i_cnt), size=len(pos_items_time_dict), p=prob)

            else:
                negative_items_sampled = np.random.choice(np.arange(self.i_cnt), size=len(pos_items_time_dict))
            for item in negative_items_sampled:
                
                # the pesudo labels are regarded as high-quality negative labels but at the beginning, the pesudo labels are very low-quality ones.
                u_seqs_neg,i_seqs_neg = helper.getSeqInTime(user,item,0)   
                
                positive_item = np.random.choice(list(pos_items_time_dict.keys()))
                t=pos_items_time_dict.get(positive_item)
                u_seqs,i_seqs=self.getSeqInTime(user,positive_item,t)
                samples.append((user,u_seqs,positive_item,i_seqs,item,i_seqs_neg))
        return samples



    def prepare_dns(self,pool=None,mode="train", epoches_size=1,sess=None,model=None):          
#        df = self.data
        df = self.test
        positive_samples = []
        negative_samples = []
                #hehehe
        for user in df["uid"].unique():

            pos_items_time_dict=self.user_item_pos_rating_time_dict.get(user,{})   # null 
            if len(pos_items_time_dict)==0:
                continue

            # generate pesudo labels for the given user
            all_rating = model.predictionItems(sess,user)                           # todo delete the pos ones            
            exp_rating = np.exp(np.array(all_rating) *self.conf.temperature)
            prob = exp_rating / np.sum(exp_rating)
            
            negative_items_argmax = np.argsort(prob)[::-1][:2]

            negative_items_sampled = np.random.choice(np.arange(self.i_cnt), size=len(pos_items_time_dict), p=prob)
            
            
            for item in negative_items_sampled:
                
                # the pesudo labels are regarded as high-quality negative labels but at the beginning, the pesudo labels are very low-quality ones.
                u_seqs,i_seqs = helper.getSeqInTime(user,item,0)   
                # if self.conf.pairwise:
                negative_samples.append((u_seqs,i_seqs,0,user,item ))
                
                # sample positive examples in a random manner
                # positive_item,t = pos_items_time_dict.items()[np.random.randint(len(pos_items_time_dict), size=1)[0]]


                positive_item = np.random.choice(list(pos_items_time_dict.keys()))
                t=pos_items_time_dict.get(positive_item)
                u_seqs,i_seqs=self.getSeqInTime(user,positive_item,t)
                positive_samples.append((u_seqs,i_seqs,1,user,positive_item))
                                    
#            for  i,(pos_itm,t) in enumerate(pos_items_time_dict.items()):
#                u_seqs,i_seqs=self.getSeqInTime(user,pos_itm,t)
#                samples.append((u_seqs,i_seqs,1,user,pos_itm))
#                u_seqs,i_seqs = helper.getSeqInTime(user,neg_itms[i],0)   
#                samples.append((u_seqs,i_seqs,0,user,neg_itms[i] ))
#                u_seqss,i_seqss= self.getSeqOverAlltime(user,neg_item_id)
#                predicted = model.prediction(sess,u_seqss,i_seqss, [user]*len(u_seqss),[neg_item_id]*len(u_seqss),sparse=True)
#                index=np.argmax(predicted)
#                samples.append((u_seqss[index],i_seqss[index],0,user,neg_item_id ))
        return positive_samples+negative_samples
    def getBatchFromSamples_pair(self,pool=None,dns=True,sess=None,model=None,fresh=True,mode="train", epoches_size=1,shuffle=True):

        pickle_name = "tmp/samples_"+ ("dns" +str(self.conf.subset_size)+"_" if dns else "uniform") + ("_pair" if self.conf.pairwise else "") +("_sparse_tensor_" if self.conf.sparse_tensor else ( "_sparse" if self.conf.is_sparse else "_") ) +self.conf.dataset+"_"+str(self.conf.user_windows_size)+"_" +mode+".pkl"
        print (pickle_name)
        if os.path.exists(pickle_name) and not fresh:
            import gc
            gc.disable()
            samples=pickle.load(open(pickle_name, 'rb'))
            gc.enable()        
        else:
            samples = self.prepare_pair(mode=mode, epoches_size=epoches_size)            
            pickle.dump(samples, open(pickle_name, 'wb'),protocol=2)

        if mode=="train" and shuffle:      
            start=time.time()            
            
            random.shuffle(samples)                      
            print("shuffle time spent %f"% (time.time()-start))
            # samples =sklearn.utils.shuffle(samples) 
            # shuffle_index= sklearn.utils.shuffle(np.arange(len(samples)))
            # samples= np.array(samples)[shuffle_index].tolist()

        n_batches = int(len(samples)/ self.conf.batch_size)
        print("%d batch"% n_batches)
                
        for i in range(0,n_batches):
            start=time.time()            
           
            batch = samples[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
                
            #u_seqs= np.array([[self.getUserVector(u_seq) for u_seq in pairs[0] ] for pairs in batch])
            #i_seqs= np.array([[self.getItemVector(i_seq) for i_seq in pairs[1] ]  for pairs in batch])
            if not self.conf.pairwise:
                u_seqs=[pair[0] for pair in batch]
                i_seqs=[pair[1] for pair in batch]
                if not self.conf.sparse_tensor and self.conf.is_sparse:
                    if pool is not None:
                        u_seqs=pool.map(sparse2dense, u_seqs)
                        i_seqs=pool.map(sparse2dense, i_seqs)
                    else:
                        u_seqs=[v for v in map(sparse2dense, u_seqs)]
                        i_seqs=[v for v in map(sparse2dense, i_seqs)]

                if self.conf.rating_flag:
                    ratings=[float(ii[2]) for ii in batch]
                else:
                    ratings=[float(ii[2]>3.99) for ii in batch]

                userids=[pair[3] for pair in batch]
                itemids=[pair[4] for pair in batch]

                # if i %10==0:
                #     print("processed %d lines"%i)
                # # print("spent %f"% (time.time()-start))
                if self.conf.sparse_tensor:
                    u_seqs,i_seqs=self.get_sparse_intput(u_seqs,i_seqs)
                yield u_seqs,i_seqs,ratings,userids,itemids
            else:
                #(user,u_seqs,item,i_seqs,item_neg,i_seqs_neg)
                user=[pair[0] for pair in batch]
                u_seqs=[pair[1] for pair in batch]
                item=[pair[2] for pair in batch]
                i_seqs=[pair[3] for pair in batch]
                item_neg=[pair[4] for pair in batch]
                i_seqs_neg=[pair[5] for pair in batch]
                if not self.conf.sparse_tensor and self.conf.is_sparse:
                    if pool is not None:
                        u_seqs=pool.map(sparse2dense, u_seqs)
                        i_seqs=pool.map(sparse2dense, i_seqs)
                        item_neg=pool.map(sparse2dense, item_neg)
                    else:
                        u_seqs=[v for v in map(sparse2dense, u_seqs)]
                        i_seqs=[v for v in map(sparse2dense, i_seqs)]
                        i_seqs_neg=[v for v in map(sparse2dense, i_seqs_neg)]
                if self.conf.sparse_tensor:
                    u_seqs=self.get_user_sparse_input(u_seqs)
                    i_seqs=self.get_item_sparse_input(i_seqs)
                    i_seqs_neg=self.get_item_sparse_input(i_seqs_neg)

                yield (user,u_seqs,item,i_seqs,item_neg,i_seqs_neg)
 


    def getBatchFromSamples_point(self,pool=None,dns=True,sess=None,model=None,fresh=True,mode="train", epoches_size=1,shuffle=True):

        pickle_name = "tmp/samples_"+ ("dns" +str(self.conf.subset_size)+"_" if dns else "uniform") +("_sparse_tensor_" if self.conf.sparse_tensor else ( "_sparse_" if self.conf.is_sparse else "_") ) +self.conf.dataset+"_"+str(self.conf.user_windows_size)+"_" +mode+".pkl"

        if os.path.exists(pickle_name) and not fresh:
            import gc
            gc.disable()
            samples=pickle.load(open(pickle_name, 'rb'))
            gc.enable()        
        else:
            samples = self.prepare_uniform(mode=mode, epoches_size=epoches_size)            
            pickle.dump(samples, open(pickle_name, 'wb'),protocol=2)

        if mode=="train" and shuffle:      
            start=time.time()            
            
            random.shuffle(samples)                      
            print("shuffle time spent %f"% (time.time()-start))
            # samples =sklearn.utils.shuffle(samples) 
            # shuffle_index= sklearn.utils.shuffle(np.arange(len(samples)))
            # samples= np.array(samples)[shuffle_index].tolist()

        n_batches = int(len(samples)/ self.conf.batch_size)
        print("%d batch"% n_batches)
                
        for i in range(0,n_batches):
            start=time.time()            
           
            batch = samples[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
                
            #u_seqs= np.array([[self.getUserVector(u_seq) for u_seq in pairs[0] ] for pairs in batch])
            #i_seqs= np.array([[self.getItemVector(i_seq) for i_seq in pairs[1] ]  for pairs in batch])

            u_seqs=[pair[0] for pair in batch]

            i_seqs=[pair[1] for pair in batch]
            if not self.conf.sparse_tensor and self.conf.is_sparse:
                if pool is not None:
                    u_seqs=pool.map(sparse2dense, u_seqs)
                    i_seqs=pool.map(sparse2dense, i_seqs)
                else:
                    u_seqs=[v for v in map(sparse2dense, u_seqs)]
                    i_seqs=[v for v in map(sparse2dense, i_seqs)]

            if self.conf.rating_flag:
                ratings=[float(ii[2]) for ii in batch]
            else:
                ratings=[float(ii[2]>3.99) for ii in batch]

            userids=[pair[3] for pair in batch]
            itemids=[pair[4] for pair in batch]

            # if i %10==0:
            #     print("processed %d lines"%i)
            # # print("spent %f"% (time.time()-start))
            if self.conf.sparse_tensor:
                u_seqs,i_seqs=self.get_sparse_intput(u_seqs,i_seqs)
            yield u_seqs,i_seqs,ratings,userids,itemids


    def getBatchFromDNS(self,pool=None,dns=True,sess=None,model=None,fresh=True,mode="train", epoches_size=1,shuffle=True):
        pickle_name = "tmp/samples_"+ ("dns" +str(self.conf.subset_size)+"_" if dns else "uniform") +("_sparse_tensor_" if self.conf.sparse_tensor else ( "_sparse_" if self.conf.is_sparse else "_") ) +self.conf.dataset+"_"+str(self.conf.user_windows_size)+"_" +mode+".pkl"

        if os.path.exists(pickle_name) and not fresh:
            import gc
            gc.disable()
            samples = pickle.load(open(pickle_name, 'rb'))
            gc.enable()
            
        else:
            samples = self.prepare_dns(sess=sess,model=model,mode=mode, epoches_size=epoches_size)
            pickle.dump(samples, open(pickle_name, 'wb'),protocol=2)

        if mode=="train" and shuffle:      
            start=time.time()            
            random.shuffle(samples)
#            random.shuffle(positive_samples)
#            random.shuffle(negative_samples)                     
            print("shuffle time spent %f"% (time.time()-start))
        
#        n_batches = int(len(negative_samples)/ self.conf.batch_size)      
        n_batches = int(len(samples)/ self.conf.batch_size)      
        print("%d batch"% n_batches)
                
        for i in range(0,n_batches):
            start=time.time()                        
            batch = samples[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
            u_seqs = [pair[0] for pair in batch]
            i_seqs = [pair[1] for pair in batch]
            if not self.conf.sparse_tensor and self.conf.is_sparse:
                if pool is not None:
                    u_seqs = pool.map(sparse2dense, u_seqs)
                    i_seqs = pool.map(sparse2dense, i_seqs)
                else:
                    u_seqs = [v for v in map(sparse2dense, u_seqs)]
                    i_seqs = [v for v in map(sparse2dense, i_seqs)]          

            userids = [pair[3] for pair in batch]
            itemids = [pair[4] for pair in batch]
            ratings = [float(ii[2]) for ii in batch]
            if self.conf.sparse_tensor:
                u_seqs,i_seqs=get_sparse_intput(u_seqs,i_seqs)
            yield u_seqs,i_seqs,ratings,userids,itemids
#            batch_b = negative_samples[i*self.conf.batch_size / 2:(i+1) * self.conf.batch_size / 2]
#            batch_a = positive_samples[i*self.conf.batch_size / 2:(i+1) * self.conf.batch_size / 2]
#            batch = np.concatnate(batch_a,batch_b)
           
#            u_seqs_pos = [pair[0] for pair in batch_a]
#            i_seqs_pos = [pair[1] for pair in batch_a]
#            if self.conf.is_sparse:
#                if pool is not None:
#                    u_seqs_pos = pool.map(sparse2dense, u_seqs_pos)
#                    i_seqs_pos = pool.map(sparse2dense, i_seqs_pos)
#                else:
#                    u_seqs_pos = [v for v in map(sparse2dense, u_seqs_pos)]
#                    i_seqs_pos = [v for v in map(sparse2dense, i_seqs_pos)]          
#
#            userids_pos = [pair[3] for pair in batch_a]
#            itemids_pos = [pair[4] for pair in batch_a]
#            ratings_pos = [float(ii[2]) for ii in batch_a]
#            
#            u_seqs_neg = [pair[0] for pair in batch_b]
#            i_seqs_neg = [pair[1] for pair in batch_b]
#            if self.conf.is_sparse:
#                if pool is not None:
#                    u_seqs_neg = pool.map(sparse2dense, u_seqs_neg)
#                    i_seqs_neg = pool.map(sparse2dense, i_seqs_neg)
#                else:
#                    u_seqs_neg = [v for v in map(sparse2dense, u_seqs_neg)]
#                    i_seqs_neg = [v for v in map(sparse2dense, i_seqs_neg)]          
#
#            userids_neg = [pair[3] for pair in batch_b]
#            itemids_neg = [pair[4] for pair in batch_b]
#            ratings_neg = [float(ii[2]) for ii in batch_b]            
#            yield u_seqs_pos,i_seqs_pos,ratings_pos,userids_pos,itemids_pos, u_seqs_neg,i_seqs_neg,ratings_neg,userids_neg,itemids_neg       

    def getUserVector(self,user_sets):
        u_seqs=[]
        for user_set in user_sets:
            u_seq=[0]*(self.i_cnt)
       
            if not user_set is None:
                for index,row in user_set.iterrows():
                    u_seq[row["itemid"]]=row["rating"]
            u_seqs.append(u_seq)
        return np.array(u_seqs)
    
    
    def getItemVector(self,item_sets):
        i_seqs=[]
        for item_set in item_sets:
            i_seq=[0]*(self.u_cnt)
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

    def evaluateRMSE(self,sess,model):
        results=np.array([])
        for u_seqss,i_seqss,ratingss,useridss,itemidss in self.getDataWithSeq(mode="test",rating_flag=True):
            predicted = model.prediction(sess, u_seqss, i_seqss, useridss, itemidss)
            # print(predicted)
            # print(ratingss)
            error=(np.array(predicted)*5-np.array(ratingss))  # different optimitic indicator
            se= np.square(error)
            results=np.append(results,se)
        mse=np.mean(results)

        return math.sqrt(mse)
    
    def getDataWithSeq(self,shuffle=True,mode="train",epoches=2,rating_flag=False):

        if True:
        # try:     
            if mp:    
                pool= Pool(cpu_count())
            else:
                pool=None
            samples=self.prepare_uniform(pool,mode=mode, epoches_size=1)
            batches=samples
            for i in range(epoches):
                if mode=="train" and shuffle:      
                    batches =sklearn.utils.shuffle(batches)

                n_batches= int(len(batches)/ self.conf.batch_size)
                for i in range(0,n_batches):
                    batch = batches[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
                    if mp:
                        u_seqs=pool.map(sparse2dense, [ii[0] for ii in batch])
                        i_seqs=pool.map(sparse2dense, [ii[1] for ii in batch])
                    else:
                        u_seqs=[record for record in map(sparse2dense, [ii[0] for ii in batch])]
                        i_seqs=[record for record in map(sparse2dense, [ii[1] for ii in batch])]
                    if rating_flag:
                        ratings=[int(ii[2]) for ii in batch]
                    else:
                        ratings=[int(ii[2]>3.99) for ii in batch]
                    userids=[ii[3] for ii in batch]
                    itemids=[ii[4] for ii in batch]
                    yield u_seqs,i_seqs,ratings,userids,itemids
        # except Exception as e:
        #     print(e)
        #     # exit()
        #     pool.close()
        if mp:
            pool.close()


        
    def getTestFeedingData(self,userid, rerank_indexs):
        u_seqs=[]
        for t in range(-1*self.conf.user_windows_size,0):
            u_seqs.append(self.user_dict[userid].get(t,None))
        i_seqss=[]
        for itemid in rerank_indexs:
            i_seqs=[]
            for t in range(-1*self.conf.user_windows_size,0):
                i_seqs.append(self.item_dict[itemid].get(t,None))
            i_seqss.append(i_seqs)
        return getUserVector1(u_seqs),[i for i in map(getItemVector1, i_seqss)]
  
  
    def evaluateMultiProcess(self,sess,model,mp=False):
        users_set=self.test_users
        # print("evaluate %d users" %len(users_set))
        # users_set=self.users
        results=None
        if mp:
            # try:
            pool=Pool(cpu_count())
            results= pool.map(getScore1,zip(list(users_set), itertools.repeat(sess),itertools.repeat(model) ))
            # except:
                # pool.close()
        else:
            results= [ i for i in map(getScore1,zip(users_set, itertools.repeat(sess),itertools.repeat(model) ))]


        return np.mean(np.array(results),0)
    def get_user_sparse_input(self,user_sequence):
        _indices,_values=[],[]
        for index,(cols,rows,values)  in enumerate(user_sequence):
            _indices.extend([index,x,y]  for x,y in zip(cols,rows) )   #sorted(zip(cols,rows),key =lambda x:x[0]*2000+x[1] )
            _values.extend(values)    
        if len(_indices)==0:
            return ([[0,0,0]],[0],[len(user_sequence),self.conf.user_windows_size,self.i_cnt ])        
        user_input= (_indices,_values,[len(user_sequence),self.conf.user_windows_size,self.i_cnt ])
        return user_input
    def get_item_sparse_input(self,item_sequence):
        _indices,_values=[],[]
        for index,(cols,rows,values)  in enumerate(item_sequence):
            _indices.extend([index,x,y]  for x,y in  zip(cols,rows))
            _values.extend(values)
        if len(_indices)==0:
            return ([[0,0,0]],[0],[len(item_sequence),self.conf.user_windows_size,self.u_cnt ])
        item_input= (_indices,_values,[len(item_sequence),self.conf.user_windows_size,self.u_cnt ])
        return item_input
    def get_sparse_intput(self,user_sequence,item_sequence):
        user_input=self.get_user_sparse_input(user_sequence)
        item_input=self.get_item_sparse_input(item_sequence)
        
        return user_input,item_input



flagFactory=Singleton()
FLAGS=flagFactory.getInstance()
# helper =getHelper() 
helper=DataHelper(FLAGS)
# from oldMF import DIS
# model = DIS(helper.i_cnt, helper.u_cnt, FLAGS.mf_embedding_dim, lamda=0.1, param=None, initdelta=0.05, learning_rate=0.05)
# model = DIS(helper.i_cnt, helper.u_cnt, FLAGS.mf_embedding_dim, lamda=0.1, param=None, initdelta=0.05, learning_rate=0.05)
# model = DIS(helper.i_cnt, helper.u_cnt, FLAGS.mf_embedding_dim, lamda=0.1, param=None, initdelta=0.05, learning_rate=0.05)  


def getScore1(args):
    rerank=True
    (user_id,sess,model)=args
    
    if model is None:
        print ("there is no model, it is random guessing instead!")
        all_rating= np.random.random( len(helper.items)+1)  #[user_id]
    # all_rating = model.predictionItems(sess,user_id)[0]#[user_id] MF generated all rating to pick the highest K candidates.
    else:
        all_rating = model.predictionItems(sess,user_id) # MF rating
        
    candiate_index = helper.items - helper.pos_items.get(user_id, set())
    scores =[ (index,all_rating[index]) for index in candiate_index ]

    sortedScores = sorted(scores ,key= lambda x:x[1], reverse = True )

    pre_rank_list= [1 if ii[0] in helper.test_pos_items.get(user_id, set()) else 0 for ii in sortedScores[:10]]
    pre_result = getResult(pre_rank_list)

    if not rerank or FLAGS.model_type=="mf":
        return result


    rerank_indexs= ([ii[0] for ii in sortedScores[:helper.conf.re_rank_list_length]])
    u_seqs,i_seqss=helper.getTestFeedingData(user_id, rerank_indexs)

    # print(np.array(u_seqs).shape)
    # print(np.array(i_seqss).shape)
    
    if model is None:
        print ("there is no model, it is random guessing instead!")
        scores=np.random.random( len(rerank_indexs))
    else:
        # scores = model.prediction(sess, [u_seqs] * helper.conf.re_rank_list_length, i_seqss, [user_id] * helper.conf.re_rank_list_length, rerank_indexs)
        # u_seqs,i_seqs=helper.get_sparse_intput([u_seqs] * helper.conf.re_rank_list_length, i_seqss)

        scores = model.prediction(sess,[u_seqs] * helper.conf.re_rank_list_length, i_seqss , [user_id] * helper.conf.re_rank_list_length, rerank_indexs,use_sparse_tensor=False)

    # 
    sortedScores = sorted(zip(rerank_indexs,scores) ,key= lambda x:x[1], reverse = True )
    rank_list= [1 if ii[0] in helper.test_pos_items.get(user_id, set()) else 0 for ii in sortedScores[:10]]

    result = getResult(rank_list)
    # print(rank_list)
    # print("rerank score: %s"%(str(result-pre_result)))
    return pre_result,result

def sparse2dense(sparse):
    return sparse.toarray()

def getResult(r):

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)
    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max




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
    if helper.conf.sparse_tensor:
        return ( rows,cols ,datas)
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
    if helper.conf.sparse_tensor:
        return ( rows,cols ,datas)
    return csr_matrix((datas, (rows, cols)), shape=(helper.conf.user_windows_size, helper.i_cnt))

def getUserVector(user_sets):
   u_seqs=[]
   for user_set in user_sets:
       u_seq=[0]*(helper.i_cnt)
   
       if not user_set is None:
           for index,row in user_set.iterrows():
               u_seq[row["itemid"]]=row["rating"]
       u_seqs.append(u_seq)
   return np.array(u_seqs)
    
    
def getItemVector(item_sets):
   i_seqs=[]
   for item_set in item_sets:
       i_seq=[0]*(helper.u_cnt)
       if not item_set is None:
           for index,row in item_set.iterrows():
               i_seq[row["uid"]]=row["rating"]
       i_seqs.append(i_seq)
   return np.array(i_seqs)

def main():
    freeze_support()
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
    # helper.evaluate(None,None)
    # start = time.time() 
    # helper.evaluate(None,None)
    # print("time spented   %.6f"%(time.time()-start))
    

    start = time.time() 
    print (helper.evaluateMultiProcess(None,None))
    print("time spented %.6f"%(time.time()-start))

#    start = time.time() 
#    print ("start")
#    print (helper.evaluateMultiProcess(None,None))
#    print("time spented %.6f"%(time.time()-start))


