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

from dataHelper  import helper
from dataHelper  import FLAGS


if __name__== "__main__":
	from multiprocessing import  freeze_support

	freeze_support()
	# flagFactory=Singleton()
	# FLAGS=flagFactory.getInstance()
	# helper=DataHelper(FLAGS)
	for x,y,z,u,i in helper.prepare():
	    print(np.array(x).shape)
	    print(u)
	    print(i)