 
import os
 
import cPickle as pickle
import numpy
import random
from multiprocessing import Process, Queue
 
 
from collections import defaultdict
import os
import numpy as np
import math
import time 


def load_data(data_path): 
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    train_list = open(data_path,'r') 
    count=0
    for line in train_list:
        temp=line.split() 
        u=int(temp[0])  
        for j in range(1,len(temp)):
            i=int(temp[j])
            user_ratings[u].add(i)
            max_i_id = max(i, max_i_id) 
        max_u_id = max(u, max_u_id) 
    print "max_u_id:", max_u_id+1
    print "max_i_id:", max_i_id+1 
    return max_u_id, max_i_id, user_ratings  
data_path = './user_favor.txt'#os.path.join('/home/duhanjun/files/python_project/vpbr/bpr-master', 'ratings.dat')
user_id_mapping, item_id_mapping, user_ratings = load_data(data_path)
user_ratings_original=user_ratings



def generate_test(user_ratings): 
    Path_test='./mymodel_test.txt'
    wfile_test=open(Path_test,'w') 
    user_test = dict()
    for u, i_list in user_ratings.items(): 
        user_test[u] = random.sample(user_ratings[u], 1)[0] 
        wfile_test.write(str(u)+' '+str(user_test[u])+'\n') 
    wfile_test.close()
    return user_test 
test_ratings = generate_test(user_ratings) 

def generate_val(user_ratings): 
    Path_val='./mymodel_val.txt'
    wfile_val=open(Path_val,'w') 
    user_val = dict()
    for u, i_list in user_ratings.items(): 
        user_val[u] = random.sample(user_ratings[u], 1)[0]
        while user_val[u] == test_ratings[u]:
            user_val[u] = random.sample(user_ratings[u], 1)[0]
        wfile_val.write(str(u)+' '+str(user_val[u])+'\n')
    wfile_val.close()
    return user_val 
val_ratings = generate_val(user_ratings)
 
def generate_train(user_ratings,test_ratings,val_ratings): 
    Path_val='./mymodel_train.txt'
    wfile_val=open(Path_val,'w') 
    user_train = defaultdict(set)
    for u, i_list in user_ratings.items(): 
        for i in user_ratings[u]:
            if i != test_ratings[u] and i !=val_ratings[u]:
                user_train[u].add(i) 
                wfile_val.write(str(u)+' '+str(i)+'\n')
    wfile_val.close()
    return user_train 
user_ratings = generate_train(user_ratings,test_ratings,val_ratings)