# coding: utf-8

import tensorflow as tf
import os
 
import cPickle as pickle
import numpy
import random
from multiprocessing import Process, Queue
import tensorflow as tf
 
from collections import defaultdict
import os
import numpy as np
import math
import time 
import pdb 
import cPickle as pickle
 
 
user_ratings_original=np.load('./data/user_favor.npy').item() 
test_ratings=np.load('./data/mymodel_test.npy').item() 
val_ratings=np.load('./data/mymodel_val.npy').item()
user_ratings=np.load('./data/mymodel_train.npy').item() 

user_ups=np.load('./data/user_ups.npy')
ups_user=np.load('./data/ups_user.npy') 
user_follows=np.load('./data/user_follow.npy')  
img_feature=np.load('./data/image_feature.npy').item()
#img_feature=img_feature1/15.0 

vision_repre=np.load('./data/user_feature_contnt_style.npy').item()

#vision_repre=vision_repre1/15.0 

user_id_mapping=len(user_ratings_original)
item_id_mapping=len(ups_user) 



print 'user:',user_id_mapping,'item:',item_id_mapping
 
u_follow_all=[[]]*user_id_mapping 
u_follow_user_all=[[]]*user_id_mapping  
u_follow_split_all=[[]]*user_id_mapping  
u_vision_f_all_b=[[]]*user_id_mapping  
for u in range(user_id_mapping):
    u_follow=[] 
    u_follow_user=[]
    u_vision_f_b=[]
    u_vision_f_a=[]
    count_follow=0 
    for follow_i in user_follows[u]:  
        u_follow.append(follow_i)  
        u_follow_user.append(u)
        u_vision_f_b.append(vision_repre[follow_i]) 
        count_follow=count_follow+1  
    u_follow_all[u]=(u_follow) 
    u_follow_user_all[u]=u_follow_user
    u_follow_split_all[u]=(count_follow) 
    u_vision_f_all_b[u]=u_vision_f_b 

 
u_up_all=[[]]*user_id_mapping 
u_up_user_all=[[]]*user_id_mapping 
u_up_feature_all=[[]]*user_id_mapping 
u_up_split_all=[[]]*user_id_mapping 
for u in range(user_id_mapping):
    u_up=[]
    u_up_user=[] 
    u_up_feature=[]
    count_up=0
    for up_i in  user_ups[u]:  
        u_up.append(up_i) 
        u_up_user.append(u)
        u_up_feature.append(img_feature[up_i])
        count_up=count_up+1  
    u_up_all[u]=(u_up) 
    u_up_user_all[u]=u_up_user
    u_up_feature_all[u]=u_up_feature
    u_up_split_all[u]=(count_up)
        
 
 
def beatch_train_generator(train_ratings,train_ratings_original,user_count,item_count,batch_size,u_i,index_batch):
    t_512=[[]]*batch_size
    img_i=[[]]*batch_size
    img_j=[[]]*batch_size
    u_up_512=[[]]*batch_size
    u_up_user_512=[[]]*batch_size
    u_up_split_512=[[]]*batch_size
    u_up_feature_512=[[]]*batch_size
    u_follow_512=[[]]*batch_size
    u_follow_user_512=[[]]*batch_size
    u_follow_split_512=[[]]*batch_size 

    vision_repre_512_a=[[]]*batch_size 
    vision_repre_512_b=[[]]*batch_size 
    vision_repre_u_aspect_512=[[]]*batch_size

    gather_upload=[]
    gather_social=[]
    #start1=time.time()
    count=0
    for u,i in u_i: 
        t = []  
        j = np.random.randint(item_count)
        while j in train_ratings_original[u]:
            j = np.random.randint(item_count)  
      
        t_512[count]=([u,i,ups_user[i],j,ups_user[j]]) 
        vision_repre_512_a[count]=vision_repre[u] 
        vision_repre_512_b[count]=u_vision_f_all_b[u]  
 
        u_up_512[count]=u_up_all[u]#(u_up)
        u_up_user_512[count]= u_up_user_all[u]#(u_up_user)
        u_up_split_512[count]= u_up_split_all[u]#(count_up )
        u_up_feature_512[count]=u_up_feature_all[u]#(u_up_feature) 

        #np.zeros(u_up_split_all[u])+count
        gather_upload=np.concatenate((gather_upload,np.zeros(u_up_split_all[u])+count))
       # for i in range(u_up_split_all[u]):
           # gather_upload.append(count) 

        u_follow_512[count]=u_follow_all[u]#(u_follow)
        u_follow_user_512[count]=u_follow_user_all[u]#(u_follow_user)
        u_follow_split_512[count]=u_follow_split_all[u]#(count_follow) 
        #for i in range(u_follow_split_all[u]):
            #gather_social.append(count) 
        gather_social=np.concatenate((gather_social,np.zeros(u_follow_split_all[u])+count))
        count=count+1 
         
    one_part=([numpy.asarray(t_512),numpy.asarray(gather_upload),numpy.asarray(gather_social),numpy.asarray(vision_repre_512_a),numpy.asarray(vision_repre_512_b),numpy.asarray(u_up_512), numpy.asarray(u_up_user_512),numpy.asarray(u_up_split_512),numpy.asarray(u_up_feature_512),numpy.asarray(u_follow_512),numpy.asarray(u_follow_user_512),numpy.asarray(u_follow_split_512)])
    #start2=time.time()
    #print  'one_part:',start2-start1
    return one_part
   

 
def one_train_generator(train_ratings,train_ratings_original,user_count,item_count,batch_size): 
    all_result=[]
    result=dict()
    result_count=0
    for u in range(user_count):  
        for i in train_ratings[u]: 
            result[result_count]=[u,i] 
            result_count=result_count+1
    result_count=result_count-1
    result_all_train_u=range(result_count)
    random.shuffle(result_all_train_u)   
    batch_size_u_i=[]   
    add_to_512=batch_size-result_count%batch_size 
    for k in range(add_to_512): 
        result_all_train_u.append(result_all_train_u[k]) 
    print len(result_all_train_u)/batch_size 
    index_batch=0
    count=1
    for index_ in result_all_train_u:
        #u,i=dict[index_] 
        batch_size_u_i.append(result[index_] )
        if count%batch_size==0:   
            temp_data=beatch_train_generator(train_ratings,train_ratings_original,user_count,item_count,batch_size,batch_size_u_i,str(index_batch))
            all_result.append(temp_data)
            index_batch=index_batch+1
            count=0
            batch_size_u_i=[] 
        count=count+1  
    return all_result
'''
start_time=time.time()    
print 'result_all_train start'
all_result_train=one_train_generator(user_ratings,user_ratings_original,user_id_mapping,item_id_mapping,512)
start_time2=time.time()    
print 'result_all_train end',start_time2-start_time,len(all_result_train)
'''
 

def one_test_val_generator(test_or_val_ratings,train_ratings_original,user_count,item_count,batch_size): 
    all_result=[]
    result=dict()
    result_count=0
    for u in range(user_count):  
        i =test_or_val_ratings[u]
        result[result_count]=[u,i] 
        result_count=result_count+1
    result_count=result_count-1
    result_all_train_u=range(result_count)
    random.shuffle(result_all_train_u)   
    batch_size_u_i=[]   
    add_to_512=batch_size-result_count%batch_size 
    for k in range(add_to_512): 
        result_all_train_u.append(result_all_train_u[k]) 
    print len(result_all_train_u)/batch_size 
    index_batch=0
    count=1
    for index_ in result_all_train_u:
        #u,i=dict[index_] 
        batch_size_u_i.append(result[index_] )
        if count%batch_size==0:   
            temp_data=beatch_train_generator(test_or_val_ratings,train_ratings_original,user_count,item_count,batch_size,batch_size_u_i,str(index_batch))
            all_result.append(temp_data)
            index_batch=index_batch+1
            count=0
            batch_size_u_i=[] 
        count=count+1  
    return all_result

'''
start_time=time.time()    
print 'result_all_train start'
all_result_test=one_test_val_generator(test_ratings,user_ratings_original,user_id_mapping,item_id_mapping,512)
start_time2=time.time()    
print 'result_all_train end',start_time2-start_time
'''
def generate_negative_100(train_ratings_original, user_ratings_test,item_count):
    t=[]  
    for u in train_ratings_original.keys():  
        i_p=user_ratings_test[u]
        temp=[] 
        rand_200=[random.randint(1, item_count) for _ in range(200)]
        for sel_100 in rand_200:  
            j_ng = sel_100
            if not (j_ng in train_ratings_original[u]): 
                temp.append(j_ng) 
        t.append([u,i_p,temp])  
    return t 
    
 

def Upload_influence_speed(batch_size,u_ups,u_ups_user,user_emb_p,user_emb_q,item_emb_x,item_emb_w,vision_,e_aj_w,e_aj_b,gather_upload):  
    
    u_p=(tf.nn.embedding_lookup(user_emb_p, u_ups_user))
    u_q=(tf.nn.embedding_lookup(user_emb_q, u_ups_user))  
    x_up_img=(tf.nn.embedding_lookup(item_emb_x, u_ups))
    w_up_img=(tf.nn.embedding_lookup(item_emb_w, u_ups)) 
    x_up=tf.concat([u_p,u_q,x_up_img,w_up_img,vision_],1)   #size_up*60   
    e_aj_temp=tf.nn.elu(tf.matmul(x_up,e_aj_w)+e_aj_b)#(size_up*60 * 60*20)+20=size_up*20
    e_aj_temp_sum=(tf.reduce_sum(e_aj_temp,1, keep_dims=True))
    e_aj_temp_sum=tf.where(e_aj_temp_sum>88,tf.ones_like(e_aj_temp_sum)*88,e_aj_temp_sum) 
    e_aj=e_aj_temp_sum#tf.exp(e_aj_temp_sum)+0.001#size_up*1 
    molecular_e_aj=tf.multiply(e_aj,x_up_img)#size_up*15
    denominator_e_aj=e_aj#size_up*1 
  
    part_mole=tf.segment_sum(molecular_e_aj,gather_upload)
    part_denom=tf.segment_sum(denominator_e_aj,gather_upload)

    alpha_all=tf.multiply(part_mole,tf.reciprocal(part_denom))
    alpha_all=tf.where(tf.is_nan(alpha_all),tf.ones_like(alpha_all)*0.001,alpha_all)  
    return alpha_all
 
def Social_influence_one_cal(batch_size,u_follows,u_follows_user,user_emb_p,user_emb_q,vision_beta_a,vision_beta_b,e_ab_w,e_ab_b,gather_social):
    #social influence
   
    follow_pa=(tf.nn.embedding_lookup(user_emb_p, u_follows))
    follow_qa=(tf.nn.embedding_lookup(user_emb_q, u_follows))  
    follow_pb=(tf.nn.embedding_lookup(user_emb_p, u_follows_user))
    follow_qb=(tf.nn.embedding_lookup(user_emb_q, u_follows_user))  
    x_follow=tf.concat([follow_pa,follow_pb,follow_qa,follow_qb,vision_beta_a,vision_beta_b],1)   #size_up*60   
    e_ab_temp=tf.nn.elu(tf.matmul(x_follow,e_ab_w)+e_ab_b)#(size_up*60 * 60*20)+20=size_up*20

    e_ab_temp_sum=(tf.reduce_sum(e_ab_temp,1, keep_dims=True))
    e_ab_temp_sum=tf.where(e_ab_temp_sum>88,tf.ones_like(e_ab_temp_sum)*88,e_ab_temp_sum)  

    e_ab=e_ab_temp_sum#tf.exp(e_ab_temp_sum)+0.001#tf.exp((tf.reduce_sum(e_ab_temp,1, keep_dims=True)))#size_up*1 
    molecular_e_ab=tf.multiply(e_ab,follow_qb)#size_up*15
    denominator_e_ab=e_ab#size_up*1   
   
    part_mole=tf.segment_sum(molecular_e_ab,gather_social)
    part_denom=tf.segment_sum(denominator_e_ab,gather_social) 

    beta_all=tf.multiply(part_mole,tf.reciprocal(part_denom))
    beta_all=tf.where(tf.is_nan(beta_all),tf.ones_like(beta_all)*0.001,beta_all)   
    return beta_all
 
def Factor_importance(alpha,beta,u_emb_base,u_emb_external,uploader_influence_i,vision_aspect,h_f_w,h_f_b):
    #I_l_a=tf.reshape(I_l_a,[batch_size,1]) 
    f1=tf.concat([alpha,u_emb_base,u_emb_external,vision_aspect],1)   
    f2=tf.concat([beta,u_emb_base,u_emb_external,vision_aspect],1)  
    f3=tf.concat([uploader_influence_i,u_emb_base,u_emb_external,vision_aspect],1) 

    e_a_1_temp=tf.reduce_sum(tf.nn.elu(tf.matmul(f1,h_f_w)+h_f_b)) 
    e_a_1_temp=tf.where(e_a_1_temp>88,tf.ones_like(e_a_1_temp)*88,e_a_1_temp)  
    e_a_1=e_a_1_temp#tf.exp(e_a_1_temp)+0.001#batch_size*

    e_a_2_temp=tf.reduce_sum(tf.nn.elu(tf.matmul(f2,h_f_w)+h_f_b)) 
    e_a_2_temp=tf.where(e_a_2_temp>88,tf.ones_like(e_a_2_temp)*88,e_a_2_temp)  
    e_a_2=e_a_2_temp#tf.exp(e_a_2_temp)+0.001#batch_size*

    e_a_3_temp=tf.reduce_sum(tf.nn.elu(tf.matmul(f3,h_f_w)+h_f_b)) 
    e_a_3_temp=tf.where(e_a_3_temp>88,tf.ones_like(e_a_3_temp)*88,e_a_3_temp)  
    e_a_3=e_a_3_temp#tf.exp(e_a_3_temp)+0.001#batch_size* 
    denominator_e_ai=e_a_1+e_a_2+e_a_3#tf.add(tf.add(e_a_1,e_a_2),e_a_3)
    gamma_a1= e_a_1/denominator_e_ai#tf.reduce_sum(e_a_1/denominator_e_ai,1, keep_dims=True)
    gamma_a2= e_a_2/denominator_e_ai#tf.reduce_sum(e_a_2/denominator_e_ai,1, keep_dims=True)
    gamma_a3= e_a_3/denominator_e_ai#tf.reduce_sum(e_a_3/denominator_e_ai,1, keep_dims=True)
    return [gamma_a1,gamma_a2,gamma_a3]

  
user_count = (user_id_mapping)-1
item_count = (item_id_mapping)-1 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
with tf.Graph().as_default(), tf.Session() as session:
   
    
    batch_size = 512
    #u, i,i_uploader,I_li_a,j,j_uploader,I_lj_a,u_ups,u_ups_user,u_ups_split,u_follows,u_follows_user,u_follows_split,loss, auc,my_get, train_op,train_ = vbpr(user_count, item_count,batch_size) 
    #user_count, item_count
    hidden_dim=15 
    hidden_img_dim=15
    hidden_dim_upload=20
    hidden_dim_social=20
    hidden_dim_factor=20
    learning_rate = 0.0005
    l2_regulization = 0.01
    bias_regulization=1.0

    u = tf.placeholder(tf.int32, [None])
    vision_repre_u_a=tf.placeholder(tf.float32, [None,1808])
    vision_repre_u_b=tf.placeholder(tf.float32, [None,1808]) 
    gather_upload=tf.placeholder(tf.int32, [None])
    gather_social=tf.placeholder(tf.int32, [None])

    i = tf.placeholder(tf.int32, [None])
    i_uploader=tf.placeholder(tf.int32, [None]) 
    j = tf.placeholder(tf.int32, [None])  
    j_uploader=tf.placeholder(tf.int32, [None]) 

    u_ups = tf.placeholder(tf.int32,[None]) 
    u_ups_user = tf.placeholder(tf.int32,[None]) 
    u_ups_split = tf.placeholder(tf.int32,[None]) 
    u_ups_feature=tf.placeholder(tf.float32, [None,1808]) 

    u_follows = tf.placeholder(tf.int32,[None]) 
    u_follows_user = tf.placeholder(tf.int32,[None]) 
    u_follows_split = tf.placeholder(tf.int32,[None])   
    train_ =tf.placeholder(tf.int32,[None])  

    user_emb_p= tf.get_variable("user_emb_p", [user_count+1, hidden_dim],
                        initializer=tf.random_normal_initializer(0, 0.01))
    user_emb_q= tf.get_variable("user_emb_q", [user_count+1, hidden_dim], 
                        initializer=tf.random_normal_initializer(0, 0.01))
    item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim], 
                            initializer=tf.random_normal_initializer(0, 0.01))
    item_emb_x = tf.get_variable("item_emb_x", [item_count+1, hidden_dim], 
                            initializer=tf.random_normal_initializer(0, 0.01))
    item_b = tf.get_variable("item_b", [item_count+1, 1], 
                            initializer=tf.constant_initializer(0.0))
    #upload influence feedforward neural net
    e_aj_w=tf.get_variable("e_aj_w", [hidden_dim*5, hidden_dim_upload],
                            initializer=tf.random_normal_initializer(0, 0.01))
    e_aj_b=tf.get_variable("e_aj_b", [hidden_dim_upload], 
                            initializer=tf.constant_initializer(0.0))
    #upload influence vision part
    e_aj_vision_w=tf.get_variable("e_aj_vision_w", [1808, hidden_dim], 
                            initializer=tf.random_normal_initializer(0, 0.01)) 

    #soclai influence feedforward neural net
    e_ab_w=tf.get_variable("e_ab_w", [hidden_dim*6, hidden_dim_social], 
                            initializer=tf.random_normal_initializer(0, 0.01))
    e_ab_b=tf.get_variable("e_ab_b", [hidden_dim_social], 
                            initializer=tf.constant_initializer(0.0))
    e_ab_vision_w=tf.get_variable("e_ab_vision_w", [1808, hidden_dim], 
                            initializer=tf.random_normal_initializer(0, 0.01))  
 
    #factor importance feedforward neural net
    h_f_w=tf.get_variable("h_f_w", [hidden_dim, hidden_dim_factor], 
                            initializer=tf.random_normal_initializer(0, 0.01))
    h_f_b=tf.get_variable("h_f_b", [hidden_dim_factor], 
                            initializer=tf.constant_initializer(0.0))  
    h_f_vision_w=tf.get_variable("h_f_vision_w", [1808, hidden_dim], 
                            initializer=tf.random_normal_initializer(0, 0.01))  
 
    user_emb_p_1=user_emb_p
    user_emb_q_1=user_emb_q
    item_emb_w_1=item_emb_w
    item_emb_x_1=item_emb_x
    item_b_1=item_b
    
 
    
    vision_alpha=tf.gather(tf.matmul(vision_repre_u_a,e_aj_vision_w),gather_upload)
    alpha=Upload_influence_speed(batch_size,u_ups,u_ups_user,user_emb_p,user_emb_q,item_emb_x,item_emb_w,vision_alpha,e_aj_w,e_aj_b,gather_upload)  
    #vision_alpha=tf.matmul(u_ups_feature,e_aj_vision_w)
     
    vision_beta_a=tf.gather(tf.matmul(vision_repre_u_a,e_ab_vision_w),gather_social)
    vision_beta_b=(tf.matmul(vision_repre_u_b,e_ab_vision_w))

    # beta=Social_influence_one_cal(batch_size,u_follows,u_follows_user,u_follows_split,user_emb_p_1,user_emb_q_1,vision_beta_a,vision_beta_b,e_ab_w,e_ab_b) 
    beta=Social_influence_one_cal(batch_size,u_follows,u_follows_user,user_emb_p_1,user_emb_q_1,vision_beta_a,vision_beta_b,e_ab_w,e_ab_b,gather_social) 
    
    #uploader Influence
    uploader_influence_i=tf.nn.embedding_lookup(user_emb_q_1, i_uploader)
    uploader_influence_j=tf.nn.embedding_lookup(user_emb_q_1, j_uploader)
   
    u_emb_base = tf.nn.embedding_lookup(user_emb_p_1, u)
    u_emb_external = tf.nn.embedding_lookup(user_emb_q_1, u)

    i_emb_base = tf.nn.embedding_lookup(item_emb_w_1, i)
    i_emb_external = tf.nn.embedding_lookup(item_emb_x_1, i)

    j_emb_base = tf.nn.embedding_lookup(item_emb_w_1, j)
    j_emb_external = tf.nn.embedding_lookup(item_emb_x_1, j) 

    item_b_i=tf.nn.embedding_lookup(item_b_1, i)
    item_b_j=tf.nn.embedding_lookup(item_b_1, j) 
       
    e_a_1_temp=tf.reduce_sum((tf.matmul(alpha,h_f_w)+h_f_b))  
    e_a_1_temp=tf.where(e_a_1_temp>88,tf.ones_like(e_a_1_temp)*88,e_a_1_temp)  
    e_a_1=tf.exp(e_a_1_temp)+0.001 

    e_a_2_temp=tf.reduce_sum((tf.matmul(beta,h_f_w)+h_f_b))
    e_a_2_temp=tf.where(e_a_2_temp>88,tf.ones_like(e_a_2_temp)*88,e_a_2_temp)    
    e_a_2=tf.exp(e_a_2_temp)+0.001 

    e_a_3_temp=tf.reduce_sum((tf.matmul(uploader_influence_i,h_f_w)+h_f_b))  
    e_a_3_temp=tf.where(e_a_3_temp>88,tf.ones_like(e_a_3_temp)*88,e_a_3_temp)     
    e_a_3=tf.exp(e_a_3_temp)+0.001

    e_a_3_temp_j=tf.reduce_sum((tf.matmul(uploader_influence_j,h_f_w)+h_f_b))
    e_a_3_temp_j=tf.where(e_a_3_temp_j>88,tf.ones_like(e_a_3_temp_j)*88,e_a_3_temp_j)    
    e_a_3_j=tf.exp(e_a_3_temp_j)+0.001
   
     
    denominator_e_ai=e_a_1+e_a_2+e_a_3#tf.add(tf.add(e_a_1,e_a_2),e_a_3)
    gamma_a1_i= e_a_1/denominator_e_ai#tf.reduce_sum(e_a_1/denominator_e_ai,1, keep_dims=True)
    gamma_a2_i= e_a_2/denominator_e_ai#tf.reduce_sum(e_a_2/denominator_e_ai,1, keep_dims=True)
    gamma_a3_i= e_a_3/denominator_e_ai#tf.reduce_sum(e_a_3/denominator_e_ai,1, keep_dims=True)

    denominator_e_ai=e_a_1+e_a_2+e_a_3_j#tf.add(tf.add(e_a_1,e_a_2),e_a_3)
    gamma_a1_j= e_a_1/denominator_e_ai#tf.reduce_sum(e_a_1/denominator_e_ai,1, keep_dims=True)
    gamma_a2_j= e_a_2/denominator_e_ai#tf.reduce_sum(e_a_2/denominator_e_ai,1, keep_dims=True)
    gamma_a3_j= e_a_3_j/denominator_e_ai#tf.reduce_sum(e_a_3/denominator_e_ai,1, keep_dims=True)
 
    
   
    R_ai_temp=(u_emb_base+gamma_a1_i*alpha+gamma_a2_i*beta+gamma_a3_i*uploader_influence_i) 
    R_ai= tf.diag_part(tf.matmul(R_ai_temp,tf.transpose(i_emb_base))) 
    
    R_aj_temp=(u_emb_base+gamma_a1_j*alpha+gamma_a2_j*beta+gamma_a3_j*uploader_influence_j) 
    R_aj= tf.diag_part(tf.matmul(R_aj_temp,tf.transpose(j_emb_base))) 
     
    x= item_b_i-item_b_j+tf.add(R_ai,-R_aj) 
    #auc=tf.Variable(0) 
    
    
    auc =[user_emb_p_1,user_emb_q_1,item_emb_w_1,item_emb_x_1,item_b_1,e_aj_w,e_aj_b,e_aj_vision_w,e_ab_w,e_ab_b,e_ab_vision_w,h_f_w,h_f_b,h_f_vision_w]#tf.reduce_mean(tf.to_float(x > 0))
    my_get=[alpha,beta,uploader_influence_i,uploader_influence_j,\
        R_ai,R_aj,item_b_i,item_b_j,R_ai_temp,i_emb_base,R_aj_temp,j_emb_base,u_emb_base,\
        gamma_a1_i,gamma_a2_i,gamma_a3_i,gamma_a1_j,gamma_a2_j,gamma_a3_j]#,\
    #u,i,j,alpha_all,molecular_e_aj,denominator_e_aj,split_index_up]#[alpha,x_up,e_aj_temp,e_aj,molecular_e_aj,denominator_e_aj]#[x,tf.sigmoid(x),tf.log(1+tf.sigmoid(x)),alpha,u_emb_base]
          
    loss=-tf.reduce_mean(tf.log(tf.sigmoid(0.1*x)))
    learning_rate_get =tf.cond((tf.count_nonzero(train_))>=2, lambda:learning_rate,lambda:0.0)
    train_op=tf.train.AdamOptimizer(learning_rate_get).minimize(loss)   

    session.run(tf.global_variables_initializer())   
     

    saver = tf.train.Saver([user_emb_p,item_emb_w,item_b])#,e_aj_w,e_aj_b])  
    saver.restore(session, '../bpr_model/mymodel199.ckpt') 

    for epoch in range(1, 130):
        Path_vbpr_train='./results_my_elu_exp_0.0001r/mymodel_train_result_all.txt'
        wfile_vbpr_train=open(Path_vbpr_train,'a')  
        if epoch==1:
            wfile_vbpr_train.write('\n+mymodel +199bpr model+ 0.1*x')
        print "epoch:", epoch
        _loss_train = 0.0 
        temp_count=0
        time_start=time.time()
        laster_loss=0.0 
        time_start=time.time()
        count =0 
        
        #train_batch=train_batch_generator_all(result_all_train,result_all_train_u,result_train_count, batch_size)
        time_start1=time.time()
        train_batch=one_train_generator(user_ratings,user_ratings_original,user_id_mapping,item_id_mapping,batch_size)
        sample_count =len(train_batch)
        time_start2=time.time()
        print time_start2-time_start1
        flag=1
        #train_batch_generator(result_all_train,result_train_count, sample_count, batch_size) 
        for  d,gather_upload_,gather_social_,vision_repre_a_,vision_repre_b_,up_,up_user,up_split,up_feature,follow_,follow_user,follow_split in train_batch:
            #pdb.set_trace()
           # print len(gather_upload_),len(gather_social_)
            #time_start3=time.time()
            #break 
            vision_repre_b_end=[] 
            for i_f in vision_repre_b_:
                for j_f in i_f:
                    vision_repre_b_end.append(j_f)

            up_feature_end=[]
            for i_f in up_feature:
                for j_f in i_f:
                    up_feature_end.append(j_f)
   
            up_end=[]
            for i_up in up_:
                for j_up in i_up:
                    up_end.append(j_up)

            up_user_end=[]
            for i_up_u in up_user:
                for j_up_u in i_up_u:
                    up_user_end.append(j_up_u)

            follow_user_end=[]
            for i_follow_u in follow_user:
                for j_follow_u in i_follow_u:
                    follow_user_end.append(j_follow_u) 

            follow_end=[]
            for i_follow in follow_:
                for j_follow in i_follow:
                    follow_end.append(j_follow) 

            if flag==1:
               time_start4=time.time()  
            _loss, _ ,auc_,get_= session.run([loss, train_op,auc,my_get], feed_dict={
                    u:d[:,0],gather_upload:gather_upload_,gather_social:gather_social_, vision_repre_u_a:vision_repre_a_, vision_repre_u_b:vision_repre_b_end,\
                    i:d[:,1],i_uploader:d[:,2],j:d[:,3], j_uploader:d[:,4],\
                    u_ups:up_end,u_ups_user:up_user_end,u_ups_split:up_split,u_ups_feature:up_feature_end,u_follows:follow_end,\
                    u_follows_user:follow_user_end,u_follows_split:follow_split,train_:[1,2,3]
                }) 
            count=count+1
            _loss_train += _loss
            laster_loss=_loss
            temp_count=temp_count+1 
            if flag==1:  
                time_start5=time.time()
                print 'time:',time_start5-time_start4,#,time_start4-time_start3 
            #print _loss, 
            #pdb.set_trace()
            #exit()
            if flag==1:
                print _loss
                flag=0
            if math.isnan(_loss): 
                print count-1
                pdb.set_trace() 
                exit() 
            #break
        train_loss=round(_loss_train/sample_count,4)
        time_end=time.time() 
        cost_time=round(time_end-time_start,4)
        print  cost_time,train_loss
        #if epoch>1:
        #    load_test('./my_model/mymodel'+str(epoch-1)+'.ckpt',train_batch,sample_count,user_count, item_count,batch_size)

        wfile_vbpr_train.write('\nepoch:'+str(epoch)+',\tCompute Loss Cost:'+str(cost_time)+'s, ') 
        wfile_vbpr_train.write('Train_loss:'+str(train_loss)+', ')  
 

        count=0
        NDCG_test=[0]*51   
        hint_test=[0]*51
        NDCG_val=[0]*51   
        hint_val=[0]*51
        _loss_val = 0.0 
        _loss_test = 0.0 
        sum_coun=user_count  


        all_result_val=one_test_val_generator(val_ratings,user_ratings_original,user_id_mapping,item_id_mapping,batch_size) 
        sample_count =len(all_result_val)


        #pdb.set_trace()
        for d,gather_upload_,gather_social_,vision_repre_a_,vision_repre_b_,up_,up_user,up_split,up_feature,follow_,follow_user,follow_split in all_result_val:# train_batch_generator_all(result_all_val,result_all_val_u,result_val_count, batch_size):
        # train_batch_generator(result_all_val,result_val_count, sample_count, batch_size):
            #break
            vision_repre_b_end=[] 
            for i_f in vision_repre_b_:
                for j_f in i_f:
                    vision_repre_b_end.append(j_f)

            up_feature_end=[]
            for i_f in up_feature:
                for j_f in i_f:
                    up_feature_end.append(j_f)
   
            up_end=[]
            for i_up in up_:
                for j_up in i_up:
                    up_end.append(j_up)

            up_user_end=[]
            for i_up_u in up_user:
                for j_up_u in i_up_u:
                    up_user_end.append(j_up_u)

            follow_user_end=[]
            for i_follow_u in follow_user:
                for j_follow_u in i_follow_u:
                    follow_user_end.append(j_follow_u) 

            follow_end=[]
            for i_follow in follow_:
                for j_follow in i_follow:
                    follow_end.append(j_follow) 
            #time_start4=time.time()
            
            _loss,val_auc= session.run([loss,auc], feed_dict={
                    u:d[:,0],gather_upload:gather_upload_,gather_social:gather_social_, vision_repre_u_a:vision_repre_a_, vision_repre_u_b:vision_repre_b_end,\
                    i:d[:,1],i_uploader:d[:,2],j:d[:,3], j_uploader:d[:,4],\
                    u_ups:up_end,u_ups_user:up_user_end,u_ups_split:up_split,u_ups_feature:up_feature_end,u_follows:follow_end,\
                    u_follows_user:follow_user_end,u_follows_split:follow_split,train_:[0,0,0]
                })  
            _loss_val += _loss 
            #print _loss
            #break
        saver = tf.train.Saver()
        print 'saver'
        saver.save(session, './results_my_elu_exp_0.0001r/my_model_elu_exp/mymodel'+str(epoch)+'.ckpt')
        print 'saver end'
      
        #if epoch%5!=1:
            #continue 

        all_result_test=one_test_val_generator(test_ratings,user_ratings_original,user_id_mapping,item_id_mapping,batch_size)
        sample_count =len(all_result_test)   
        for  d,gather_upload_,gather_social_,vision_repre_a_,vision_repre_b_,up_,up_user,up_split,up_feature,follow_,follow_user,follow_split in all_result_test:#train_batch_generator_all(result_all_test,result_all_test_u,result_test_count, batch_size):
        #train_batch_generator(result_all_test,result_test_count, sample_count, batch_size):
           
            vision_repre_b_end=[] 
            for i_f in vision_repre_b_:
                for j_f in i_f:
                    vision_repre_b_end.append(j_f)

            up_feature_end=[]
            for i_f in up_feature:
                for j_f in i_f:
                    up_feature_end.append(j_f)
   
            up_end=[]
            for i_up in up_:
                for j_up in i_up:
                    up_end.append(j_up)

            up_user_end=[]
            for i_up_u in up_user:
                for j_up_u in i_up_u:
                    up_user_end.append(j_up_u)

            follow_user_end=[]
            for i_follow_u in follow_user:
                for j_follow_u in i_follow_u:
                    follow_user_end.append(j_follow_u) 

            follow_end=[]
            for i_follow in follow_:
                for j_follow in i_follow:
                    follow_end.append(j_follow)  
            #time_start4=time.time()
            test_loss,test_auc= session.run([loss,auc], feed_dict={
                    u:d[:,0],gather_upload:gather_upload_,gather_social:gather_social_, vision_repre_u_a:vision_repre_a_, vision_repre_u_b:vision_repre_b_end,\
                    i:d[:,1],i_uploader:d[:,2],j:d[:,3], j_uploader:d[:,4],\
                    u_ups:up_end,u_ups_user:up_user_end,u_ups_split:up_split,u_ups_feature:up_feature_end,u_follows:follow_end,\
                    u_follows_user:follow_user_end,u_follows_split:follow_split,train_:[0,0,0]
                }) 
            auc_get=test_auc
            _loss_test += test_loss 
            #break
            #break
            
        val_loss=round(_loss_val/sample_count,4)
        print 'val_loss',val_loss
        wfile_vbpr_train.write('Val Loss:'+str(val_loss)+', ') 

        test_loss=round(_loss_test/sample_count,4)
        print 'test_loss',test_loss
        wfile_vbpr_train.write('Test Loss:'+str(test_loss)+'\n') 
  
        [user_emb_p_1,user_emb_q_1,item_emb_w_1,item_emb_x_1,item_b_1,e_aj_w_1,e_aj_b_1,e_aj_vision_w_1,e_ab_w_1,e_ab_b_1,e_ab_vision_w_1,h_f_w_1,h_f_b_1,h_f_vision_w_1]=auc_get 
      
        i_j_100_val=generate_negative_100(user_ratings_original, val_ratings,item_count)
        for user_id in range (0,sum_coun): 
            user_sel=i_j_100_val[user_id][0]
            i_id=i_j_100_val[user_id][1]
            #print user_sel,i_id
            molecular_e_aj=[0.0,0.0,0.0,0.0,0.0,0,0,0,0,0,0,0,0,0,0]
            denominator_e_aj=0.0 
            
            user_sel_list=[] 
            user_sel_feature=[] 
            user_sel_feature1=[]
            for i_usersel in range(0,len(user_ups[user_sel])):
                user_sel_list.append(user_sel)
                #user_sel_feature=np.concatenate((user_sel_feature,img_feature[i_usersel]),0)
                user_sel_feature1.append(vision_repre[user_sel])
            #temp=np.take(user_emb_p,user_sel_list,0)
            #print temp
            #pdb.set_trace()
            vision_user_up=np.matmul(user_sel_feature1,e_aj_vision_w_1)
            #vision_user_up=np.matmul(user_sel_feature1,e_aj_vision_w_1) 
            x_up2=np.concatenate([np.take(user_emb_p_1,user_sel_list,0),np.take(user_emb_q_1,user_sel_list,0),\
                np.take(item_emb_x_1,user_ups[user_sel],0),np.take(item_emb_w_1,user_ups[user_sel],0),vision_user_up],1)
            e_aj_temp2=(numpy.dot(x_up2,e_aj_w_1)+e_aj_b_1)
            e_aj2=np.exp(np.sum(e_aj_temp2,1))+0.001 
            alpha= np.matmul((e_aj2).T,(np.take(item_emb_x_1,user_ups[user_sel],0)))/np.sum(e_aj2)
          
            user_sel_list=[]
            user_sel_feature_a=[]
            user_sel_feature_b=[]
            for i_usersel in range(0,len(user_follows[user_sel])):
                user_sel_list.append(user_sel)
                user_sel_feature_a.append(vision_repre[user_sel])
                user_sel_feature_b.append(vision_repre[i_usersel])
            
            vision_user_social_a=np.matmul(user_sel_feature_a,e_ab_vision_w_1)
            vision_user_social_b=np.matmul(user_sel_feature_b,e_ab_vision_w_1)
            #u_vision_f_all
            x_follow2=np.concatenate([np.take(user_emb_p_1,user_sel_list,0),np.take(user_emb_p_1,user_follows[user_sel],0),\
                np.take(user_emb_q_1,user_sel_list,0),np.take(user_emb_q_1,user_follows[user_sel],0),vision_user_social_a,vision_user_social_b],1)
            e_ab_temp2=(numpy.dot(x_follow2,e_ab_w_1)+e_ab_b_1) 
            e_ab2=np.exp(np.sum(e_ab_temp2,1))+0.001   
            beta= np.matmul((e_ab2).T,(np.take(user_emb_q_1,user_follows[user_sel],0)))/np.sum(e_ab2)  
             

            u_emb_base=user_emb_p_1[user_sel]
            u_emb_external=user_emb_q_1[user_sel]
            
            vision_user_ql=np.matmul(vision_repre[user_sel],h_f_vision_w_1)
            uploader_influence_i=user_emb_q_1[ups_user[i_id]] 
            e_a_1=np.exp(np.sum((numpy.dot(alpha,h_f_w_1)+h_f_b_1)))+0.01   
            e_a_2=np.exp(np.sum((numpy.dot(beta,h_f_w_1)+h_f_b_1)))+0.01  #tf.exp(np.sum(tf.nn.relu(tf.matmul(f2,h_f_w)+h_f_b)))  
            e_a_3=np.exp(np.sum((numpy.dot(uploader_influence_i,h_f_w_1)+h_f_b_1)))+0.01  #tf.exp(np.sum(tf.nn.relu(tf.matmul(f3,h_f_w)+h_f_b)))   
            denominator_e_ai=e_a_1+e_a_2+e_a_3
            gamma_a1= e_a_1/denominator_e_ai
            gamma_a2= e_a_2/denominator_e_ai
            gamma_a3= e_a_3/denominator_e_ai
             
            user_i_temp=(u_emb_base+gamma_a1*alpha+gamma_a2*beta+gamma_a3*uploader_influence_i)
            #u_emb_base+0.5*alpha+0.5*beta#
            user_i=item_b_1[i_id]+np.dot(user_i_temp,(item_emb_w_1[i_id]).T)  

            idx_=1  
            negative100=i_j_100_val[user_id][2]
            #time2=time.time() 
            for sel_100 in range(0,100):   
                j_ng = negative100[sel_100]    
                uploader_influence_i=user_emb_q_1[ups_user[j_ng]]   
                e_a_1=np.exp(np.sum((numpy.dot(alpha,h_f_w_1)+h_f_b_1)))+0.01   
                e_a_2=np.exp(np.sum((numpy.dot(beta,h_f_w_1)+h_f_b_1)))+0.01  #tf.exp(np.sum(tf.nn.relu(tf.matmul(f2,h_f_w)+h_f_b)))  
                e_a_3=np.exp(np.sum((numpy.dot(uploader_influence_i,h_f_w_1)+h_f_b_1)))+0.01  #tf.exp(np.sum(tf.nn.relu(tf.matmul(f3,h_f_w)+h_f_b)))    
                denominator_e_ai=e_a_1+e_a_2+e_a_3
                gamma_a1= e_a_1/denominator_e_ai
                gamma_a2= e_a_2/denominator_e_ai
                gamma_a3= e_a_3/denominator_e_ai
                user_j_temp=(u_emb_base+gamma_a1*alpha+gamma_a2*beta+gamma_a3*uploader_influence_i)
                #u_emb_base+0.5*alpha+0.5*beta#
                user_j=item_b_1[j_ng]+np.dot(user_j_temp,(item_emb_w_1[j_ng]).T)  
                #user_j=item_b[j_ng]+user_item_[user_sel,j_ng]+numpy.dot(numpy.transpose(user_img_w[user_sel]),img_feature[j_ng])+img_feature_b[j_ng]#(,(numpy.dot(image_features[j_ng],img_emb_w)))
                if user_j>user_i: 
                    idx_=idx_+1     
            #time3=time.time() 
            if idx_<=50: 
                NDCG_val[idx_]=NDCG_val[idx_]+(math.log(2))/math.log(idx_+1)
                hint_val[idx_]=hint_val[idx_]+1   

        i_j_100_val=generate_negative_100(user_ratings_original, test_ratings,item_count)
        g1_all=[]
        g2_all=[]
        g3_all=[]        
        for user_id in range (0,sum_coun): 
            user_sel=i_j_100_val[user_id][0]
            i_id=i_j_100_val[user_id][1]
            #print user_sel,i_id
            molecular_e_aj=[0.0,0.0,0.0,0.0,0.0,0,0,0,0,0,0,0,0,0,0]
            denominator_e_aj=0.0 
              
                
            user_sel_list=[] 
            user_sel_feature=[] 
            user_sel_feature1=[]
            for i_usersel in range(0,len(user_ups[user_sel])):
                user_sel_list.append(user_sel)
                #user_sel_feature=np.concatenate((user_sel_feature,img_feature[i_usersel]),0)
                user_sel_feature1.append(vision_repre[user_sel])
            #temp=np.take(user_emb_p,user_sel_list,0)
            #print temp
            #pdb.set_trace()
            vision_user_up=np.matmul(user_sel_feature1,e_aj_vision_w_1)
            #vision_user_up=np.matmul(user_sel_feature1,e_aj_vision_w_1) 
            x_up2=np.concatenate([np.take(user_emb_p_1,user_sel_list,0),np.take(user_emb_q_1,user_sel_list,0),\
                np.take(item_emb_x_1,user_ups[user_sel],0),np.take(item_emb_w_1,user_ups[user_sel],0),vision_user_up],1)
            e_aj_temp2=(numpy.dot(x_up2,e_aj_w_1)+e_aj_b_1)
            e_aj2=np.exp(np.sum(e_aj_temp2,1))+0.001 
            alpha= np.matmul((e_aj2).T,(np.take(item_emb_x_1,user_ups[user_sel],0)))/np.sum(e_aj2)
          
            user_sel_list=[]
            user_sel_feature_a=[]
            user_sel_feature_b=[]
            for i_usersel in range(0,len(user_follows[user_sel])):
                user_sel_list.append(user_sel)
                user_sel_feature_a.append(vision_repre[user_sel])
                user_sel_feature_b.append(vision_repre[i_usersel])
            
            vision_user_social_a=np.matmul(user_sel_feature_a,e_ab_vision_w_1)
            vision_user_social_b=np.matmul(user_sel_feature_b,e_ab_vision_w_1)
            #u_vision_f_all
            x_follow2=np.concatenate([np.take(user_emb_p_1,user_sel_list,0),np.take(user_emb_p_1,user_follows[user_sel],0),\
                np.take(user_emb_q_1,user_sel_list,0),np.take(user_emb_q_1,user_follows[user_sel],0),vision_user_social_a,vision_user_social_b],1)
            e_ab_temp2=(numpy.dot(x_follow2,e_ab_w_1)+e_ab_b_1) 
            e_ab2=np.exp(np.sum(e_ab_temp2,1))+0.001   
            beta= np.matmul((e_ab2).T,(np.take(user_emb_q_1,user_follows[user_sel],0)))/np.sum(e_ab2)  
             

            u_emb_base=user_emb_p_1[user_sel]
            u_emb_external=user_emb_q_1[user_sel]
            
            vision_user_ql=np.matmul(vision_repre[user_sel],h_f_vision_w_1)
            uploader_influence_i=user_emb_q_1[ups_user[i_id]] 
            e_a_1=np.exp(np.sum((numpy.dot(alpha,h_f_w_1)+h_f_b_1)))+0.01   
            e_a_2=np.exp(np.sum((numpy.dot(beta,h_f_w_1)+h_f_b_1)))+0.01  #tf.exp(np.sum(tf.nn.relu(tf.matmul(f2,h_f_w)+h_f_b)))  
            e_a_3=np.exp(np.sum((numpy.dot(uploader_influence_i,h_f_w_1)+h_f_b_1)))+0.01  #tf.exp(np.sum(tf.nn.relu(tf.matmul(f3,h_f_w)+h_f_b)))   
            denominator_e_ai=e_a_1+e_a_2+e_a_3
            gamma_a1= e_a_1/denominator_e_ai
            gamma_a2= e_a_2/denominator_e_ai
            gamma_a3= e_a_3/denominator_e_ai
            
            g1_all.append(gamma_a1)
            g2_all.append(gamma_a2)
            g3_all.append(gamma_a3)
            
            user_i_temp=(u_emb_base+gamma_a1*alpha+gamma_a2*beta+gamma_a3*uploader_influence_i)
            #u_emb_base+0.5*alpha+0.5*beta#
            user_i=item_b_1[i_id]+np.dot(user_i_temp,(item_emb_w_1[i_id]).T)  

            idx_=1  
            negative100=i_j_100_val[user_id][2]
            #time2=time.time() 
            for sel_100 in range(0,100):   
                j_ng = negative100[sel_100]    
                uploader_influence_i=user_emb_q_1[ups_user[j_ng]]   
                e_a_1=np.exp(np.sum((numpy.dot(alpha,h_f_w_1)+h_f_b_1)))+0.01   
                e_a_2=np.exp(np.sum((numpy.dot(beta,h_f_w_1)+h_f_b_1)))+0.01  #tf.exp(np.sum(tf.nn.relu(tf.matmul(f2,h_f_w)+h_f_b)))  
                e_a_3=np.exp(np.sum((numpy.dot(uploader_influence_i,h_f_w_1)+h_f_b_1)))+0.01  #tf.exp(np.sum(tf.nn.relu(tf.matmul(f3,h_f_w)+h_f_b)))    
                denominator_e_ai=e_a_1+e_a_2+e_a_3
                gamma_a1= e_a_1/denominator_e_ai
                gamma_a2= e_a_2/denominator_e_ai
                gamma_a3= e_a_3/denominator_e_ai
                user_j_temp=(u_emb_base+gamma_a1*alpha+gamma_a2*beta+gamma_a3*uploader_influence_i)
                #u_emb_base+0.5*alpha+0.5*beta#
                user_j=item_b_1[j_ng]+np.dot(user_j_temp,(item_emb_w_1[j_ng]).T)  
                #user_j=item_b[j_ng]+user_item_[user_sel,j_ng]+numpy.dot(numpy.transpose(user_img_w[user_sel]),img_feature[j_ng])+img_feature_b[j_ng]#(,(numpy.dot(image_features[j_ng],img_emb_w)))
                if user_j>user_i: 
                    idx_=idx_+1  
            if idx_<=50: 
                NDCG_test[idx_]=NDCG_test[idx_]+(math.log(2))/math.log(idx_+1)
                hint_test[idx_]=hint_test[idx_]+1
        print 'g1:',np.average(g1_all),np.std(g1_all),
        print 'g2:',np.average(g2_all),np.std(g2_all), 
        print 'g3:',np.average(g3_all),np.std(g3_all) 

        save_id=[1,2,3,4,5,6,7,8,9,10,15,20,25,30]  
        Path_vbpr_val_wule='./results_my_elu_exp_0.0001r/mymodel_val_top30.txt'
        wfile_vbpr_val_wule=open(Path_vbpr_val_wule,'a')

        Path_vbpr_val='./results_my_elu_exp_0.0001r/mymodel_val_top50.txt'
        wfile_vbpr_val=open(Path_vbpr_val,'a')

        wfile_vbpr_train.write('Validation,\tHIT: ')
        wfile_vbpr_val.write('epoch:'+str(epoch)+' val Hit_ratio:\n')  
        wfile_vbpr_val_wule.write('epoch:'+str(epoch)+' val Hit_ratio:\n')  
        temp_hint=0
        for d_i in range(1,51):
            temp_hint=temp_hint+hint_val[d_i] 
            mean_temp_hint=round(temp_hint*1.0/sum_coun,4)
            wfile_vbpr_val.write('top'+str(d_i)+':'+str(temp_hint)+' '+ str(mean_temp_hint)+'\n') 
            if d_i==5:
                wfile_vbpr_train.write('top5:'+str(mean_temp_hint)+'\t') 
            if d_i==10:
                wfile_vbpr_train.write('top10:'+str(mean_temp_hint)+'\t')  
            if d_i in save_id:
                wfile_vbpr_val_wule.write('top'+str(d_i)+':'+str(mean_temp_hint)+' ')
        wfile_vbpr_val_wule.write('\n')

        wfile_vbpr_train.write('NDCG: ') 
        wfile_vbpr_val.write('epoch:'+str(epoch)+' val NDCG:\n')
        wfile_vbpr_val_wule.write('epoch:'+str(epoch)+' val NDCG:\n')
        temp_ndcg=0
        for d_i in range(1,51):
            temp_ndcg=temp_ndcg+NDCG_val[d_i] 
            mean_temp_ndcg=round(temp_ndcg/sum_coun,4)
            wfile_vbpr_val.write('top'+str(d_i)+':'+str(temp_ndcg)+' '+ str(mean_temp_ndcg)+'\n') 
            if d_i==5:
                wfile_vbpr_train.write('top5:'+str(mean_temp_ndcg)+'\t') 
            if d_i==10:
                wfile_vbpr_train.write('top10:'+str(mean_temp_ndcg)+';\t') 
            if d_i in save_id:
                wfile_vbpr_val_wule.write('top'+str(d_i)+':'+str(mean_temp_ndcg)+' ')
        wfile_vbpr_val_wule.write('\n') 
        
        wfile_vbpr_train.write('\n')
        wfile_vbpr_val.write('\n')
        wfile_vbpr_val_wule.write('\n') 
        wfile_vbpr_val.close()
        wfile_vbpr_val_wule.close() 


        Path_vbpr_test_wule='./results_my_elu_exp_0.0001r/mymodel_test_top30.txt'
        wfile_vbpr_test_wule=open(Path_vbpr_test_wule,'a')

        Path_vbpr_test='./results_my_elu_exp_0.0001r/mymodel_test_top50.txt'
        wfile_vbpr_test=open(Path_vbpr_test,'a')   

        wfile_vbpr_train.write('TEST,\t\tHIT: ') 
        wfile_vbpr_test.write('epoch:'+str(epoch)+'test Hit_ratio:\n') 
        wfile_vbpr_test_wule.write('epoch:'+str(epoch)+'test Hit_ratio:\n')
        temp_hint=0
        for d_i in range(1,51):
            temp_hint=temp_hint+hint_test[d_i] 
            mean_temp_hint=round(temp_hint*1.0/sum_coun,4)
            wfile_vbpr_test.write('top'+str(d_i)+':'+str(temp_hint)+' '+ str(mean_temp_hint)+'\n')
            if d_i==5:
                wfile_vbpr_train.write('top5:'+str(mean_temp_hint)+'\t') 
            if d_i==10:
                wfile_vbpr_train.write('top10:'+str(mean_temp_hint)+'\t') 
            if d_i in save_id:
                wfile_vbpr_test_wule.write('top'+str(d_i)+':'+str(mean_temp_hint)+' ')
        wfile_vbpr_test_wule.write('\n') 

        wfile_vbpr_train.write('NDCG: ') 
        wfile_vbpr_test.write('epoch:'+str(epoch)+' test NDCG:\n')
        wfile_vbpr_test_wule.write('epoch:'+str(epoch)+' test NDCG:\n')
        temp_ndcg=0
        for d_i in range(1,51):
            temp_ndcg=temp_ndcg+NDCG_test[d_i] 
            mean_temp_ndcg=round(temp_ndcg/sum_coun,4)
            wfile_vbpr_test.write('top'+str(d_i)+':'+str(temp_ndcg)+' '+ str(mean_temp_ndcg)+'\n') 
            if d_i==5:
                wfile_vbpr_train.write('top5:'+str(mean_temp_ndcg)+'\t') 
            if d_i==10:
                wfile_vbpr_train.write('top10:'+str(mean_temp_ndcg)+'\n')  
            if d_i in save_id:
                wfile_vbpr_test_wule.write('top'+str(d_i)+':'+str(mean_temp_ndcg)+' ')
        wfile_vbpr_test_wule.write('\n') 
       
        wfile_vbpr_test.write('\n')
        wfile_vbpr_test_wule.write('\n')

        wfile_vbpr_test.close()
        wfile_vbpr_test_wule.close() 
        wfile_vbpr_train.close()
        






