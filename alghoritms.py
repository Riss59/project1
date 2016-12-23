import numpy as np
import pandas as pd
import time
import cProfile
import pstats
from sklearn import preprocessing
from sklearn.datasets import load_iris

def odl(x,newdata_ii): 
    return(np.sqrt(sum(x.subtract(newdata_ii).apply(float)**2)));

def get_kneighbours(decision_table,newdata_ii,k,method_type):
    array_d=[]
    for i in range(len(decision_table.iloc[:,0])):
           array_d.append(odl(decision_table.iloc[i,:-1],newdata_ii))
    nearest_dt=pd.concat([decision_table,pd.DataFrame(array_d)],axis=1)
    nearest_dt.dropna()
    return(nearest_dt)

def calc_membership(k_averDist,sum_denominator,nearest_dt,num_class,num_Class,control,type_="gradual"):
    #numClass[1, i] <- nrow(nearest.dt[which(nearest.dt[,(ncol(nearest.dt) - 1)] == i), ,drop = FALSE])
    ## calculate membership of each class based on Keller et. al.'s technique.
    miu_class=[]
    m=control["m"]
    type_=control["type_membership"]
    #res=pd.concat([pd.DataFrame([1,5,2]),pd.DataFrame([1,2,3])],axis=0)
    #miu_class=miu_class.append(pd.DataFrame([1,5,2]).T)
    #pd.DataFrame([1,5,2])
    #miu_class.loc[len(miu_class)]=[1,2,3]
    
    array_tau=[]
    for j in range(num_class):
        array_miu=[]
        for i in range(len(nearest_dt)):
            if(nearest_dt[i,(len(nearest_dt[0])-1-1)]==j):
                if(type_=="gradual"):
                    w=(0.51+(num_Class[int(nearest_dt[i,len(nearest_dt[0])-1-1])]/float(len(nearest_dt))*0.49))
                elif(type_=="crisp"):
                    w=1
        
            else:
                if(type_=="gradual"):
                    w=num_Class[j]/float(len(nearest_dt))*0.49
                elif(type_=="crisp"):
                    w=0
            array_miu.append(w*1/np.power(nearest_dt[i,len(nearest_dt[0])-1],(2/(m-1)))/sum_denominator)
        miu1=sum(array_miu)
        temp1=sum(miu1/(1+k_averDist*nearest_dt[:,-1]**(2/(m-1))))
        array_tau.append((1/float(len(nearest_dt)))*temp1)
    miu_class=array_tau
        #suma miu class musi być równa 1
    return(miu_class)        


# Funkcja Pomiar podobieństwa
# @param nearest_dt matrix of data nearest neighobur
# @control a dictionary of parameters 

def calc_similiarity_degree(k_averDist,nearest_dt,control,num_Class):
    num_class=control["num_class"]
    m=control["m"]
    sum_denominator=sum(1/nearest_dt[:,len(nearest_dt[1])-1]**(2/(m-1)))
    miu_clas=calc_membership(k_averDist=k_averDist,sum_denominator=sum_denominator,nearest_dt=nearest_dt,num_class=num_class,num_Class=num_Class,control=control,type_="gradual")
    return (miu_clas)



def C_FRNN_O_FRST(decision_table,newdata,control):
    m=control["m"]
    type=control["type_membership"]
    num_class=control["num_class"]
    normalize=control["normalize"]
    if(normalize==False):
        object=decision_table[:,:-1]
        decision_scale_nom=preprocessing.MinMaxScaler().fit(object)
        decsion_table_norm=decision_scale_nom.transform(object)
        newdata=decision_scale_nom.transform(newdata)
        result_decision_table=decision_table[:,len(decision_table[0])-1]
        dec_table=np.c_[decsion_table_norm,result_decision_table]
    else:
        dec_table=decision_table
    num_inputvar=float(len(dec_table[0])) - 1
    num_instances=float(len(dec_table))
    array_w=[]
    for i in range(num_class):
        array_w.append(sum(dec_table[:,len(dec_table[0])-1]==i)) 
    num_Class=array_w
    res_class=[]
    for i in range(len(newdata)):
        distance=[]
        for j in range(dec_table.shape[0]):
            distance.append(np.power(np.sum(np.power(dec_table[j,:-1]-newdata[i,:],2)),0.5))
        k_averDist=1/((1/num_instances)*np.sum(np.power(distance,2/(m-1))))
        nearest_dt=np.c_[dec_table,distance]
        nearest_dt[nearest_dt[:,len(nearest_dt[1])-1]==0]=0.00001
        miu=calc_similiarity_degree(k_averDist,nearest_dt,control,num_Class)
        res_class.append(np.argmax(miu))
    return_class=pd.DataFrame(res_class)
    return(return_class)



def FKNN(decision_table,newdata,control):
    k=control["k"]
    m=control["m"]
    num_class=control["num_class"]
    normalize=control["normalize"]
    if(normalize==False):
        object=decision_table[:,:-1]
        decision_scale_nom=preprocessing.MinMaxScaler().fit(object)
        decsion_table_norm=decision_scale_nom.transform(object)
        newdata=decision_scale_nom.transform(newdata)
        result_decision_table=decision_table[:,len(decision_table[0])-1]
        dec_table=np.c_[decsion_table_norm,result_decision_table]
    else:
        dec_table=decision_table
    array_predicted=[]
    temp=np.zeros((len(dec_table),num_class))
    for i in range(len(dec_table)):
        temp[i,:]=np.array([0]*dec_table[i,-1]+[1]+[0]*(num_class-dec_table[i,-1]-1))
    labels=temp
    for i in range(len(newdata)):
        distance=np.sqrt(np.sum((dec_table[:,:-1]  - newdata[i])**2,axis=1))
        idx=np.argpartition(distance,k)[:k]
        idx=np.sort(idx)
        distance=np.array(distance)
        #nearest_k=np.c_[dec_table[idx],np.array(distance)[idx]]
        distance[distance[:]==0]=0.00001
        weight=distance[idx]**(-2/(m-1))                            
        test_out=np.dot(weight,labels[idx,:])/sum(weight)
        #number_of_element_in_class=[]
        #for c in range(num_class):
         #   number_of_element_in_class.append(sum(nearest_k[:,len(nearest_k[0])-1-1]==c))
        #sum_denominator=sum(1/nearest_k[:,len(nearest_k[0])-1]**(2/(m-1)))
        
        #u=[]
        #for c in range(num_class):
            #u_ii=[]
            #for n in range(len(nearest_k)):
                #if(nearest_k[n,len(nearest_k[0])-1-1]==c):
                    #u_i=0.51+ number_of_element_in_class[c]/float(k)*0.49
               # else:
                  #  u_i=number_of_element_in_class[c]/float(k)*0.49
                #u_ii.append(u_i*1/nearest_k[n,len(nearest_k[0])-1]**(2/(m-1)))        
           # u.append(sum(u_ii)/sum_denominator)
        
        array_predicted.append(np.argmax(test_out))
    return(array_predicted)                

import time
import cProfile
import pstats
