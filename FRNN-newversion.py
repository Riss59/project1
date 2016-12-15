
#!/usr/bin/env python

import numpy as np
import pandas as pd

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
    # calculate membership of newdata to each class
    ## miu.class is a matrix (k, num.class) where k is number of neighbor
    sum_denominator=sum(1/nearest_dt[:,len(nearest_dt[1])-1]**(2/(m-1)))
    miu_clas=calc_membership(k_averDist=k_averDist,sum_denominator=sum_denominator,nearest_dt=nearest_dt,num_class=num_class,num_Class=num_Class,control=control,type_="gradual")
    ## calculate sum on denominator of the similarity eq.
    ## calculate membership function of class for determining class of newdata
    return (miu_clas)



def C_FRNN_O_FRST(decision_table,newdata,control):
    m=control["m"]
    type=control["type_membership"]
    num_class=control["num_class"]
    object=decision_table[:,:-1]
    decision_scale_nom=preprocessing.MinMaxScaler().fit(object)
    decsion_table_norm=decision_scale_nom.transform(object)
    newdata=decision_scale_nom.transform(newdata)
    result_decision_table=decision_table[:,len(decision_table[0])-1]
    dec_table=np.c_[decsion_table_norm,result_decision_table]
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
        ## calculate and get K-nearest neighbor (in this case, K = num.instances)
        nearest_dt=np.c_[dec_table,distance]
        nearest_dt[nearest_dt[:,len(nearest_dt[1])-1]==0]=0.00001
        ## calculate membership of each class based on nearest.dt (in this case: all training data)
        miu=calc_similiarity_degree(k_averDist,nearest_dt,control,num_Class)
        ## calculate fuzzy-rough ownership function
        
        res_class.append(np.argmax(miu))
    return_class=pd.DataFrame(res_class)
    return(return_class)

import time
import cProfile
import pstats
print("Init")
tic=time.clock()
data = load_iris()
data_x=pd.DataFrame(data.data)
wynik_x=pd.DataFrame(data.target)
data=pd.concat([data_x,wynik_x],axis=1)
decision_table_iris=data.sample(frac=0.7)
newdata_iris=data.loc[~data.index.isin(decision_table_iris.index)]
wyniki_newdata=newdata_iris.iloc[:,-1]
wyniki_newdata.index=range(45)
newdata_iris=newdata_iris.iloc[:,:-1]
control={"num_class":3,"m":3,"type_membership":"gradual"}
toc=time.clock()
print(toc-tic)

tic=time.clock()
wynik2=C_FRNN_O_FRST(decision_table_iris,newdata_iris,control)
toc=time.clock()
print("FRNN",toc-tic)
wynik2
porownanie=pd.concat([pd.DataFrame(wynikitestow1),pd.DataFrame(wyniki_newdata)],axis=1)

decision_table=decision_table_iris
newdata=newdata_iris
decision_table=np.array(decision_table)
newdata=np.array(newdata_iris)
new
