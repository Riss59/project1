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

def calc_membership(nearest_dt,num_class,type_="gradual"):
    num_Class=pd.DataFrame
    array_w=[]
    for i in range(num_class):
        array_w.append(nearest_dt[nearest_dt.iloc[:,(nearest_dt.shape[1]-1-1)]==i].shape[0]) 
    num_Class=pd.DataFrame(array_w).T
    
    #numClass[1, i] <- nrow(nearest.dt[which(nearest.dt[,(ncol(nearest.dt) - 1)] == i), ,drop = FALSE])
    ## calculate membership of each class based on Keller et. al.'s technique.
    miu_class=pd.DataFrame()
    #res=pd.concat([pd.DataFrame([1,5,2]),pd.DataFrame([1,2,3])],axis=0)
    #miu_class=miu_class.append(pd.DataFrame([1,5,2]).T)
    #pd.DataFrame([1,5,2])
    #miu_class.loc[len(miu_class)]=[1,2,3]
    for i in range(nearest_dt.shape[0]):
        array_miu_class=[]
        for j in range(num_class):
            if(nearest_dt.iloc[i,(nearest_dt.shape[1]-1-1)]==j):
                if(type_=="gradual"):
                    array_miu_class.append(0.51+(num_Class.iloc[0,int(nearest_dt.iloc[i,nearest_dt.shape[1]-1-1])]/float(nearest_dt.shape[0])*0.49))
                elif(type_=="crisp"):
                    array_miu_class.append(1)
        
            else:
                if(type_=="gradual"):
                    array_miu_class.append(num_Class.iloc[0,j]/float(nearest_dt.shape[0])*0.49)
                elif(type_=="crisp"):
                    array_miu_class.append(0)
        miu_class=miu_class.append(pd.DataFrame(array_miu_class).T)
        #suma miu class musi być równa 1
    return(miu_class)        


# Funkcja Pomiar podobieństwa
# @param nearest_dt matrix of data nearest neighobur
# @control a dictionary of parameters 
def calc_similiarity_degree(nearest_dt,control):
    num_clas=control["num_class"]
    m=control["m"]
    # calculate membership of newdata to each class
    ## miu.class is a matrix (k, num.class) where k is number of neighbor
    miu_clas=calc_membership(nearest_dt=nearest_dt,num_class=num_class,type_="gradual")
    ## calculate sum on denominator of the similarity eq.
    sum_denominator=sum(1/nearest_dt.iloc[:,nearest_dt.shape[1]-1]**(2/(m-1)))
    ## calculate membership function of class for determining class of newdata
    miu=np.matmul(miu_clas.T,1/np.power(nearest_dt.iloc[:,nearest_dt.shape[1]-1],(2/(m-1)))/sum_denominator)
    return (miu)




def C_FRNN_O_FRST(decision_table,newdata,control):
    m=control["m"]
    type=control["type_membership"]
    num_class=control["num_class"]
    object=decision_table.iloc[:,:-1]
    decision_scale_nom=preprocessing.MinMaxScaler().fit(object)
    decsion_table_norm=pd.DataFrame(decision_scale_nom.transform(object))
    newdata=pd.DataFrame(decision_scale_nom.transform(newdata))
    result_decision_table=pd.DataFrame(decision_table.iloc[:,decision_table.shape[1]-1])
    result_decision_table.index=range(result_decision_table.shape[0])
    dec_table=pd.concat([decsion_table_norm,result_decision_table],axis=1)
    num_inputvar=float(dec_table.shape[1]) - 1
    num_instances=float(dec_table.shape[0])
    res_class=[]
    for i in range(newdata.shape[0]):
        num_instances=float(num_instances)
        distance=[]
        for j in range(dec_table.shape[0]):
            distance.append(np.power(np.sum(np.power(dec_table.iloc[j,:-1]-newdata.iloc[i,:],2)),0.5))
        k_averDist=1/((1/num_instances)*np.sum(np.power(distance,2/(m-1))))
        ## calculate and get K-nearest neighbor (in this case, K = num.instances)
        nearest_dt=get_kneighbours(dec_table,newdata.iloc[i,:],num_instances,"cos")
        nearest_dt[nearest_dt.iloc[:,nearest_dt.shape[1]-1]==0]=0.00001
        ## calculate membership of each class based on nearest.dt (in this case: all training data)
        miu=calc_similiarity_degree(nearest_dt,control)
        ## calculate fuzzy-rough ownership function
        array_tau=[]
        for c in range(num_class):
            temp=0
            for jj in range(int(num_instances)):
                distance_i=np.power(np.sum(np.power(dec_table.iloc[jj,:-1]-newdata.iloc[i,:],2)),0.5)
                temp=temp+miu[c]/(1+k_averDist*np.power(distance_i,(2/(m-1))))
            array_tau.append((1/float(num_instances))*temp)
        res_class.append(np.argmax(array_tau))
    return_class=pd.DataFrame(res_class)
    return(return_class)



data = load_iris()
data_x=pd.DataFrame(data.data)
wynik_x=pd.DataFrame(data.target)
data=pd.concat([data_x,wynik_x],axis=1)
decision_table_iris=data.iloc[range(110),:]
newdata_iris=data.iloc[range(110,150),:-1]
wyniki_newdata=data.iloc[range(110,150),4]
wyniki_newdata.index=range(40)
control={"num_class":3,"m":3,"type_membership":"cos"}
wynikitestow=C_FRNN_O_FRST(decision_table_iris,newdata_iris,control)
porownanie=pd.concat([pd.DataFrame(wynikitestow),pd.DataFrame(wyniki_newdata)],axis=1)




