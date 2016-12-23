import pandas as pd 
import numpy as np 
import sys
import string as s
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold,StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import plotly
from scipy.spatial.distance import mahalanobis
import scipy as sp
import statsmodels

filepath="F:\data_281016_v2.csv"
n_splits=4
dane=getstratisfiedSample(filepath)
dane=sample
threshold_PCA=90
n=np.array(dane.iloc[:,0])
m=np.array(dane.iloc[:,range(1,dane.shape[1])])
______________________________________________________________________________

Train_sets_x, Test_sets_x, Train_y, Test_y = train_test_split(m, n, test_size=0.3, stratify=n)

directory_mv="E:\Tests"
if not os.path.exists(directory_mv):
    os.makedirs(directory_mv)
os.chdir(directory_mv)
#_______________________________________________
#
# Wypełniamy srednia missin values
#
#_______________________________________________
Train_sets_x_wmv=pd.DataFrame(Train_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
Train=scale(Train_sets_x_wmv)
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)
plot_name=str("PCA"+"mean"+str(i)+".png")
plt.plot(var1)
plt.savefig(plot_name)
plt.show(block=False)
plt.savefig(plot_name)
pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)
Test_sets_x_wmv=pd.DataFrame(Test_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
Test=scale(Test_sets_x_wmv)
Test=pca.fit_transform(Test)
skf=StratifiedKFold(Train_y,n_splits,True)
#Testowanie
FKNN_result_mean=[]
for k in range(1,10):
    auc_result1=[] 
    for m in [-1,0,0.5,0.9,0.99,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            #resultsname=str("FRNN-Missing_values-"+"mean"+"_crosvalidation_m_"+str(m)+"-i-"+str(i)+".csv")
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            #confusion_matrix(y_test,wyniki1)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_result_mean.append(auc_result1)

bb=FKNN_result_mean
pd.DataFrame(bb).to_csv("FKNN-srednia.csv")
os.getcwd()
control={"k":5,"m":1.5,"num_class":2,"normalize":True}
wynikimoje=FKNN(np.c_[X,Train_y],Test,control)


#___________________________________________________________
#
# Wypełniamy srednia missing values oraz usuwamy kolumny
#
#___________________________________________________________
Train_sets_x, Test_sets_x, Train_y, Test_y = train_test_split(m, n, test_size=0.3, stratify=n)

wyniki=column_reject(Train_sets_x)
Deletedcolumn_Train_sets_x=wyniki["data"]
deleted_rows=wyniki["del_columns"]

Train_sets_x_wmv_cr=pd.DataFrame(Deletedcolumn_Train_sets_x).fillna(pd.DataFrame(Deletedcolumn_Train_sets_x).mean())
#         kmo=KMO(Train_sets_x_wmv_cr)
#         kmo>0.5
Train=scale(Train_sets_x_wmv_cr)
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)
plot_name=str("PCA"+"mean"+str(i)+".png")
plt.plot(var1)
plt.savefig(plot_name)
plt.show(block=False)
plt.savefig(plot_name)
pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)
Test_sets_x_wmv=pd.DataFrame(np.delete(Test_sets_x,deleted_rows,1)).fillna(pd.DataFrame(np.delete(Test_sets_x,deleted_rows,1)).mean())
Test=scale(Test_sets_x_wmv)
Test_sets_x_wmv=pca.fit_transform(Test)
skf=StratifiedKFold(Train_y,n_splits,True)
#Testowanie
FKNN_result_mean_cr=[]
for k in range(3,10):
    auc_result1=[] 
    print(k)
    for m in [-1,0,0.5,0.9,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = Train[train_index], Train[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            #resultsname=str("FRNN-Missing_values-"+"mean"+"_crosvalidation_m_"+str(m)+"-i-"+str(i)+".csv")
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            #confusion_matrix(y_test,wyniki1)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_result_mean_cr.append(auc_result1)

bb=FKNN_result_mean_cr
pd.DataFrame(bb).to_csv("FKNN-srednia-deleted_columns.csv")

#_____________________________________________
#
# Wypełniamy mediana missing values
# 
#________________________________________________

Train_sets_x_wmmv=pd.DataFrame(Train_sets_x).fillna(pd.DataFrame(Train_sets_x).median())
Train=scale(Train_sets_x_wmmv)
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)
plot_name=str("PCA"+"mean"+str(i)+".png")
plt.plot(var1)
plt.savefig(plot_name)
plt.show(block=False)
plt.savefig(plot_name)
pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)
Test_sets_x_wmv=pd.DataFrame(Test_sets_x).fillna(pd.DataFrame(Train_sets_x).median())
Test=scale(Test_sets_x_wmv)
Test=pca.fit_transform(Test)
skf=StratifiedKFold(Train_y,n_splits,True)
#Testowanie
FKNN_result_median=[]
for k in range(1,10):
    auc_result1=[] 
    for m in [-1,0,0.5,0.9,0.99,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = Train[train_index], Train[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            #resultsname=str("FRNN-Missing_values-"+"mean"+"_crosvalidation_m_"+str(m)+"-i-"+str(i)+".csv")
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            #confusion_matrix(y_test,wyniki1)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_result_median.append(auc_result1)

mediana_b=FKNN_result_mean
pd.DataFrame(mediana_b).to_csv("FKNN-mediana.csv")


#_____________________________________________
#
#        Wypełniamy mediana missin values
# 
#________________+columnreject__________________________

Train_sets_x, Test_sets_x, Train_y, Test_y = train_test_split(m, n, test_size=0.3, stratify=n)

wyniki=column_reject(Train_sets_x)
Deletedcolumn_Train_sets_x=wyniki["data"]
deleted_rows=wyniki["del_columns"]

Train_sets_x_wmmv_cr=pd.DataFrame(Deletedcolumn_Train_sets_x).fillna(pd.DataFrame(Deletedcolumn_Train_sets_x).median())
#         kmo=KMO(Train_sets_x_wmv_cr)
#         kmo>0.5
Train=scale(Train_sets_x_wmmv_cr)
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)
plot_name=str("PCA"+"mean"+str(i)+".png")
plt.plot(var1)
plt.savefig(plot_name)
plt.show(block=False)
plt.savefig(plot_name)
pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)
Test_sets_x_wmv=pd.DataFrame(np.delete(Test_sets_x,deleted_rows,1)).fillna(pd.DataFrame(np.delete(Test_sets_x,deleted_rows,1)).median())
Test=scale(Test_sets_x_wmv)
Test_sets_x_wmv=pca.fit_transform(Test)
skf=StratifiedKFold(Train_y,n_splits,True)
#Testowanie
FKNN_result_median_cr=[]
for k in range(3,10):
    auc_result1=[] 
    print(k)
    for m in [-1,0,0.5,0.9,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = Train[train_index], Train[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            #resultsname=str("FRNN-Missing_values-"+"mean"+"_crosvalidation_m_"+str(m)+"-i-"+str(i)+".csv")
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            #confusion_matrix(y_test,wyniki1)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_result_median_cr.append(auc_result1)

bb=FKNN_result_median_cr
pd.DataFrame(bb).to_csv("FKNN-mediana-deleted_columns.csv")


#_____________________________________________
#
#        Wypełniamy srednia missin values
# 
#_____________+univariate test_________

Train_sets_x, Test_sets_x, Train_y, Test_y = train_test_split(m, n, test_size=0.3, stratify=n)

Train_sets_x_mean_univariate=pd.DataFrame(Train_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
Train_sets_x_wmmv=np.array(Train_sets_x_mean_univariate)
columns_univariate_outlier=None
for i in range(len(Train_sets_x_wmmv[0])):
    if(i==0):
        columns_univariate_outlier=reject_outliers(Train_sets_x_wmmv[:,i])
    else:
        Out=reject_outliers(Train_sets_x_wmmv[:,i])
        columns_univariate_outlier=np.c_[columns_univariate_outlier,Out]
#manahanobi
#numpycc_df=Train_sets_x_wmmv
columns_univariate_outlier=pd.DataFrame(columns_univariate_outlier)
columns_univariate_outlier)=columns_univariate_outlier.fillna(pd.DataFrame(Train_sets_x).mean())
#Cov_Sx = numpycc_df.cov() 
#IC = sp.linalg.inv(Cov_Sx + np.eye(Cov_Sx.shape[0])*10**(-6))
#meanCol = numpycc_df.mean().values
#results=mahalanobisR(numpycc_df,meanCol,IC)
#Train_sets_x_wmmv_ou=results["data"]
#rows=results["rows"]
#Train=np.array(Train_sets_x_wmmv_ou)

Train=scale(np.array(columns_univariate_outlier))
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)
#plot_name=str("PCA"+"mean"+str(i)+".png")
#plt.plot(var1)
#plt.savefig(plot_name)
#plt.show(block=False)
#plt.savefig(plot_name)
pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)
Test_sets_x_wmv=pd.DataFrame(Test_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
Test=scale(Test_sets_x_wmv)
Test=pca.fit_transform(Test)

#Train_y=Train_y[rows]
skf=StratifiedKFold(Train_y,n_splits,True)   
# Testowanie 
FKNN_results_mean_univ=[]
for k in range(3,10):
    print("k:",k)
    auc_result1=[] 
    for m in [-1,0,0.5,0.9,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = Train[train_index], Train[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            #resultsname=str("FRNN-Missing_values-"+"mean"+"_crosvalidation_m_"+str(m)+"-i-"+str(i)+".csv")
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            #confusion_matrix(y_test,wyniki1)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_results_mean_univ.append(auc_result1)
result_outliers_csv__mean_univ=FKNN_results_mean_univ
pd.DataFrame(result_outliers_csv__mean_univ).to_csv("E:\PKO\Missing_values\FKNN_mean_univ.csv")

#_____________________________________________
#
#        Wypełniamy mediana missin values
# 
#_____________+univariate test_________

Train_sets_x, Test_sets_x, Train_y, Test_y = train_test_split(m, n, test_size=0.3, stratify=n)

Train_sets_x_median_univariate=pd.DataFrame(Train_sets_x).fillna(pd.DataFrame(Train_sets_x).median())
Train_sets_x_wmmv=np.array(Train_sets_x_median_univariate)
columns_univariate_outlier=None
for i in range(len(Train_sets_x_wmmv[0])):
    if(i==0):
        columns_univariate_outlier=reject_outliers(Train_sets_x_wmmv[:,i])
    else:
        Out=reject_outliers(Train_sets_x_wmmv[:,i])
        columns_univariate_outlier=np.c_[columns_univariate_outlier,Out]

columns_univariate_outlier=pd.DataFrame(columns_univariate_outlier)
columns_univariate_outlier=columns_univariate_outlier.fillna(pd.DataFrame(Train_sets_x).median())
Train=scale(np.array(columns_univariate_outlier))
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)

pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)
Test_sets_x_wmv=pd.DataFrame(Test_sets_x).fillna(pd.DataFrame(Train_sets_x).median())
Test=scale(Test_sets_x_wmv)
Test=pca.fit_transform(Test)

skf=StratifiedKFold(Train_y,n_splits,True)   
# Testowanie 
FKNN_results_median_univ=[]
for k in range(3,10):
    print("k:",k)
    auc_result1=[] 
    for m in [-1,0,0.5,0.9,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = Train[train_index], Train[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            #resultsname=str("FRNN-Missing_values-"+"mean"+"_crosvalidation_m_"+str(m)+"-i-"+str(i)+".csv")
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            #confusion_matrix(y_test,wyniki1)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_results_median_univ.append(auc_result1)
result_outliers_csv__median_univ=FKNN_results_median_univ
pd.DataFrame(result_outliers_csv__median_univ).to_csv("FKNN-_median_univ.csv")

#_____________________________________________
#
#        Wypełniamy srednia missin values
# 
#_____________+multivariate test_________

Train_sets_x, Test_sets_x, Train_y, Test_y = train_test_split(m, n, test_size=0.3, stratify=n)

Train_sets_x_wmmv=pd.DataFrame(Train_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
#Train_sets_x_wmmv=np.array(Train_sets_x_wmmv)

Cov_Sx = Train_sets_x_wmmv.cov() 
IC = sp.linalg.inv(Cov_Sx + np.eye(Cov_Sx.shape[0])*10**(-6))
meanCol = Train_sets_x_wmmv.mean().values
results=mahalanobisR(Train_sets_x_wmmv,meanCol,IC)
Train_sets_x_wmmv_ou=results["data"]
rows=results["rows"]
Train_y=Train_y[rows]

Train=np.array(Train_sets_x_wmmv_ou)
Train=scale(Train)
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)
pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)

Test_sets_x_wmv=pd.DataFrame(Test_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
Test=scale(Test_sets_x_wmv)
Test=pca.fit_transform(Test)

skf=StratifiedKFold(Train_y,n_splits,True)   
# Testowanie 
FKNN_results_mean_mahanalobis=[]
for k in range(3,10):
    print("k:",k)
    auc_result1=[] 
    for m in [-1,0,0.5,0.9,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = Train[train_index], Train[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            #resultsname=str("FRNN-Missing_values-"+"mean"+"_crosvalidation_m_"+str(m)+"-i-"+str(i)+".csv")
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            #confusion_matrix(y_test,wyniki1)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_results_mean_mahanalobis.append(auc_result1)
result_outliers_csv_mean_multivariate=FKNN_results_mean_mahanalobis
pd.DataFrame(result_outliers_csv_mean_multivariate).to_csv("FKNN-mean-mahanalobis.csv")

#_____________________________________________
#
#        Wypełniamy mediana missin values
# 
#_____________+multivariate test_________

Train_sets_x, Test_sets_x, Train_y, Test_y = train_test_split(m, n, test_size=0.3, stratify=n)

Train_sets_x_wmmv=pd.DataFrame(Train_sets_x).fillna(pd.DataFrame(Train_sets_x).median())
Train_sets_x_wmmv=np.array(Train_sets_x_wmmv)
#numpycc=None
#for i in range(596):
#    if(i==0):
#        numpycc=reject_outliers(Train_sets_x_wmmv[:,i])
#    else:
##        Out=reject_outliers(Train_sets_x_wmmv[:,i])
 #       numpycc=np.c_[numpycc,Out]
#manahanobi
numpycc_df=Train_sets_x_wmmv
numpycc_df=pd.DataFrame(numpycc_df)
Cov_Sx = numpycc_df.cov() 
IC = sp.linalg.inv(Cov_Sx + np.eye(Cov_Sx.shape[0])*10**(-6))
meanCol = numpycc_df.mean().values
results=mahalanobisR(numpycc_df,meanCol,IC)
Train_sets_x_wmmv_ou=results["data"]
rows=results["rows"]
Train=np.array(Train_sets_x_wmmv_ou)

Train=scale(Train_sets_x_wmmv_ou)
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)
#plot_name=str("PCA"+"mean"+str(i)+".png")
#plt.plot(var1)
#plt.savefig(plot_name)
#plt.show(block=False)
#plt.savefig(plot_name)
pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)
Test_sets_x_wmv=pd.DataFrame(Test_sets_x).fillna(pd.DataFrame(Train_sets_x).median())
Test=scale(Test_sets_x_wmv)
Test=pca.fit_transform(Test)

Train_y=Train_y[rows]
skf=StratifiedKFold(Train_y,n_splits,True)   
# Testowanie 
FKNN_results_median=[]
for k in range(3,10):
    print("k:",k)
    auc_result1=[] 
    for m in [-1,0,0.5,0.9,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = Train[train_index], Train[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            #resultsname=str("FRNN-Missing_values-"+"mean"+"_crosvalidation_m_"+str(m)+"-i-"+str(i)+".csv")
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            #confusion_matrix(y_test,wyniki1)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_results_median.append(auc_result1)
result_outliers_csv_median=FKNN_results_median
pd.DataFrame(result_outliers_csv_median).to_csv("FKNN-median-multivariate.csv")


#_____________________________________________
#
#        Wypełniamy srednia missin values
# 
#_____________univariate+ multivariate test___

Train_sets_x, Test_sets_x, Train_y, Test_y = train_test_split(m, n, test_size=0.3, stratify=n)

Train_sets_x_wmmv=pd.DataFrame(Train_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
Train_sets_x_wmmv=np.array(Train_sets_x_wmmv)
columns_univariate_outlier=None
for i in range(len(Train_sets_x_wmmv[0])):
    if(i==0):
        columns_univariate_outlier=reject_outliers(Train_sets_x_wmmv[:,i])
    else:
        Out=reject_outliers(Train_sets_x_wmmv[:,i])
        columns_univariate_outlier=np.c_[columns_univariate_outlier,Out]
#manahanobi
columns_univariate_outlier=pd.DataFrame(columns_univariate_outlier)
Cov_Sx = columns_univariate_outlier.cov() 
IC = sp.linalg.inv(Cov_Sx + np.eye(Cov_Sx.shape[0])*10**(-6))
meanCol = columns_univariate_outlier.mean().values
results=mahalanobisR(columns_univariate_outlier,meanCol,IC)
Train_sets_x_wmmv_ou=results["data"]
rows=results["rows"]
Train_y=Train_y[rows]

Train=np.array(Train_sets_x_wmmv_ou)
Train=scale(Train_sets_x_wmmv_ou)
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)

pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)
Test_sets_x_wmv=pd.DataFrame(Test_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
Test=scale(Test_sets_x_wmv)
Test=pca.fit_transform(Test)

skf=StratifiedKFold(Train_y,n_splits,True)   
# Testowanie 
FKNN_results_mean_univar_multivar=[]
for k in range(3,10):
    print("k:",k)
    auc_result1=[] 
    for m in [-1,0,0.5,0.9,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = Train[train_index], Train[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_results_mean_univar_multivar.append(auc_result1)
result_outliers_csv_mean_univ_multiv=FKNN_results_mean_univar_multivar
pd.DataFrame(result_outliers_csv_mean_univ_multiv).to_csv("FKNN-mean_univariate-multivariate.csv")

#_____________________________________________
#
#        Wypełniamy mediana missin values
# 
#_____________univariate+ multivariate test___

Train_sets_x, Test_sets_x, Train_y, Test_y = train_test_split(m, n, test_size=0.3, stratify=n)

Train_sets_x_wmmv=pd.DataFrame(Train_sets_x).fillna(pd.DataFrame(Train_sets_x).median())
Train_sets_x_wmmv=np.array(Train_sets_x_wmmv)
columns_univariate_outlier=None
for i in range(len(Train_sets_x_wmmv[0])):
    if(i==0):
        columns_univariate_outlier=reject_outliers(Train_sets_x_wmmv[:,i])
    else:
        Out=reject_outliers(Train_sets_x_wmmv[:,i])
        columns_univariate_outlier=np.c_[columns_univariate_outlier,Out]
#manahanobi
# Czy one nie maja tutaj pustych wartosci ?????????????????????????
columns_univariate_outlier=pd.DataFrame(columns_univariate_outlier)
Cov_Sx = columns_univariate_outlier.cov() 
IC = sp.linalg.inv(Cov_Sx + np.eye(Cov_Sx.shape[0])*10**(-6))
meanCol = columns_univariate_outlier.mean().values
results=mahalanobisR(columns_univariate_outlier,meanCol,IC)
Train_sets_x_wmmv_ou=results["data"]
rows=results["rows"]
Train_y=Train_y[rows]

Train=np.array(Train_sets_x_wmmv_ou)
Train=scale(Train_sets_x_wmmv_ou)
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)

pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)
Test_sets_x_wmv=pd.DataFrame(Test_sets_x).fillna(pd.DataFrame(Train_sets_x).median())
Test=scale(Test_sets_x_wmv)
Test=pca.fit_transform(Test)

skf=StratifiedKFold(Train_y,n_splits,True)   
# Testowanie 
FKNN_results_median_univar_multivar=[]
for k in range(3,10):
    print("k:",k)
    auc_result1=[] 
    for m in [-1,0,0.5,0.9,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = Train[train_index], Train[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_results_median_univar_multivar.append(auc_result1)
result_outliers_csv_median_univ_multiv=FKNN_results_median_univar_multivar
pd.DataFrame(result_outliers_csv_median_univ_multiv).to_csv("FKNN-median_univariate-multivariate.csv")


###############################################################################
#______________________________________________________________________________
#_________________________________XGBOOST______________________________________
#______________________________________________________________________________
###############################################################################

#_______________________________________________
#
# Wypełniamy srednia missin values
#
#_______________________________________________
Train_sets_x_wmv=pd.DataFrame(Train_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
Train=scale(Train_sets_x_wmv)

Test_sets_x_wmv=pd.DataFrame(Test_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
Test=scale(Test_sets_x_wmv)
skf=StratifiedKFold(Train_y,n_splits,True)

#                               USTALAMY X
x=None
#Testowanie
FKNN_result_mean=[]
for k in range(1,10):
    auc_result1=[] 
    for m in [-1,0,0.5,0.9,0.99,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_result_mean.append(auc_result1)

bb=FKNN_result_mean
pd.DataFrame(bb).to_csv("FKNN-srednia.csv")
os.getcwd()
control={"k":5,"m":1.5,"num_class":2,"normalize":True}
wynikimoje=FKNN(np.c_[X,Train_y],Test,control)













   
#---------------------------------------
# Odstające wartosci i odrzucenie kilku kolumn
# srednia
#--------------------------------------- 

Train_sets_x, Test_sets_x, Train_y, Test_y = train_test_split(m, n, test_size=0.3, stratify=n)

Train=scale(Train_sets_x_wmv)
pca=PCA()    
pca.fit(Train)        
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(var,decimals=4)*100)
number=sum(var1<threshold_PCA)
plot_name=str("PCA"+"mean"+str(i)+".png")
plt.plot(var1)
plt.savefig(plot_name)
plt.show(block=False)
plt.savefig(plot_name)
pca=PCA(n_components=number)
pca.fit(Train)    
X=pca.fit_transform(Train)
Test_sets_x_wmv=pd.DataFrame(Test_sets_x).fillna(pd.DataFrame(Train_sets_x).mean())
Test=scale(Test_sets_x_wmv)
Test=pca.fit_transform(Test)
skf=StratifiedKFold(Train_y,n_splits,True)
#Testowanie
FKNN_result_mean=[]
for k in range(1,10):
    auc_result1=[] 
    for m in [-1,0,0.5,0.9,0.99,1.01,1.05,1.1,1.5,2,3]:
        i=0
        auc_per_fold=[]
        for train_index, test_index in skf:
            i+=1
            print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
            X_train, X_test = Train[train_index], Train[test_index]
            y_train, y_test = Train_y[train_index], Train_y[test_index]
            #resultsname=str("FRNN-Missing_values-"+"mean"+"_crosvalidation_m_"+str(m)+"-i-"+str(i)+".csv")
            Xxtrain=np.c_[X_train,y_train]
            control={"k":k,"m":m,"num_class":2,"normalize":True}
            wyniki1=FKNN(Xxtrain,X_test,control)
            #confusion_matrix(y_test,wyniki1)
            auc=metrics.roc_auc_score(y_test,wyniki1)
            i+=1
            auc_per_fold.append(auc)
        auc_result1.append(np.average(auc_per_fold))
    FKNN_result_mean.append(auc_result1)

bb=FKNN_result_mean
pd.DataFrame(bb).to_csv("FKNN-srednia.csv")
os.getcwd()
    
































# Model FRNN    
for m in [0.5,0.7,0.95,0.99,1.01,1.05]:
    i=0
    auc_per_fold=[]
    for train_index, test_index in skf:
        i+=1
        print("Stratified-TRAIN:", train_index, "Stratified-TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Train_y[train_index], Train_y[test_index]
        control={"num_class":2,"m":m,"type_membership":"gradual","normalize":True}
        resultsname=str("FRNN-Missing_values-"+"mean"+"_crosvalidation_m_"+str(m)+"-i-"+str(i)+".csv")
        Xxtrain=np.c_[X_train,y_train]
        wyniki=C_FRNN_O_FRST(Xxtrain,X_test,control)
        pd.DataFrame(wyniki).to_csv(resultsname)
        auc=metrics.roc_auc_score(y_test,wyniki)
        auc_per_fold.append(auc)
    auc_result.append(np.average(auc_per_fold))

    
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=6)
    neigh.fit(train[:,:-1], y_train) 
    s=neigh.predict(Y)
    confusion_matrix(y_test,wyniki1)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # SPRAWDZENIE CZY DLA NAJLEPSZEJ METODY DLA MISSING VALUES, USUNIĘCIE KOLUMN 
# Z LICZBĄ BRAKÓW POWYŻEJ progu percent_miss POPRAWI REZULTATY 
method_for_missing_values ='wybrana'
percent_miss=0.05   
i=0    
for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = m[train_index], m[test_index]
    y_train, y_test = n[train_index], n[test_index]
    percent_of_missing_values=X_train.isnull()/len(X_train)
    if((percent_of_missing_values>).any()):
        X_train=X_train.iloc[:,list((percent_of_missing_values>percent_miss))]
        #wypełniamy missing values na zb treninogwym
        X=pd.DataFrame(X_train).fillna(pd.DataFrame(X_train).mean())
        X=scale(X)
        pca=PCA()    
        pca.fit(X)        
        var=pca.explained_variance_ratio_
        var1=np.cumsum(np.round(var,decimals=4)*100)
        number=sum(var1<threshold_PCA)
        plot_name=str("PCA"+"mean"+str(i)+".png")
        plt.plot(var1)
        plt.savefig(plot_name)
        plt.show(block=False)
        plt.savefig(plot_name)
        pca=PCA(n_components=number)
        pca.fit(X)    
        X=pca.fit_transform(X)
        #Przygotowujemy dane testowe
        X_test=pd.DataFrame(X_test).fillna(pd.DataFrame(X_test).mean())
        X_test=scale(X_test)
        Y=pca.fit_transform(X_test)
        control={"num_class":2,"m":3,"type_membership":"cos"}
        resultsname=str("DelCol-"+"Missing_values-"+"mean"+"_crosvalidation_"+str(i))
        C_FRNN_O_FRST(pd.concat([pd.DataFrame(X),pd.DataFrame(y_train)],axis=1),pd.DataFrame(Y),control).to_csv(resultsname)
        i+=1


    

        
        
        
