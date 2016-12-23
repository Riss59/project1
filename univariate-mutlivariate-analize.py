
import outliers
import plotly
import scipy
from scipy.spatial.distance import mahalanobis
import scipy as sp
import pandas as pd
import statsmodels
###########
#######
##########
def reject_outliers(sr, iq_range=50):
    pcnt = (100 - iq_range) / 2
    qlow, median, qhigh = np.percentile(sr,[pcnt, 50, 100-pcnt])
    iqr = qhigh - qlow
    sr[sr<qlow-1.5*iqr]=np.nan
    sr[sr>qhigh+1.5*iqr]=np.nan
    return (sr)   

              
def column_reject(sr):
    array_delete_columns=[]
    for i in range(int(len(sr[0]))):
        if(sum(np.isnan(sr[:,i]))/float(len(sr))>0.15):
                print i
                array_delete_columns.append(i)
    sr=np.delete(sr,array_delete_columns,1)
    results={"data":sr,"del_columns":array_delete_columns}
    return (results)
                          


def mahalanobisR(X,meanCol,IC):
    m = []
    for i in range(X.shape[0]):
        m.append(mahalanobis(np.array(X.iloc[i,:]),meanCol,IC) ** 2)
    p_value=1-sp.stats.chi2.cdf(m,X.shape[1])
    rows=p_value>0.01
    results={"data":X[rows],"rows":rows}
    return(results)

def KMO(set):
    cor=np.corrcoef(set)
    inv_cor=np.linalg.inv(cor)
    Matrix = [[0 for x in range(inv_cor.shape[0])] for y in range(inv_cor.shape[0])] 
    for i in range(inv_cor.shape[0]):
        for j in range(i,cor.shape[1]):
            while True:
                try:
                    Matrix[i][j]=-inv_cor[i,j]/(inv_cor[i,i]*inv_cor[j,j])**(0.5)
                except TypeError:
                    Matrix[i][j]=0
            Matrix[j,i]=Matrix[i,j]
    nominator=sum(inv_cor^2)-sum(np.diagonal(inv_cor^2))
    denominator=nominator+sum(Matrix^2)-sum(np.diagonal(Matrix^2))
    kmo=nominator/denominator
    return(kmo)
    

 #
#xhx=Train_sets_x_wmmv
#xhx=pd.DataFrame(xhx)
#Sx = xhx.cov() 
#IC = sp.linalg.inv(Sx + np.eye(Sx.shape[0])*10**(-6))
#meanCol = xhx.mean().values
#X=xhx
 #
#set=Train_sets_x_wmv_cr   
# Results

from sklearn import metrics
import pandas as pd
from ggplot import *

fpr, tpr, _ = metrics.roc_curve(ytest, preds)

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')
