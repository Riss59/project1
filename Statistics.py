from scipy.stats import describe
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import os 
filepath="F:\data_281016_v2.csv"
    
#         Statyski Deskrypcyjne

def descriptive_statistics(dane)
    summary_statistics=[]
    for i in range(len(m[0])):
        column_data=dane[i]
        decription=pd.DataFrame(column_data).describe()
        decription[0][0]=int(len(m)-decription[0][0])
        decription=np.array(decription).T
        summary_statistics.append(decription)   
    summary_statistics=pd.concat([pd.DataFrame(cos) for cos in summary_statistics])
    summary_statistics.columns=['Missing Values','Mean',"Std",'Min','1 Quartile','Median','3 Quartile','Max']
    summary_statistics.index=["V"+str(i+1) for i in range(len(summary_statistics))]
    summary_statistics.to_csv("Statistics.csv")

    
               # Histogram 

def makehistogram(filepath):
    default=pd.read_csv(filepath,usecols=[0])
    default_nr=sum(default.iloc[:,0])
    non_default_nr=len(default)-default_nr
    objects = ('Default','No Default')
    y_pos = np.arange(len(objects))
    performance = [default_nr,non_default_nr]
    bar_width = 0.35
    plt.bar(y_pos, performance, align='center',width=bar_width ,alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Ilosc')
    plt.title('Histogram zmiennej Target')
    plt.savefig('hist.png') 
    
