#wybor proby ze zbioru danych
#Funkcja getsizesample liczy ilosc elementow elementow proby 
#Funkcja getstratisfiedSampe wczytuje wylosowane przypadkowo warstwowo dane i zwraca
# data frame ktory zawiera probe
import math
import random
#Stratisfied Sample
def getstratisfiedSample(filepath):
    Target_data=data=pd.read_csv(filepath,usecols=[0])
    percent_default=sum(Target_data.iloc[:,0])/float(len(Target_data))
    number_of_sample=getsizesample(percent_default,len(Target_data),0.01)
    number_from_default=int(math.ceil(percent_default*number_of_sample))
    number_from_non_default=number_of_sample-number_from_default
    range_of_non_default=Target_data[Target_data.iloc[:,0]==0].index.tolist()
    range_of_default=Target_data[Target_data.iloc[:,0]==1].index.tolist()
    indx_rows_non_default=random.sample(range_of_non_default,number_from_non_default)
    indx_rows_default=random.sample(range_of_default,number_from_default)
    iter_csv = pd.read_csv(filepath, iterator=True, chunksize=10000)
    df_default = pd.concat([chunk[chunk.index.isin(chunk.index&indx_rows_default)] for chunk in iter_csv])
    iter_csv = pd.read_csv(filepath, iterator=True, chunksize=10000)
    df_non_default = pd.concat([chunk[chunk.index.isin(chunk.index&indx_rows_non_default)] for chunk in iter_csv])
    sample=pd.concat([df_non_default,df_default])

#napisac funkcje ktora bedzie liczyla ilosc elementow probki
def getsizesample(Percent,Number_of_elements,error_rate):
    Z=2.59   # przedział ufnoci 1%
    n=Percent*(1-Percent)/((error_rate/Z)**2+Percent*(1-Percent)/Number_of_elements)
    return(n)
