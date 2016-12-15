#wybor proby ze zbioru danych
#Funkcja getsizesample liczy ilosc elementow elementow proby 
#Funkcja getstratisfiedSampe wczytuje wylosowane przypadkowo warstwowo dane i zwraca
# data frame ktory zawiera probe
import math
import random
#Stratisfied Sample
def getstratisfiedSampe
#number_of_sample=getsizesample
Target_data=data=pd.read_csv("F:\data_281016_v2.csv",usecols=[0])
percent_default=sum(Target_data.iloc[:,0])/float(len(Target_data))
number_of_sample=8365
number_from_default=int(math.ceil(percent_default*number_of_sample))
number_from_non_default=number_of_sample-number_from_default
range_of_non_default=Target_data[Target_data.iloc[:,0]==0].index.tolist()
range_of_default=Target_data[Target_data.iloc[:,0]==1].index.tolist()
indx_rows_non_default=random.sample(range_of_non_default,number_from_non_default)
indx_rows_default=random.sample(range_of_default,number_from_default)
#zrobic tu petle
iter_csv = pd.read_csv("F:\data_281016_v2.csv", iterator=True, chunksize=10000)
df_default = pd.concat([chunk[chunk.index.isin(chunk.index&indx_rows_default)] for chunk in iter_csv])
iter_csv = pd.read_csv("F:\data_281016_v2.csv", iterator=True, chunksize=10000)
df_non_default = pd.concat([chunk[chunk.index.isin(chunk.index&indx_rows_non_default)] for chunk in iter_csv])
sample=pd.concat([df_non_default,df_default])

#napisac funkcje ktora bedzie liczyla ilosc elementow probki
def getsizesample=:
