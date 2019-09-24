import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
file_name1='result_max.csv'

my_data = genfromtxt(file_name1, delimiter=',')
my_data=my_data[1:,1:]
TP=np.sum(my_data[0:38,0:38].diagonal())
Total=np.sum(my_data[0:38,0:38])
accuracy=TP/Total
rowSum=np.sum(my_data[0:38,0:38], axis=1)
columnSum=np.sum(my_data[0:38,0:38], axis=0)
recall=my_data[0:38,0:38].diagonal()/rowSum
precision=my_data[0:38,0:38].diagonal()/columnSum
fig, ax = plt.subplots()

ax.matshow(my_data, cmap=plt.cm.Blues)
plt.show()
ax.matshow(recall, cmap=plt.cm.Blues)
plt.show()

ax.matshow(precision, cmap=plt.cm.Blues)
plt.show()

for i in range(1):
    for j in range(38):
        c = my_data[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()


selectRecall=recall[[0,1,2,5,7,8,9,11,12,13,15,16,18,20,21,25,26,28,29,30,31,32,33,34,35,36]]
selectPrecision=precision[[0,1,2,5,7,8,9,11,12,13,15,16,18,20,21,25,26,28,29,30,31,32,33,34,35,36]]
# error1=my_data[[0,1,2,5,7,8,9,11,12,13,15,16,18,20,21,25,26,28,29,30,31,32,33,34,35,36],39]-selectRecall
# # error2=my_data[39,[0,1,2,5,7,8,9,11,12,13,15,16,18,20,21,25,26,28,29,30,31,32,33,34,35,36]]-selectPrecision
file_name2='select_simple.csv'
dataframe = pd.DataFrame({ 'recall': selectRecall,'precision':selectPrecision})
dataframe.to_csv(file_name2, index=False, sep=',')