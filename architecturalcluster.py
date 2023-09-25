import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from st_dbscan import ST_DBSCAN
import csv

file ='./report/rule_70.csv'
f = open(file, 'rt')
reader = csv.reader(f)
headers = next(reader, None)
headers = headers[1:]

df = pd.read_csv(file)
print(df.head(5))
df = df.drop(['model_name'], axis=1)
df = df.map(lambda x: x * 1000 if isinstance(x, (int, float)) else x)

# transform to numpy array
data = df.loc[:, headers].values
st_dbscan = ST_DBSCAN(eps1 = 15, eps2 = 5, min_samples=2)
st_dbscan.fit(data) 




def plot(data, labels):
    colors = ['#6da832','#ff7f00','#cab2d6','#6a3d9a']
    
    for i in range(-1, len(set(labels))):
        if i == -1:
            col = [0, 0, 0, 1]
        else:
            col = colors[i % len(colors)]
        
        clust = data[np.where(labels==i)]
        plt.scatter(clust[:,0], clust[:,1], c=[col], s=3)
    
    plt.title('Architectural Clustering',fontsize=20)
    plt.xlabel('Similarity',fontsize=14)
    plt.ylabel('Similarity',fontsize=14)
    plt.show(block=True)

    return None

plot(data[:,1:], st_dbscan.labels) 