import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

file_path = './report/rule_60.csv'
# Carica i dati dal file CSV
data = pd.read_csv(file_path)

# Escludi la prima colonna (presumendo che sia l'ID o una colonna non utile per il clustering)
X = data.iloc[:, 1:].values

# Standardizza le feature
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
X_scaled = X

# Utilizza NearestNeighbors per stimare il parametro eps (distanza massima tra due campioni)
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
eps = np.percentile(distances[:, 1], 90)

# Esegui DBSCAN con il parametro eps stimato
dbscan = DBSCAN(eps=eps, min_samples=5)

cluster_labels = dbscan.fit_predict(X_scaled)

# Calcola il numero di cluster trovati
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

# Calcola l'indice di Silhouette
silhouette_avg = metrics.silhouette_score(X_scaled, cluster_labels)

# Crea una colormap con un numero sufficiente di colori per i cluster
colors = plt.cm.tab20(np.linspace(0, 1, len(set(cluster_labels))))


# Trova gli indici dei punti in ciascun cluster
cluster_indices = {}
for label in set(cluster_labels):
    cluster_indices[label] = np.where(cluster_labels == label)[0]

for label, indices in cluster_indices.items():
    cluster_data = data.iloc[indices]
    print(f"Cluster {label}:")
    print(cluster_data.describe())
    print()

# Crea un plot 2D dei cluster
plt.figure(figsize=(8, 6))

for cluster_label, color in zip(set(cluster_labels), colors):
    
    cluster_points = X_scaled[cluster_labels == cluster_label]
    plt.scatter(cluster_points[:, 36], cluster_points[:, 1], c=[color], label=f'Cluster {cluster_label}')

plt.title('Clustering DBSCAN con Auto-Tuning')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Aggiungi informazioni sul numero di cluster e Silhouette Score al grafico
plt.text(0.05, 0.9, f'Numero di Cluster: {n_clusters}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'Silhouette Score: {silhouette_avg:.2f}', transform=plt.gca().transAxes)

# Sposta la legenda a destra
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()