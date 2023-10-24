import numpy as np
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

# Genera una matrice di similarità fittizia (assumiamo di avere già una matrice di similarità)
# Ecco un esempio con una matrice di similarità casuale
matrix = np.random.rand(10, 10)
# Imposta la diagonale a zeri
np.fill_diagonal(matrix, 0)

# Rendi la matrice simmetrica copiando gli elementi nella parte superiore a quella inferiore
matrix = np.triu(matrix, k=1) + np.triu(matrix, k=1).T

print(matrix)



# Calcola il linkage (collegamento) gerarchico
Z = hierarchy.linkage(distance.squareform(matrix), method='average')

# Crea il dendrogramma
plt.figure()
dn = hierarchy.dendrogram(Z)

# Mostra il dendrogramma
plt.show()
