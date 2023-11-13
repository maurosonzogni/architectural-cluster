import pandas as pd

from scipy.cluster import hierarchy

import matplotlib.pyplot as plt

from scipy.spatial.distance import  squareform

# File path
file_path = './report/single_level/OR_single.match.structural.similarity.rule_02.csv'

# Load data from the CSV file
data = pd.read_csv(file_path)


# Extract the similarity similarities from the dataframe
similarities = data.iloc[1:, :].values  # Exclude the "model_name" column
similarities = data.iloc[:, 1:].values  # Exclude the "model_name" column

print(similarities)

# Converti la matrice delle distanze in una matrice di distanza condensata
# questo perchè la nostra è già una matrice di distanza, ma non nella forma che serve
condensed_distance_matrix = squareform(similarities)

Z = hierarchy.linkage(condensed_distance_matrix, 'average')

plt.figure()

dn = hierarchy.dendrogram(Z, labels=data['model_name'].values)

plt.show()

