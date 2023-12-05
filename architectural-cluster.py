import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib.pyplot as plt

from scipy.spatial.distance import  squareform
from sklearn.metrics import silhouette_score

# per inferire il topic
from nltk import FreqDist
from nltk.tokenize import  word_tokenize

import re

def remove_numbers(sentence):
    # Utilizza un'espressione regolare per eliminare i numeri e l'underscore attaccato
    sentence_without_numbers = re.sub(r'\d+_', '', sentence)
    return sentence_without_numbers

def infer_cluster_label(cluster_labels):
    
    # Unire tutte le etichette del cluster in un unico testo
    cluster_text = ' '.join(cluster_labels)

    cluster_text = remove_numbers(cluster_text)
    # Replace underscores with spaces
    cluster_text = cluster_text.replace("_", " ")
    
    
    
    cluster_text = cluster_text.lower()

    # remove circumstances common words
    cluster_text = cluster_text.replace("this", "")
    # Se si rimuove instance alcuni modelli è come se non avessero parole, es 173 e 852
    cluster_text = cluster_text.replace("instance", "")
    cluster_text = cluster_text.replace("impl", "")
    cluster_text = cluster_text.replace("imp", "")
    cluster_text = cluster_text.replace("sensor", "")
    cluster_text = cluster_text.replace("system", "")
    cluster_text = cluster_text.replace("subsystem", "")
    # single e dual?
    cluster_text = cluster_text.replace("single", "")
    cluster_text = cluster_text.replace("dual", "")
    # integration e standard
    cluster_text = cluster_text.replace("integration", "")
    cluster_text = cluster_text.replace("standard", "")
    # with hardware sub devices
    cluster_text = cluster_text.replace("with", "")
    cluster_text = cluster_text.replace("sub", "")
    cluster_text = cluster_text.replace("functional", "")
    cluster_text = cluster_text.replace("devices", "")
    cluster_text = cluster_text.replace("hardware", "")


    #
    cluster_text = cluster_text.replace("\n", "")
    

    # Tokenizzazione delle parole
    words = word_tokenize(cluster_text)

    # Calcolo della frequenza delle parole
    freq_dist = FreqDist(words)

    # Estrazione delle parole chiave (ad esempio, le prime 3 parole più comuni)
    keywords = freq_dist.most_common(2)

    # Creazione di un'etichetta rappresentativa utilizzando le parole chiave
    cluster_label = ' '.join(words for words, freq_dist in keywords)

    return cluster_label

# File path
file_path= './report/multiple_level/OR_single.match.structural.similarity.rule_06.csv'
#file_path= 'similarity_values.csv'

# Load data from the CSV file
data = pd.read_csv(file_path)

# Extract the similarity similarities from the dataframe
similarities = data.iloc[1:, :].values  # Exclude the "model_name" column
similarities = data.iloc[:, 1:].values  # Exclude the "model_name" column

print(similarities)

# Converti la matrice delle distanze in una matrice di distanza condensata
# questo perchè la nostra è già una matrice di distanza, ma non nella forma che serve
condensed_distance_matrix = squareform(similarities)

# La funzione hierarchy.linkage del modulo scipy.cluster è utilizzata per calcolare una matrice di collegamenti (linkage matrix) da una matrice di distanza o da un vettore di distanza condensato. Questa matrice di collegamenti viene successivamente utilizzata per costruire il dendrogramma, che rappresenta la struttura gerarchica del clustering.

# Z = hierarchy.linkage(y, method='single', metric='euclidean')
#   y: Matrice di distanza o vettore di distanza condensato.
#   method: Metodo di collegamento da utilizzare. Può essere uno dei seguenti: 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'. 
#           Questi metodi determinano come la distanza tra i cluster viene calcolata a partire dalle distanze dei singoli punti all'interno dei cluster.
#   metric: Specifica la metrica di distanza da utilizzare. Può essere una stringa che rappresenta una delle metriche di distanza supportate da scipy.spatial.distance.pdist. 
#           La metrica predefinita è 'euclidean'.
# La funzione restituisce la matrice di collegamenti Z, che è una matrice (n-1) x 4, dove n è il numero di osservazioni. 
# Ogni riga di questa matrice rappresenta una fusione tra due cluster e contiene le seguenti informazioni:
#   Indice del primo cluster.
#   Indice del secondo cluster.
#   Distanza tra il primo e il secondo cluster.
#   Numero di osservazioni nel cluster risultante dalla fusione.

# Il dendrogramma viene quindi costruito utilizzando questa matrice di collegamenti e 
# visualizza la sequenza delle fusioni tra cluster durante il processo di clustering gerarchico.

Z = linkage(condensed_distance_matrix, 'average')

plt.figure()

dn = dendrogram(Z, labels=data['model_name'].values)

# Scegli un'altezza di taglio per formare i cluster
# 0.9, corrisponde ad avere un unico cluster
# 0.0 ad avere che ogni singolo elemento è un cluster
#########################################################
# Possibile taglio, dataset di progetti selezionati, regola multi level:
# 1- soglia 0.2, valore taglio 0.35 (0.49435585086747874)
# 2- soglia 0.4, valore taglio 0.4 (0.6452050465514358)
# 3- soglia 0.6, valore taglio 0.4 (0.6248658309736844)
# 4- soglia 0.8, valore taglio 0.4 (0.5989859109716965)
# Nota: i valori riportati sopra massimizzano la silhouette media
#########################################################
cut_height = 0.4  # Puoi regolare questo valore in base alle tue esigenze
# Assegna i cluster in base all'altezza di taglio
labels = fcluster(Z, t=cut_height, criterion='distance')

# Creare un dizionario per memorizzare le etichette per ciascun cluster
clusters = {}

# Popola il dizionario con le etichette per ciascun cluster
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = [data['model_name'].values[i]]
    else:
        clusters[label].append(data['model_name'].values[i])

# Stampa le etichette per ciascun cluster
#for cluster_num, cluster_labels in clusters.items():
   # cluster_label = infer_cluster_label(cluster_labels)
    #print(f"Cluster {cluster_label} ({cluster_num}): {cluster_labels}")
# Creazione di un DataFrame
df_data = []

for cluster_num, cluster_labels in clusters.items():
    cluster_label = infer_cluster_label(cluster_labels)
    df_data.append([cluster_label, cluster_num, cluster_labels])

# Definizione delle colonne
columns = ['cluster_topic', 'cluster_number', 'contained_models_name']

# Creazione del DataFrame
df = pd.DataFrame(df_data, columns=columns)

# Salvataggio del DataFrame in un file Excel
excel_file_path = 'output_clusters.xlsx'
df.to_excel(excel_file_path, index=False)

# Stampa del DataFrame
print(df)

print(f"\nI risultati sono stati salvati nel file Excel: {excel_file_path}")

# Calcola l'indice di silhouette per tutti i campioni insieme
# Nota: se si vuole calcolare la silhouette dei singoli cluster è necessario che il cluster contenga almeno 2 label
silhouette_avg = silhouette_score(similarities, labels, metric='precomputed')
print(f"Silhouette Score dell'intero clustering: {silhouette_avg}")

    
plt.show()









