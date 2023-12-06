import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib.pyplot as plt

from scipy.spatial.distance import  squareform
from sklearn.metrics import silhouette_score

# per inferire il topic
from nltk import FreqDist
from nltk.tokenize import  word_tokenize

# local modules
from utils_module import load_config, create_parent_folders, remove_numbers, remove_substrings


def infer_cluster_label(cluster_labels, number_of_topics_to_infer):
    """
    Infer a representative label for a cluster based on the frequency of words in the cluster labels.

    Args:
        cluster_labels (list): A list of strings representing labels assigned to the cluster.
        number_of_topics_to_infer (int): The number of topics to infer from the cluster labels.

    Returns:
        str: A representative label inferred from the cluster labels.

    Example:
        >>> infer_cluster_label(["data analysis", "data visualization", "data processing"], 2)
        'data analysis visualization'
    """
    # Concatenate all cluster labels into a single text
    cluster_text = ' '.join(cluster_labels)

    # Remove numeric digits from the cluster text
    cluster_text = remove_numbers(cluster_text)

    # Replace underscores with spaces
    cluster_text = cluster_text.replace("_", " ")

    # Remove common words specified in the configuration
    cluster_text = remove_substrings(cluster_text, config['common_words_to_exclude'])

    # Tokenize the words
    words = word_tokenize(cluster_text)

    # Calculate word frequency
    freq_dist = FreqDist(words)

    # Extract keywords (e.g., the top N most common words)
    keywords = freq_dist.most_common(number_of_topics_to_infer)

    # Create a representative label using the keywords
    cluster_label = ' '.join(word for word, freq in keywords)

    return cluster_label


# Configuration file path 
config_file_path = 'configurations/architectural_cluster_config.json'

# Load configuration
config = load_config(config_file_path)

# Load data from the CSV file
data = pd.read_csv(config['file_path'])

# Extract the similarity similarities from the dataframe
similarities = data.iloc[1:, :].values  # Exclude the "model_name" column
similarities = data.iloc[:, 1:].values  # Exclude the "model_name" column

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
labels = fcluster(Z, t=config['cluster_cut_height'], criterion='distance')

# Creare un dizionario per memorizzare le etichette per ciascun cluster
clusters = {}

# Popola il dizionario con le etichette per ciascun cluster
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = [data['model_name'].values[i]]
    else:
        clusters[label].append(data['model_name'].values[i])


# Calcola l'indice di silhouette per tutti i campioni insieme
# Nota: se si vuole calcolare la silhouette dei singoli cluster è necessario che il cluster contenga almeno 2 label
silhouette_avg = silhouette_score(similarities, labels, metric='precomputed')
# Creazione di un DataFrame
metrics_data = [[config['file_path'], silhouette_avg]]

# Creazione del DataFrame
metrics_df = pd.DataFrame(metrics_data, columns=['file_path','silhouette_avg'])

# Creazione di un DataFrame
cluster_data = []

for cluster_num, cluster_labels in clusters.items():
    cluster_label = infer_cluster_label(cluster_labels, config['numeber_of_topic_to_infer'])
    cluster_data.append([cluster_label, cluster_num, cluster_labels])

# Creazione del DataFrame
cluster_df = pd.DataFrame(cluster_data, columns=['cluster_topic', 'cluster_number', 'contained_models_name'])

# Create the folder structure if it doesn't exist
create_parent_folders(config['clusters_output_xlsx'])

metrics_df.to_excel(config['clusters_output_xlsx'], sheet_name=config['metrics_sheet_name'], index=False)

with pd.ExcelWriter(config['clusters_output_xlsx'], engine='openpyxl', mode='a') as writer:
    cluster_df.to_excel(writer, sheet_name=config['cluster_sheet_name'], index=False)


print(f"\nI risultati sono stati salvati nel file Excel: {config['clusters_output_xlsx']}")
  
plt.show()
