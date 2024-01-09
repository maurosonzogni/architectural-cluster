import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.metrics import silhouette_score

from nltk import FreqDist
from nltk.tokenize import word_tokenize

from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, NamedStyle

# local modules
from utils_module import generate_link, load_config, create_parent_folders, remove_numbers, remove_substrings

from bard_module import infer_topic_with_bard



def infer_cluster_label(cluster_labels, number_of_topics_to_infer):
    cluster_text = ' '.join(cluster_labels)
    cluster_text = remove_numbers(cluster_text)
    cluster_text = cluster_text.replace("_", " ").replace("->", " ").replace("-", " ").replace(".", " ")
    cluster_text = remove_substrings(cluster_text, config['common_words_to_exclude'])
    words = word_tokenize(cluster_text)
    freq_dist = FreqDist(words)
    keywords = freq_dist.most_common(number_of_topics_to_infer)
    return [word for word, freq in keywords]  # Restituisce una lista di parole chiave


def build_cluster_contents(linkage_matrix, leaf_labels):
    """
    Build a dictionary containing the contents of each cluster at each stage of the linkage.

    :param linkage_matrix: The linkage matrix from hierarchical clustering
    :param leaf_labels: A list or array of leaf labels (names)
    :return: A dictionary where keys are cluster indices and values are sets of leaf names in that cluster
    """
    # Inizializza il dizionario con ogni foglia come un cluster separato
    cluster_contents = {i: {name} for i, name in enumerate(leaf_labels)}
    n = len(leaf_labels)

    # Itera attraverso ogni fusione nella matrice di linkage
    for i, row in enumerate(linkage_matrix):
        cluster1, cluster2 = int(row[0]), int(row[1])
        new_cluster = n + i
        # Combina i contenuti dei due cluster uniti
        cluster_contents[new_cluster] = cluster_contents[cluster1] | cluster_contents[cluster2]

    return cluster_contents

# Ri-eseguire il codice con la definizione della funzione poiché lo stato dell'esecuzione è stato reimpostato.

def build_cluster_information(cluster_contents):
    """
    Build a list of dictionaries with cluster number, inferred topics, and models.

    :param cluster_contents: A dictionary with sets of model names for each cluster index
    :param number_of_topics_to_infer: The number of topics to use for inferring labels
    :return: A list of dictionaries, each representing a cluster with its number, inferred topics, and models
    """
    cluster_info = []
    if config['method_to_infer_topics'] == 'NATIVE':
        for index, models in cluster_contents.items():
            topics = infer_cluster_label(models, 2)
            cluster_info.append({
                "number_of_cluster": index,
                "topic": topics,
                "models": list(models)
            })
    else:
        for index, models in cluster_contents.items():
            topics = infer_topic_with_bard(models,"","")
            cluster_info.append({
                "number_of_cluster": index,
                "topic": topics,
                "models": list(models)
            })
        
    return cluster_info


def build_model_to_topics_output(cluster_info_list):
    """
    Build a dictionary in the specified format: {model: "nomemodello", topics: ["topic1", "topic2", ...]}.

    :param cluster_info_list: A list of dictionaries, each representing a cluster with its number, inferred topics, and models
    :return: A list of dictionaries in the specified format
    """
    model_to_topics = {}
    for cluster in cluster_info_list:
        for model in cluster["models"]:
            if model not in model_to_topics:
                model_to_topics[model] = set()
            model_to_topics[model].update(cluster["topic"])

    # Convert the dictionary to the desired format
    formatted_output = [{"model": model, "topics": list(topics)} for model, topics in model_to_topics.items()]

    return formatted_output



# Configuration file path
config_file_path = 'configurations/architectural_cluster_config.json'

# Load configuration
config = load_config(config_file_path)

# Load data from the CSV file
data = pd.read_csv(config['file_path'])

# Extract the similarity similarities from the dataframe
similarities = data.iloc[1:, :].values  # Exclude the "model_name" column
similarities = data.iloc[:, 1:].values  # Exclude the "model_name" column

# Convert the distance matrix into a condensed distance matrix
# This is because our matrix is already a distance matrix, but not in the required form
condensed_distance_matrix = squareform(similarities)

# The hierarchy.linkage function from the scipy.cluster module is used to compute a linkage matrix from a distance matrix or a condensed distance vector.
# This linkage matrix is later used to construct the dendrogram, representing the hierarchical structure of the clustering.
# Z = hierarchy.linkage(y, method='single', metric='euclidean')
#   y: Distance matrix or condensed distance vector.
#   method: Linkage method to use. It can be one of the following: 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'.
#           These methods determine how the distance between clusters is calculated from distances of individual points within clusters.
#   metric: Specifies the distance metric to use. It can be a string representing one of the distance metrics supported by scipy.spatial.distance.pdist.
#           The default metric is 'euclidean'.
# The function returns the linkage matrix Z, which is an (n-1) x 4 matrix, where n is the number of observations.
# Each row of this matrix represents a fusion between two clusters and contains the following information:
#   Index of the first cluster.
#   Index of the second cluster.
#   Distance between the first and second cluster.
#   Number of observations in the resulting merged cluster.

# The dendrogram is then constructed using this linkage matrix and
# visualizes the sequence of cluster fusions during the hierarchical clustering process.

# Calculate linkage matrix using average linkage method
Z = linkage(condensed_distance_matrix, 'average')

# Create a new figure
plt.figure()

# Set the window title for the dendrogram
plt.gcf().canvas.get_tk_widget().master.title('Dendrogram')

# Set the title for the plot
plt.title('Models Cluster')

# Adjust subplot parameters for better layout
plt.subplots_adjust(left=0.08, right=0.950, bottom=0.435, top=0.935)

# Create a dendrogram with hierarchical clustering labels
dn = dendrogram(Z, labels=data['model_name'].values)

# Ora puoi usare `cluster_contents` per vedere quali modelli appartengono a ciascun cluster a ogni livello.
cluster_contents = build_cluster_contents(Z, data['model_name'].values)

cluster_info_list = build_cluster_information(cluster_contents)

# Utilizzo:
# Assumi che `cluster_info_list` sia già definito come discusso in precedenza
model_to_topics_map = build_model_to_topics_output(cluster_info_list)

# Estrarre le coordinate dei nodi interni
icoords = dn['icoord']
dcoords = dn['dcoord']

# Etichettare i nodi interni
for i, (x, y) in enumerate(zip(icoords, dcoords)):
    x_mean = np.mean(x)
    y_max = max(y)
     # Controllare se il nodo è interno (non una foglia)
    if len(set(x)) > 1:  # Un nodo interno ha più di un valore unico in 'x'
        plt.text(x_mean, y_max, f'Nodo {i}', ha='center', va='bottom')



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
# Assign cluster labels using hierarchical clustering
labels = fcluster(Z, t=config['cluster_cut_height'], criterion='distance')


# Initialize a dictionary to store models assigned to each cluster
clusters = {}

# Populate the dictionary with models assigned to each cluster
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = [data['model_name'].values[i]]
    else:
        clusters[label].append(data['model_name'].values[i])

# Calculate silhouette score for all samples together
# Note: To calculate silhouette for individual clusters, each cluster must contain at least 2 labels
silhouette_avg = silhouette_score(similarities, labels, metric='precomputed')

# Create a DataFrame for metrics
metrics_data = [[config['file_path'],config['cluster_cut_height'], silhouette_avg, config['numeber_of_topic_to_infer']]]
metrics_df = pd.DataFrame(metrics_data, columns=[
                          'file_path','cluster_cut_height', 'silhouette_avg', 'numeber_of_topic_to_infer'])

# Initialize a list to store cluster information
cluster_data = []


if config['method_to_infer_topics'] == 'NATIVE':
    # Populate the list with cluster information, including representative labels
    for cluster_num, cluster_labels in clusters.items():
        cluster_label = infer_cluster_label(cluster_labels, config['numeber_of_topic_to_infer'])
        cluster_data.append([cluster_label, cluster_num, cluster_labels])
else:
    # Populate the list with cluster information, including representative labels
    for cluster_num, cluster_labels in clusters.items():
        cluster_label = infer_topic_with_bard(cluster_labels,"","")
        cluster_data.append([cluster_label, cluster_num, cluster_labels])
    
    
# Create a DataFrame for cluster information
cluster_df = pd.DataFrame(cluster_data, columns=[
                          'cluster_topic', 'cluster_number', 'contained_models_name'])

# Trova il massimo numero di colonne tra tutte le liste
max_num_colonne = max(len(labels) for labels in clusters.values())

# Costruisci le colonne separatamente
for i in range(max_num_colonne):
    colonna_label = f'model_{i+1}'
    cluster_df[colonna_label] = [generate_link(labels[i]) if i < len(labels) else '' for labels in clusters.values()]


# Create the folder structure if it doesn't exist
create_parent_folders(config['clusters_output_xlsx'])

# Write the silhouette score DataFrame to Excel
metrics_df.to_excel(config['clusters_output_xlsx'],
                    sheet_name=config['metrics_sheet_name'], index=False)

  # Convertendo la lista di dizionari in un DataFrame
model_cluster_chain_df = pd.DataFrame(model_to_topics_map)

    # Convertendo la colonna 'topics' in stringa per una migliore visualizzazione in Excel
model_cluster_chain_df['topics'] = model_cluster_chain_df['topics'].apply(lambda x: ', '.join(x))

    
# Crea uno stile per l'attributo hyperlink
hyperlink_style = NamedStyle(name='hyperlink', font=Font(underline='single', color='0563C1'))
# Append the cluster information DataFrame to the existing Excel file
with pd.ExcelWriter(config['clusters_output_xlsx'], engine='openpyxl', mode='a') as writer:

    model_cluster_chain_df.to_excel(
        writer, sheet_name=config['model_cluster_chain_sheet_name'], index=False)
    
    cluster_df.to_excel(
        writer, sheet_name=config['cluster_sheet_name'], index=False)
    
    ws = writer.sheets[config['cluster_sheet_name']]


    # Imposta le colonne da trasformare in link (inizia dalla quarta colonna)
    colonne_da_trasformare = cluster_df.columns[3:]
    # Itera attraverso le colonne e le celle per aggiungere l'attributo hyperlink
    
    # Itera attraverso le colonne e le celle per aggiungere l'attributo hyperlink
    for colonna in colonne_da_trasformare:
        indice_colonna = cluster_df.columns.get_loc(colonna) + 1
        for col in ws.iter_cols(min_col=indice_colonna, max_col=indice_colonna, min_row=2):
            for indice, cella in enumerate(col, start=2):
                cella.style = hyperlink_style
                valore_cella = cella.value
                valore_ipo = None if valore_cella is None else f'=HYPERLINK("{valore_cella}", "{valore_cella}")'
                ws[f'{get_column_letter(indice_colonna)}{indice}'] = valore_ipo

    # Imposta la larghezza automatica delle colonne
    for col_num, _  in enumerate(cluster_df.columns, 1):
        col_letter = get_column_letter(col_num)
        max_length = 0
        for row in ws[f'{col_letter}']:
            try:
                if len(str(row.value)) > max_length:
                    max_length = len(row.value)
            except:
                pass
        adjusted_width = (max_length + 2)
        ws.column_dimensions[col_letter].width = adjusted_width


print(
    f"\nThe results have been saved in the Excel file: {config['clusters_output_xlsx']}")

# Display the dendrogram and silhouette analysis
plt.show()
