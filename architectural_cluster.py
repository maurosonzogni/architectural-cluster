import pandas as pd
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
    cluster_text = cluster_text.replace("-", " ")

    # Remove common words specified in the configuration
    cluster_text = remove_substrings(
        cluster_text, config['common_words_to_exclude'])

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

# Populate the list with cluster information, including representative labels
for cluster_num, cluster_labels in clusters.items():
    cluster_label = infer_cluster_label(
        cluster_labels, config['numeber_of_topic_to_infer'])
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

# Crea uno stile per l'attributo hyperlink
hyperlink_style = NamedStyle(name='hyperlink', font=Font(underline='single', color='0563C1'))
# Append the cluster information DataFrame to the existing Excel file
with pd.ExcelWriter(config['clusters_output_xlsx'], engine='openpyxl', mode='a') as writer:
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
