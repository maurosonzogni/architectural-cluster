import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils_module import load_config

# Funzione per calcolare la similarità, con un parametro per l'ordine
def calculate_similarity(str1, str2, consider_order=False):
    if consider_order:
        # Calcolo della similarità del coseno
        vectorizer = CountVectorizer().fit([str1, str2])
        vec1 = vectorizer.transform([str1]).toarray()
        vec2 = vectorizer.transform([str2]).toarray()
        return cosine_similarity(vec1, vec2)[0][0]
    else:
        # Calcolo della similarità di Jaccard
        set1 = set(str1.split())
        set2 = set(str2.split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union else 0

# Configuration file path
config_file_path = 'configurations/cluster_chain_validator_config.json'

# Load configuration
config = load_config(config_file_path)

# Leggi i due dataset
df1 = pd.read_excel(config['file_path_1'], config['model_cluster_chain_sheet_name'])
df2 = pd.read_excel(config['file_path_2'], config['model_cluster_chain_sheet_name'])

# Extracting the file name
column_name_1 = "topics_of_" + os.path.basename(config['file_path_1'])
column_name_2 = "topics_of_" + os.path.basename(config['file_path_2'])

# Rinomina le colonne
df1 = df1.rename(columns={"topics": column_name_1})
df2 = df2.rename(columns={"topics": column_name_2})

# Unisci i DataFrame
merged_df = pd.merge(df1[['model', column_name_1]], df2[['model', column_name_2]], on='model', how='inner')


# Calcola la similarità
merged_df['similarity'] = merged_df.apply(lambda x: calculate_similarity(x[column_name_1], x[column_name_2], config['consider_order']), axis=1)

# Salva il DataFrame
merged_df.to_excel(config['output_file'], index=False)
