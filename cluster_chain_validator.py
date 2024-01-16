import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils_module import load_config  # Custom utility module for configuration

# Function to calculate similarity between two strings
def calculate_similarity(str1, str2, consider_order=False):
    """
    Calculate the similarity between two strings.
    
    Args:
        str1 (str): First string.
        str2 (str): Second string.
        consider_order (bool): If True, use cosine similarity (order-sensitive). Otherwise, use Jaccard similarity (order-insensitive).

    Returns:
        float: Similarity score between the two strings.
    """
    if consider_order:
        # Using cosine similarity (order-sensitive)
        vectorizer = CountVectorizer().fit([str1, str2])
        vec1 = vectorizer.transform([str1]).toarray()
        vec2 = vectorizer.transform([str2]).toarray()
        return cosine_similarity(vec1, vec2)[0][0]
    else:
        # Using Jaccard similarity (order-insensitive)
        set1 = set(str1.split())
        set2 = set(str2.split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union else 0

# Path to the configuration file
config_file_path = 'configurations/cluster_chain_validator_config.json'

# Load configuration from the file
config = load_config(config_file_path)

# Reading datasets from configured file paths
df1 = pd.read_excel(config['file_path_1'], config['model_cluster_chain_sheet_name'])
df2 = pd.read_excel(config['file_path_2'], config['model_cluster_chain_sheet_name'])

# Extracting column names from file names
column_name_1 = "topics_of_" + os.path.basename(config['file_path_1'])
column_name_2 = "topics_of_" + os.path.basename(config['file_path_2'])

# Renaming columns for clarity
df1 = df1.rename(columns={"topics": column_name_1})
df2 = df2.rename(columns={"topics": column_name_2})

# Merging the two dataframes on the 'model' column
merged_df = pd.merge(df1[['model', column_name_1]], df2[['model', column_name_2]], on='model', how='inner')

# Calculating similarity between topic columns of the merged dataframe
merged_df['similarity'] = merged_df.apply(lambda x: calculate_similarity(x[column_name_1], x[column_name_2], config['consider_order']), axis=1)

# Saving the merged dataframe to an Excel file
merged_df.to_excel(config['output_file'], index=False)
