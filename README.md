
# Architectural cluster


### Word2Vec Training and Similarity Calculation (word2vec.py):
This Python script revolves around processing text data from multiple files to train a Word2Vec model. It begins by tokenizing and preprocessing sentences, creating a Word2Vec model using the Skip Gram approach. Subsequently, the script calculates pairwise similarity scores between models based on the trained Word2Vec model. The similarity values are normalized to a range between 0 and 1. Finally, the script saves the computed similarity scores to a CSV file for further analysis. The process involves managing configurations, handling text data, and employing Word2Vec embeddings to capture semantic relationships between sentences.

### Hierarchical Clustering Script (architectural_cluster.py):
This Python script conducts hierarchical clustering analysis on a collection of models, leveraging their pairwise similarity scores. It relies on several libraries, including pandas, matplotlib, scipy, and scikit-learn. Additionally, the script incorporates functions from a local module named utils_module. The primary objective is to cluster models based on their similarities, generating a dendrogram for visualization. The resulting clusters are analyzed, and representative labels are inferred. Silhouette scores are computed to assess the quality of the clustering. The script produces detailed outputs, including a dendrogram plot and cluster information stored in an Excel file.



## Requirements

- **Python Version:**
  - Python 3.6 or later.

- **Python Libraries:**
  - pandas
  - matplotlib
  - scipy
  - scikit-learn
  - nltk
  - gensim
  - os
  - csv
- **Additional Dependencies:**
    - Ensure that the utils_module is in the same directory as the scripts.
## Installation
  ```bash
  pip install pandas matplotlib scipy scikit-learn nltk gensim 
  ```
## Configuration
Set configuration parameters in the respective configuration files:
  **Architectural cluster:**
    File path: configurations/architectural_cluster_config.json 
  - `file_path`: Path to the CSV file containing model data.
  - `common_words_to_exclude`: Common words to exclude during cluster topic inference.
  - `cluster_cut_height`: Cut height to form clusters during hierarchical analysis.
  - `numeber_of_topic_to_infer`: Number of topics to infer for each cluster.
  - `clusters_output_xlsx`: Path to the output Excel file for cluster results.
  - `metrics_sheet_name`: Name of the Excel sheet for metrics results.
  - `cluster_sheet_name`: Name of the Excel sheet for cluster results.

  **Word2Vec Configuration:**
    File path: configurations/word2vec_config.json
  - `model_text_file_path`: Path to the directory containing model text files.
  - `similarity_csv_file_path`: Path to the output CSV file for Word2Vec similarity.

## Explanation and Usage

### Architectural Cluster Script (architectural_cluster.py)
**infer_cluster_label Function:**

- Infers a representative label for a cluster based on the frequency of words in the cluster labels.
- Takes a list of cluster labels and the number of topics to infer.
- Uses NLTK for tokenization and frequency distribution.

**Configuration Setup:**

- Loads configuration settings from a JSON file.

**Data Loading:**

- Reads data from a CSV file specified in the configuration.

**Similarity Matrix Extraction:**

- Extracts similarity scores from the DataFrame excluding the "model_name" column.

**Hierarchical Clustering:**

- Converts the similarity matrix into a condensed distance matrix.
- Calculates the linkage matrix using the average linkage method.

**Dendrogram Plotting:**

- Creates a dendrogram using the hierarchical clustering results.

**Cluster Assignment:**

- Assigns cluster labels using a specified cutoff height for hierarchical clustering.

**Cluster Analysis:**

- Calculates silhouette score for all samples together.
- Infers representative labels for each cluster using the infer_cluster_label function.

**Results Output:**

- Saves results to an Excel file, including silhouette scores and cluster information.

**Display:**

- Displays the dendrogram and silhouette analysis.


**Run Script:**
```bash
python architectural_cluster.py
```
### Word2Vec Similarity Script (word2vec.py)
**Configuration Setup:**

Loads configuration settings from a JSON file.

**Data Preparation:**

- Iterates through text files in a specified directory.
- Reads the content of each file, preprocesses the text, and tokenizes it into sentences.
- Stores the sentences in a dictionary (`model_sentences`) where keys are model names and values are lists of sentences.

**Text Preprocessing:**

- Tokenizes and preprocesses all sentences.
- Removes stopwords and special characters, converts text to lowercase, and handles specific replacements.

**Word2Vec Model Training:**

- Creates a Word2Vec model using the Skip Gram approach.
- Vector size is set to 100, the minimum count of words is set to 1, and the window size is set to 10.

**Similarity Calculation:**

- Calculates the similarity between all pairs of sentences using the trained Word2Vec model.
- Normalizes the similarity values to a range between 0 and 1.

**CSV File Creation:**

- Saves the calculated similarity values to a CSV file.
- Each row and column correspond to model names, and the cell values represent the normalized similarity between the models.

**Output Display:**

- Prints a message indicating that similarity values have been saved to the specified CSV file.

**Run Script:**
```bash
python word2vec.py
```


## Authors

- [@maurosonzogni](https://github.com/maurosonzogni)

