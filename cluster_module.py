from ai_modules.gtp_module import infer_topic_with_GPT
from utils_module import remove_numbers, remove_substrings
from nltk import FreqDist
from nltk.tokenize import word_tokenize


def infer_cluster_topics(cluster_labels, number_of_topics_to_infer, common_words_to_exclude):
    """
    Infers the most common labels for a cluster of texts.

    This function processes a list of cluster labels to identify the most common keywords. It does so by
    tokenizing the combined text of the labels, filtering out common words and numbers, and then using
    frequency distribution to find the most common words.

    :param cluster_labels: A list of labels (strings) associated with a cluster.
    :param number_of_topics_to_infer: The number of top keywords to return.
    :param common_words_to_exclude: A list of common words to exclude from the analysis.
    :return: A list of the most frequent keywords in the cluster labels.
    """

    # Combine all labels into a single string
    cluster_text = ' '.join(cluster_labels)

    # Remove numbers and replace certain punctuation with spaces for better word separation
    cluster_text = remove_numbers(cluster_text)  # Removing numbers using regex for efficiency
    cluster_text = cluster_text.replace("_", " ").replace("->", " ").replace("-", " ").replace(".", " ")

    # Remove specified common words
    cluster_text = remove_substrings(cluster_text, common_words_to_exclude)

    # Tokenize the text into words
    words = word_tokenize(cluster_text)

    # Calculate the frequency distribution of the words
    freq_dist = FreqDist(words)

    # Retrieve the most common words up to the specified number
    keywords = freq_dist.most_common(number_of_topics_to_infer)

    # Return only the words, not their frequencies
    return [word for word, freq in keywords]



def build_cluster_contents_map(linkage_matrix, leaf_labels, cut_height, min_cluster_size, max_cluster_size):
    """
    Builds a map of clusters formed at and above a specified height in a hierarchical clustering dendrogram.

    This function processes a linkage matrix obtained from hierarchical clustering and constructs a mapping
    of clusters to their member labels. It considers only those clusters that are formed at or above a 
    specified cut height. The function also records the indices of clusters that were merged to form a new cluster.

    :param linkage_matrix: The linkage matrix obtained from hierarchical clustering.
    :param leaf_labels: A list or array of labels corresponding to the leaves in the dendrogram.
    :param cut_height: The height at which the dendrogram is cut to form clusters.
    :return: A dictionary where each key is a cluster index, and each value is a dictionary containing:
             - 'members': a set of labels in that cluster
             - 'merged': a string representation of the indices of the clusters that were merged
    """
    # Initialize a dictionary with each leaf as a separate cluster
    cluster_contents = {i: {'members': {name}, 'merged': None} for i, name in enumerate(leaf_labels)}
    n = len(leaf_labels)

    # Iterate through each merge in the linkage matrix
    for i, row in enumerate(linkage_matrix):
        cluster1, cluster2, height, _ = row
        new_cluster = n + i

        # Merge clusters but do not add them to the final dictionary if below the cut height
        merged_members = cluster_contents.get(cluster1, {'members': set()})['members'] | cluster_contents.get(cluster2, {'members': set()})['members']
        # For sure cluster are integer.
        if height == cut_height:
            merged_clusters = str(int(cluster1)) + " || " + str(int(cluster2))
        else:
            merged_clusters = str(int(cluster1)) + " | " + str(int(cluster2))
        cluster_contents[new_cluster] = {'members': merged_members, 'merged': merged_clusters}

        # Remove old clusters if they are below the cut height
        if height < cut_height:
            if cluster1 in cluster_contents:
                del cluster_contents[cluster1]
            if cluster2 in cluster_contents:
                del cluster_contents[cluster2]

    # Filter out intermediate clusters formed below the cut height and those with fewer than 5 elements
    cluster_contents_map = {k: v for k, v in cluster_contents.items() if len(v['members']) >= min_cluster_size and len(v['members']) <= max_cluster_size}

    # get cluster_indexes of remained clusters
    cluster_indexes = [i for i in cluster_contents_map]

    
    for index in cluster_indexes:
        # se numero compare in almeno un merged, allora è il più basso e sta sul taglio
        for i in cluster_contents_map:
        
            if str(index) in cluster_contents_map[i]['merged']:
                cluster_contents_map[i]['merged'] = (cluster_contents_map[i]['merged']).replace("|", "---")

    # NOTE: clusters must have at least 2 elements, if at a certain height the model has not yet "joined" a cluster, it is therefore automatically excluded
    # it is important to verify the minimum number of elements in the clusters, the requirement to have clusters with at least N elements leads to the exclusion of clusters at the cut height
    # thus resulting in certain models not being considered at a certain level
    
    return cluster_contents_map


def build_cluster_information(cluster_contents, method_to_infer_topics, numeber_of_topic_to_infer, common_words_to_exclude):
    """
    Build a list of dictionaries with cluster number, inferred topics, and models.

    :param cluster_contents: A dictionary with sets of model names for each cluster index
    :param number_of_topics_to_infer: The number of topics to use for inferring labels
    :return: A list of dictionaries, each representing a cluster with its number, inferred topics, and models
    """
    cluster_info = []
    
    method_to_infer_topics_not_found = True
    # NOTE Python does not have a native switch construction. If one wishes to add a different mode, add a custom if condition.
    if method_to_infer_topics == 'GPT':
        method_to_infer_topics_not_found = False
        for index, models in cluster_contents.items():
            topics = infer_topic_with_GPT(models['members'],"","",'user')
            cluster_info.append({
                "number_of_cluster": index,
                "topic": topics,
                "models": list(models['members'])
            })
    # Both, native and default case
    if method_to_infer_topics_not_found or method_to_infer_topics == 'NATIVE':
        for index, models in cluster_contents.items():
            topics = infer_cluster_topics(models['members'], numeber_of_topic_to_infer, common_words_to_exclude)
            cluster_info.append({
                "number_of_cluster": index,
                "topic": topics,
                "models": list(models['members'])
            })

    return cluster_info



def build_model_topics_chain_in_clusters(cluster_info_list):
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