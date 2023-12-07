import os
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import csv

# local modules
from utils_module import load_config, create_parent_folders

config_file_path = 'configurations/word2vec_config.json'

config = load_config(config_file_path)

# List to store sentences from all files
all_sentences = []

# Dictionary to store model names and sentences
model_sentences = {}

directory_path = config['model_text_file_path']
# Iterate through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        # Extract model name from the filename without extension
        model_name = os.path.splitext(filename)[0]

        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Replace underscores with spaces
        text = text.replace("_", " ")
        text = text.lower()

        # TODO valutare se rimuovere le stesse di architectural_cluster_config o farne uno custom differente
        # potrebbe essere sensato provare a separare le parlole quando si trova una maiuscola?
        text = text.replace("this", "")
        # Se si rimuove instance alcuni modelli è come se non avessero parole, es 173 e 852
        text = text.replace("instance", "")
        text = text.replace("impl", "")
        text = text.replace("imp", "")
        text = text.replace("\n", "")
        text = text.replace("->", " ")
        text = text.replace(".", " ")

        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        all_sentences.extend(sentences)

        # Store the sentences in the dictionary with the model name

        model_sentences[model_name] = sentences

# Tokenize and preprocess the sentences
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

for sentence in all_sentences:
    words = word_tokenize(sentence)
    # word not in stop_words this exclude some single char es "i", "t"
    words = [word.lower() for word in words if word.isalnum()
             and word not in stop_words]

    preprocessed_sentences.append(words)

# Create Skip Gram model
model = Word2Vec(preprocessed_sentences, min_count=1,
                 vector_size=100, window=10, sg=1)

# Create a list to store similarity values
similarity_values = []

# Calculate the similarity between all pairs of sentences
model_names = list(model_sentences.keys())

for i, model_name1 in enumerate(model_names):
    for j, model_name2 in enumerate(model_names):

        if not preprocessed_sentences[i] or not preprocessed_sentences[j]:
            similarity = 0.0
            if not preprocessed_sentences[i] and not preprocessed_sentences[j]:
                similarity = 1.0
        else:
            # Calculate the similarity between the two sentences
            similarity = model.wv.n_similarity(
                preprocessed_sentences[i], preprocessed_sentences[j])

        # model.wv.n_similarity return a value +1 and -1, so we need to normalize it
        normalized_similaity = (similarity + 1) / 2

        # 1-similarity to align with structual similarity
        # 0.0 equals
        # 1.0 completely different
        similarity_values.append(
            [model_name1, model_name2, (1 - round(normalized_similaity, 3))])

create_parent_folders(config['similarity_csv_file_path'])

# Save the similarity values to a CSV file
with open(config['similarity_csv_file_path'], 'w', newline='') as file:
    writer = csv.writer(file)
    header = ["model_name"] + model_names
    writer.writerow(header)
    for model_name in model_names:
        row = [model_name] + [0] * len(model_names)
        for similarity_value in similarity_values:
            name1, name2, value = similarity_value
            if name1 == model_name:
                row[model_names.index(name2) + 1] = value
        writer.writerow(row)

print(
    f"Similarity values have been saved to {config['similarity_csv_file_path']}")
