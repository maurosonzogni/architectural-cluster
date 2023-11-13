import os
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import nltk
#nltk.download("punkt")
import gensim
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import csv

# Directory containing your text files
directory_path = "***"

# List to store sentences from all files
all_sentences = []

# Dictionary to store model names and sentences
model_sentences = {}

# Iterate through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        model_name = filename.split('.')[0]  # Extract model name from the filename

        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Replace underscores with spaces
        text = text.replace("_", " ")
        text = text.lower()

        # remove circumstances common words
        text = text.replace("this", "")
        # Se si rimuove instance alcuni modelli è come se non avessero parole, es 173 e 852
        #text = text.replace("instance", "")
        text = text.replace("impl", "")
        text = text.replace("imp", "")
        text = text.replace("\n", "")

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
    words = [word.lower() for word in words if word.isalnum() and word not in stop_words]
    preprocessed_sentences.append(words)

# Create Skip Gram model
model = gensim.models.Word2Vec(preprocessed_sentences, min_count=1, vector_size=100, window=5, sg=1)

# Create a list to store similarity values
similarity_values = []

# Calculate the similarity between all pairs of sentences
model_names = list(model_sentences.keys())

for i, model_name1 in enumerate(model_names):
    sentences1 = model_sentences[model_name1]
    for j, model_name2 in enumerate(model_names):
        sentences2 = model_sentences[model_name2]

        for sentence1 in sentences1:
            for sentence2 in sentences2:
                # Tokenize and preprocess the sentences
                sentence1_words = [word.lower() for word in word_tokenize(sentence1) if word.isalnum() and word not in stop_words]
                sentence2_words = [word.lower() for word in word_tokenize(sentence2) if word.isalnum() and word not in stop_words]

                # Calculate the similarity between the two sentences
                similarity = model.wv.n_similarity(sentence1_words, sentence2_words)
                # 1-similarity to align with structual similarity
                # 0.0 equals
                # 1.0 completely different
                similarity_values.append([model_name1, model_name2, 1 - similarity])

# Save the similarity values to a CSV file
csv_file = "similarity_values.csv"
with open(csv_file, 'w', newline='') as file:
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

print(f"Similarity values have been saved to {csv_file}")
