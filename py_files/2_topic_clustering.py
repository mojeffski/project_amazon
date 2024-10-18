#!/usr/bin/env python
# coding: utf-8

# ### 2. Clustering Task:
# For transformer model
import csv
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import string
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from gensim.models.ldamodel import LdaModel
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
from huggingface_hub import login

# For LDA model
import os
import nltk
from nltk.corpus import stopwords

get_ipython().system("pip install gensim")


# Load environment variables from the .env file
load_dotenv()

# Access the Hugging Face token
hf_token = os.getenv("HF_TOKEN")

# Use the token to authenticate with Hugging Face
login(token=hf_token)


project_path = "dataset/original/1429_1.csv"


df = pd.read_csv(project_path)

product_names = df.categories.value_counts().keys().to_list()


# ## Kmeans approach:

# Load pre-trained transformer model and tokenizer (example:
# 'distilbert-base-uncased')
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings from product names
def get_embeddings(texts):
    embeddings = []
    for text in texts:
        # Tokenize the text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True)

        # Get embeddings from the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Average the token embeddings to get a single embedding for the
        # sentence
        embeddings.append(
            outputs.last_hidden_state.mean(
                dim=1).squeeze().cpu().numpy())

    return np.array(embeddings)


# Get embeddings for all product names
embeddings = get_embeddings(product_names)

# Create embeddings DataFrame
embeddings_df = pd.DataFrame(embeddings, index=product_names)

# Clustering with KMeans
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings_df)

# Create a DataFrame to show product names and their corresponding clusters
df_results = pd.DataFrame({"Product Name": product_names, "Cluster": clusters})

# Display the clustering results
print(df_results.sort_values("Cluster").head())

# Reduce the dimensionality of the embeddings to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_df)

# Create a scatter plot of the clusters in 2D space
plt.figure(figsize=(10, 7))
plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=clusters,
    cmap="viridis",
    s=50,
    alpha=0.7,
)
plt.colorbar()
plt.title("Product Clusters Visualized with t-SNE")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# Add labels to some points for better understanding
for i in range(len(df_results)):
    if i % 10 == 0:  # Label every 10th point to avoid overcrowding
        plt.text(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            df_results["Product Name"].iloc[i],
            fontsize=9,
        )

plt.show()

# #visual inspecition shows that there are many categories mixed together. Therefore, a second approach was started relying on Topic Modeling / LDA

# ## LDA aproach
# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")


# Sample processed documents (you can replace this with your actual list
# of documents)
documents = product_names

# 1. Tokenize and preprocess the documents
stop_words = set(stopwords.words("english"))
punctuation_table = str.maketrans("", "", string.punctuation)


def preprocess(doc):
    tokens = word_tokenize(doc)  # Convert to lower case and tokenize
    tokens = [
        word.translate(punctuation_table) for word in tokens
    ]  # Remove punctuation
    tokens = [
        word for word in tokens if word.isalpha()
    ]  # Remove non-alphabetical tokens
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


processed_docs = [preprocess(doc) for doc in documents]

# 2. Create a dictionary representation of the documents
dictionary = Dictionary(processed_docs)

# 3. Create a Bag of Words corpus
# corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Download necessary resources
nltk.download("punkt")  # Download the Punkt tokenizer
nltk.download("stopwords")

# Sample processed documents (you can replace this with your actual list
# of documents)
documents = product_names

# 1. Tokenize and preprocess the documents
stop_words = set(stopwords.words("english"))
punctuation_table = str.maketrans("", "", string.punctuation)


def preprocess(doc):
    tokens = word_tokenize(doc.lower())  # Convert to lower case and tokenize
    tokens = [
        word.translate(punctuation_table) for word in tokens
    ]  # Remove punctuation
    tokens = [
        word for word in tokens if word.isalpha()
    ]  # Remove non-alphabetical tokens
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


processed_docs = [preprocess(doc) for doc in documents]

# 2. Create a dictionary representation of the documents
dictionary = Dictionary(processed_docs)

# 3. Create a Bag of Words corpus
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# Function to train LDA and evaluate
def train_and_evaluate_lda(
    corpus,
    dictionary,
    processed_docs,
    num_topics=5,
    passes=30,
    alpha=0.001,
    random_state=None,
):
    lda_model = LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        alpha=alpha,
        random_state=random_state,
    )

    # Calculate Coherence Score
    coherence_model_lda = CoherenceModel(
        model=lda_model,
        texts=processed_docs,
        dictionary=dictionary,
        coherence="c_v")
    coherence_score = coherence_model_lda.get_coherence()

    # Calculate Perplexity
    perplexity = lda_model.log_perplexity(corpus)

    return lda_model, coherence_score, perplexity


# Now you can proceed with your original code for iterating over different
# random states
best_model = None
best_score = {
    "iteration": None,
    "combined_score": -float("inf"),  # Starting score is very low
    "coherence": -1,  # Start with a very low coherence
    "perplexity": float("inf"),  # Start with a very high perplexity
}

# Dictionary to store results for each iteration
model_scores = {}

for i in range(100):  # Or 1000 iterations
    random_state = i
    lda_model, coherence_score, perplexity = train_and_evaluate_lda(
        corpus, dictionary, processed_docs, random_state=random_state
    )

    # Track each model's performance
    combined_score = 0.7 * coherence_score + 0.3 * (
        1 / perplexity
    )  # Higher coherence, lower perplexity is better

    model_scores[i] = {
        "coherence": coherence_score,
        "perplexity": perplexity,
        "combined_score": combined_score,
    }
    # Check if this model is the best based on combined score
    if combined_score > best_score["combined_score"]:
        # Delete all other models in the folder
        best_score["iteration"] = i
        best_score["combined_score"] = combined_score
        best_score["coherence"] = coherence_score
        best_score["perplexity"] = perplexity
        best_model = lda_model
        lda_model.save(f"models/lda_models/best_lda_model_{i}.model")

# Print the best results
print(f"Best Model Iteration: {best_score['iteration']}")
print(f"Best Coherence Score: {best_score['coherence']}")
print(f"Best Perplexity: {best_score['perplexity']}")
print(f"Best Combined Score: {best_score['combined_score']}")

# Optionally: print or save model_scores dictionary for all iterations
print(model_scores)

get_ipython().system("pip install pyLDAvis")

lda_model = LdaModel.load(
    f"models/lda_models/best_lda_model_{best_score['iteration']}.model"
)

# Prepare the visualization data
lda_display = gensimvis.prepare(lda_model, corpus, dictionary)
# Save the visualization to an HTML file
pyLDAvis.save_html(
    lda_display,
    "flask_project/templates/lda_visualization.html")

# 1. Get the first term from each topic and use them as labels
topic_labels = {}
for idx, topic in lda_model.show_topics(
        num_topics=4, num_words=1, formatted=False):
    # Get the top term for each topic
    first_term = topic[0][0]
    topic_labels[idx] = first_term

# 2. Prepare the data for export
output_data = []

for i, doc in enumerate(corpus):
    # Get the dominant topic for each document
    topic_distribution = lda_model.get_document_topics(doc)
    dominant_topic = max(topic_distribution, key=lambda x: x[1])[
        0
    ]  # Get the topic with the highest probability

    # Get the label for the dominant topic (first term only)
    label = topic_labels.get(dominant_topic, f"Topic {dominant_topic}")

    # Append the original document (product names), dominant topic, and the
    # label to the output data
    output_data.append([product_names[i], dominant_topic, label])

# 3. Export the data to a CSV file
csv_filename = "dataset/interim/amazon_review_categories.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["category", "dominant_topic", "label"])  # CSV header
    writer.writerows(output_data)

print(f"Document-topic data exported to {csv_filename}")