# Import necessary libraries

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import openai
import pandas as pd
import faiss
import argparse

from sentence_transformers import SentenceTransformer

similarity_threshold_tf = 0.25
similarity_threshold_sem = 0.7

nltk.download('punkt')  # Download data for tokenizer

# Function to read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

#NOTE - vectorized using FAISS - can use other vectorization library 
def cosine_similarity_vectorized(X, Y, type):
    #normalize to make inner product equivalent to dot product

    if type == 'tf':
        X = X.toarray().astype(np.float32)
        Y = Y.toarray().astype(np.float32)

    # Ensure the arrays are C-contiguous
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)

    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    if len(Y.shape) == 1:
        Y = Y.reshape(1, -1)

    faiss.normalize_L2(X)
    faiss.normalize_L2(Y)
    dimension = Y.shape[1]  # Dimensionality of the vectors
    #Inner product index
    index = faiss.IndexFlatIP(dimension)
    index.add(Y)
    #D matrix is cosine similarity, I is indices of most similar embeddings
    D, I = index.search(X, k=Y.shape[0])
    return D

# Function to split the document into sentences
def split_into_sentences(document):
    sentences = nltk.sent_tokenize(document)
    return [sentence.strip() for sentence in sentences]

# Function to find similar sentences in two documents
def find_similar_sentences(doc1, doc2, vectorize = False):
    # Split documents into sentences
    sentences_doc1 = split_into_sentences(doc1)

    sentences_doc2 = split_into_sentences(doc2)

    # Combine all sentences for vectorization
    all_sentences = sentences_doc1 + sentences_doc2
    
    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    all_sentences_vectors = vectorizer.fit_transform(all_sentences)

    # Generate transformer based embeddings
    embeddings = np.array([model.encode(sentence) for sentence in all_sentences])

    # Compute cosine similarity for both embeddings


    if vectorize:
        similarity_matrix_tf = cosine_similarity_vectorized(all_sentences_vectors[:len(sentences_doc1)], all_sentences_vectors[len(sentences_doc1):], 'tf')
        similarity_matrix_sem = cosine_similarity_vectorized(embeddings[:len(sentences_doc1)], embeddings[len(sentences_doc1):], 'se')

    else:
        similarity_matrix_tf = cosine_similarity(all_sentences_vectors[:len(sentences_doc1)], all_sentences_vectors[len(sentences_doc1):])
        similarity_matrix_sem = cosine_similarity(embeddings[:len(sentences_doc1)], embeddings[len(sentences_doc1):])

    similar_sentences_df = pd.DataFrame(columns=["Index Doc1", "Index Doc2", "Sentence Doc1", "Sentence Doc2", "Similarity_TF", "Similarity_Sem"])
    # Find sentences with similarity above the threshold
    for i in range(similarity_matrix_tf.shape[0]):
        for j in range(similarity_matrix_tf.shape[1]):
            #inclde the text as long it has high similarity either from the tf-idf or transformer embedding
            if similarity_matrix_tf[i][j] > similarity_threshold_tf or similarity_matrix_sem[i][j] > similarity_threshold_sem:
                similar_sentences_df = pd.concat([similar_sentences_df, pd.DataFrame([{
                    "Index Doc1": i,
                    "Index Doc2": j,
                    "Sentence Doc1": sentences_doc1[i],
                    "Sentence Doc2": sentences_doc2[j],
                    "Similarity_TF": similarity_matrix_tf[i][j],
                    "Similarity_Sem": similarity_matrix_sem[i][j]
                }])], ignore_index=True)
    return similar_sentences_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', dest='model', type=str, default='all-MiniLM-L6-v2')        # Model used in algorithm
    parser.add_argument('--path1', dest='path1', type=str, default='')             # Path to first text
    parser.add_argument('--path2', dest='path2', type=str, default='')             # Path to second text
    parser.add_argument('--out_path', dest='out_path', type=str, default='')             # Path to second text
    parser.add_argument('--vectorize', default=False, action='store_true') 


    args = parser.parse_args()

    model = SentenceTransformer(args.model)

    path1 = args.path1
    path2 = args.path2
    out_path = args.out_path

    # Read texts from files
    text1 = read_text_from_file(path1)
    text2 = read_text_from_file(path2)

    similar_sentences = find_similar_sentences(text1, text2, args.vectorize)
    similar_sentences.to_csv(out_path, index = False)





