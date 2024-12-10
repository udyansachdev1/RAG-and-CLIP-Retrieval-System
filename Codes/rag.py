import os

os.environ["OPENAI_API_KEY"] = (
    "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"
)


import streamlit as st
import pandas as pd
import numpy as np
import torch
import cv2

# from transformers import DistilBertTokenizer
import torch.nn.functional as F

# from keybert import KeyLLM
# from sentence_transformers import SentenceTransformer
from SPARQLWrapper import SPARQLWrapper, JSON
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download("wordnet")

from clip_class import *

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model_texter = DistilBertModel.from_pretrained("distilbert-base-uncased")


def get_distilbert_embeddings(texts):
    """
    Generate embeddings using DistilBERT.
    Args:
        texts (list of str): Input texts to generate embeddings for.
    Returns:
        torch.Tensor: Tensor containing embeddings for each input text.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_texter(**inputs)
    # Use the mean of the last hidden states for embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


class DistilBertEmbeddingFunction:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def embed_documents(self, texts):
        embeddings = get_distilbert_embeddings(texts).detach().cpu().numpy()
        return embeddings.tolist()  # Convert numpy array to list

    def embed_query(self, query):
        embedding = get_distilbert_embeddings([query]).detach().cpu().numpy()
        return embedding[
            0
        ].tolist()  # Convert numpy array to list and return the first (and only) embedding


def setup_rag(data, additional_context):
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = []

    # Add the original abstracts
    for keyword, abstract in data.items():
        chunks = text_splitter.split_text(abstract)
        texts.extend(chunks)

    # Add the additional context
    for context in additional_context:
        chunks = text_splitter.split_text(context)
        texts.extend(chunks)

    # Initialize the custom embedding function
    embedding_function = DistilBertEmbeddingFunction(tokenizer, model_texter)

    # Create vector store with the custom embedding function
    db = Chroma.from_texts(texts, embedding_function)

    # Create retriever
    retriever = db.as_retriever()

    # Create OpenAI language model
    llm = OpenAI()

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return qa_chain
