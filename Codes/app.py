import os

os.environ["OPENAI_API_KEY"] = (
    "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"
)

import streamlit as st
import pandas as pd
import numpy as np
import torch
import cv2
from transformers import DistilBertTokenizer
import torch.nn.functional as F
from keybert import KeyLLM
from sentence_transformers import SentenceTransformer
from SPARQLWrapper import SPARQLWrapper, JSON
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from clip_class import *

st.set_page_config(
    page_title="Your App Title",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load necessary data and initialize model
@st.cache_resource
def load_data_and_model():
    captions = pd.read_csv("captions.csv")
    id_to_value_map = captions.set_index("image")["caption"].to_dict()
    image_embeddings = np.load("image_embeddings.npy")
    image_embeddings = torch.from_numpy(image_embeddings).to(CFG.device)

    model = CLIPModel().to(CFG.device)
    model.load_state_dict(
        torch.load(
            "/Users/udyansachdev/Downloads/661_final/Weights/best_model_config_(0.0005, 1e-05, 1e-06, 0.01).pt",
            map_location=CFG.device,
        )
    )
    model.eval()

    return captions, id_to_value_map, image_embeddings, model


captions, id_to_value_map, image_embeddings, model = load_data_and_model()


# Utility functions
def find_matches(model, image_embeddings, query, image_filenames, n=5):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    image = cv2.imread(f"{CFG.image_path}/{matches[0]}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return matches, image


def preprocess_keyword(keyword):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = word_tokenize(keyword)
    processed_tokens = [
        stemmer.stem(lemmatizer.lemmatize(token)).capitalize() for token in tokens
    ]
    processed_phrase = " ".join(processed_tokens)
    return processed_phrase, tokens


def query_dbpedia_for_keywords(keywords):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    results_dict = {}

    for keyword in keywords:
        processed_phrase, unigrams = preprocess_keyword(keyword)
        query = f"""
        SELECT ?abstract ?label
        WHERE {{
            ?subject rdfs:label "{keyword}"@en.
            ?subject dbo:abstract ?abstract.
            FILTER(LANG(?abstract) = "en")
        }}
        LIMIT 1
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        if results["results"]["bindings"]:
            abstract = results["results"]["bindings"][0]["abstract"]["value"]
            results_dict[keyword] = abstract
            continue

        query = f"""
        SELECT ?abstract ?label
        WHERE {{
            ?subject rdfs:label "{processed_phrase}"@en.
            ?subject dbo:abstract ?abstract.
            FILTER(LANG(?abstract) = "en")
        }}
        LIMIT 1
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        if results["results"]["bindings"]:
            abstract = results["results"]["bindings"][0]["abstract"]["value"]
            results_dict[keyword] = abstract
            continue

        unigram_abstracts = []
        for unigram in set(unigrams):
            query = f"""
            SELECT ?abstract ?label
            WHERE {{
                ?subject rdfs:label "{unigram}"@en.
                ?subject dbo:abstract ?abstract.
                FILTER(LANG(?abstract) = "en")
            }}
            LIMIT 1
            """
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

            if results["results"]["bindings"]:
                abstract = results["results"]["bindings"][0]["abstract"]["value"]
                unigram_abstracts.append(abstract)

        if unigram_abstracts:
            combined_abstract = " ".join(unigram_abstracts)
            results_dict[keyword] = combined_abstract

    return results_dict


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

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    db = Chroma.from_texts(texts, embeddings)

    # Create retriever
    retriever = db.as_retriever()

    # Create OpenAI language model
    llm = OpenAI()

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return qa_chain


# Streamlit app
st.title("Visual and Contextual Question Answering")

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Image Retrieval")
    query = st.text_input("Enter your image query:")
    if query:
        matches, image = find_matches(
            model, image_embeddings, query, valid_df["image"].values, n=5
        )
        st.image(image, caption="Retrieved Image", use_column_width=True)

        extracted_values = [id_to_value_map[i] for i in matches if i in id_to_value_map]
        # st.write("Extracted keywords:")
        # st.write(", ".join(extracted_values))

with col2:
    st.subheader("Question Answering")
    if query:
        contextual_captions = captions.loc[
            captions["image"] == matches[0], "caption"
        ].tolist()

        with st.spinner("Preparing for question answering..."):
            data = query_dbpedia_for_keywords(extracted_values)

        qa_chain = setup_rag(data, contextual_captions)

        question = st.text_input("Ask a question about the image:")
        if question:
            with st.spinner("Generating answer..."):
                answer = qa_chain.run(question)
            st.write("Answer:")
            st.write(answer)

    else:
        st.write(
            "Please enter an image query first to retrieve contextual information."
        )

# Add a sidebar with additional information
st.sidebar.title("About")
st.sidebar.info(
    "This application uses a CLIP model for image retrieval and a QA chain "
    "for answering questions based on the retrieved image and contextual information."
)
st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    1. Enter an image query in the left column.
    2. View the retrieved image.
    3. Ask a question about the image in the right column.
    4. Get an AI-generated answer based on the image context.
    """
)
