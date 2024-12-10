import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import cv2
import torch.nn.functional as F

# Placeholder imports - replace with your actual implementations
from clip_class import *
from keywords import *
from rag import *

# Ensure OpenAI API key is set securely
os.environ["OPENAI_API_KEY"] = (
    "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"
)


# Caching Utility Functions
@st.cache_data
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def find_matches_cached(model, image_embeddings, query, image_filenames, n=5):
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


@st.cache_resource
def process_caption_cached(caption):
    return process_caption(caption)


@st.cache_resource
def setup_rag_cached(data, contextual_captions):
    return setup_rag(data, contextual_captions)


@st.cache_resource
def answer_question_cached(question, corpus):
    return answer_question(question, corpus)


# Custom CSS for enhanced UI
def local_css():
    st.markdown(
        """
        <style>
        .main-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 1px solid #4a4a4a;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .highlight {
            background-color: #e6f3ff;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Utility functions
def validate_query(query):
    if not query or len(query.strip()) < 2:
        st.warning("Please enter a meaningful query (at least 2 characters).")
        return False
    return True


def calculate_answer_confidence(question, answer, context):
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(context_words.intersection(answer_words))
    return min(overlap / len(answer_words), 1.0) if answer_words else 0.0


@st.cache_resource
def load_data_and_model():
    captions = pd.read_csv("captions.csv")
    id_to_value_map = captions.set_index("image")["caption"].to_dict()
    image_embeddings = np.load("image_embeddings.npy")
    image_embeddings = torch.from_numpy(image_embeddings).to(CFG.device)
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(
        torch.load(
            "best_model_config_(0.0005, 1e-05, 1e-06, 0.01).pt",
            map_location=CFG.device,
        )
    )
    model.eval()
    return captions, id_to_value_map, image_embeddings, model


def main():
    st.set_page_config(
        page_title="Visual Q&A Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    local_css()
    captions, id_to_value_map, image_embeddings, model = load_data_and_model()

    # Main title and description
    st.title("🖼️ Visual Question Answering Explorer")
    st.markdown(
        """
        ### Unlock Insights from Images
        Enter a query to find relevant images and ask context-based questions.
        """
    )

    col1, col2 = st.columns([2, 3])

    # Image Retrieval Section
    with col1:
        st.subheader("🔎 Image Retrieval")
        query = st.text_input("Enter your image search query:")

        # Initialize session state for matches and selected image
        if "matches" not in st.session_state:
            st.session_state.matches = []
            st.session_state.selected_image = None
            st.session_state.selected_image_path = None

        # Process query and retrieve matches
        if query and validate_query(query):
            st.session_state.matches, primary_image = find_matches_cached(
                model, image_embeddings, query, valid_df["image"].values, n=5
            )

        # Render radio buttons for selecting an image from matches
        if st.session_state.matches:
            st.session_state.selected_image_path = st.radio(
                "Select an image:",
                st.session_state.matches,
                format_func=lambda x: f"Image {st.session_state.matches.index(x) + 1}",
                key="image_selector",
            )

        # Load and display the selected image
        if st.session_state.selected_image_path:
            selected_image = load_image(
                f"{CFG.image_path}/{st.session_state.selected_image_path}"
            )
            st.image(selected_image, caption=f"Selected Image", use_column_width=True)

        # Display matched images in columns
        if st.session_state.matches:
            st.subheader("Matched Images")
            cols = st.columns(5)

            for i, img_path in enumerate(st.session_state.matches):
                with cols[i]:
                    img = load_image(f"{CFG.image_path}/{img_path}")
                    st.image(img, caption=f"Image {i+1}", width=100)

    # Contextual Question Answering Section
    with col2:
        st.subheader("❓ Contextual Question Answering")
        if st.session_state.matches:
            contextual_captions = captions.loc[
                captions["image"] == st.session_state.selected_image_path, "caption"
            ].tolist()
            extracted_values = [id_to_value_map[st.session_state.selected_image_path]]
            with st.spinner("Preparing context..."):
                data = process_caption_cached(extracted_values[0])
                qa_chain = setup_rag_cached(data, contextual_captions)

            question = st.text_input(
                "Ask a question about the image:",
                value=st.session_state.get("question", ""),
                key="question_input",
            )
            st.session_state.question = question

            if question:
                with st.spinner("Generating context-aware answer..."):
                    data_string = " ".join(
                        [f"{key}: {value}" for key, value in data.items()]
                    )
                    answer = qa_chain.run(question)
                    full_context = (
                        [data_string] + contextual_captions + [question] + [answer]
                    )
                    corpus = " ".join(full_context)
                    answer = answer_question_cached(question, corpus)
                    confidence = calculate_answer_confidence(question, answer, corpus)
                st.markdown("### Answer")
                st.markdown(f"**Confidence:** {confidence:.2%}")
                st.write(answer)
        else:
            st.info("Please search for an image first to enable Q&A.")

    # Sidebar Content
    st.sidebar.title("🤖 About Visual Q&A")
    st.sidebar.markdown(
        """
        ### How It Works
        - **Image Retrieval:** Advanced CLIP model finds semantically similar images
        - **Context Understanding:** Analyzes image captions and metadata
        - **Intelligent Q&A:** Generates answers based on contextual understanding
        """
    )
    st.sidebar.title("📝 Instructions")
    st.sidebar.markdown(
        """
        1. Enter an image search query
        2. Select the most relevant image
        3. Ask a question about the image
        4. Receive a context-aware answer
        """
    )


if __name__ == "__main__":
    main()
