import os
import re
from openai import OpenAI
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from SPARQLWrapper import SPARQLWrapper, JSON

os.environ["OPENAI_API_KEY"] = (
    "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"
)

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_keywords_openai(text, max_keywords=10):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts keywords.",
            },
            {
                "role": "user",
                "content": f"Extract up to {max_keywords} important keywords or phrases from the following text and give the proper noun proper first capital letter:\n\n{text}\n\nKeywords:",
            },
        ],
        max_tokens=60,
        n=1,
        temperature=0.5,
    )
    keywords = response.choices[0].message.content.strip().split(", ")
    return keywords[:max_keywords]


def expand_keywords(keywords):
    lemmatizer = WordNetLemmatizer()
    expanded = []

    for keyword in keywords:
        if keyword.lower() in STOP_WORDS:
            continue

        if keyword not in expanded:
            expanded.append(keyword)

        tokens = word_tokenize(keyword)
        processed_tokens = []

        for token in tokens:
            if token.lower() in STOP_WORDS:
                continue

            lemma_noun = lemmatizer.lemmatize(token.lower(), pos="n")
            lemma_verb = lemmatizer.lemmatize(token.lower(), pos="v")

            variations = [
                token,
                token.lower(),
                token.capitalize(),
                lemma_noun,
                lemma_verb,
                lemma_noun.capitalize(),
                lemma_verb.capitalize(),
            ]

            unique_variations = list(
                set(v for v in variations if v.lower() not in STOP_WORDS)
            )
            processed_tokens.extend(unique_variations)

        if len(tokens) > 1 and not all(token.lower() in STOP_WORDS for token in tokens):
            lowercase_version = " ".join(
                token.lower() for token in tokens if token.lower() not in STOP_WORDS
            )
            titlecase_version = " ".join(
                token.capitalize()
                for token in tokens
                if token.lower() not in STOP_WORDS
            )

            if lowercase_version not in expanded:
                expanded.append(lowercase_version)
            if titlecase_version not in expanded:
                expanded.append(titlecase_version)

        for variation in processed_tokens:
            if variation not in expanded:
                expanded.append(variation)

    expanded = [keyword for keyword in expanded if keyword.lower() not in STOP_WORDS]

    return expanded


def clean_keywords(raw_keywords):
    raw_text = " ".join(raw_keywords)
    raw_text = raw_text.replace("\n", " ")
    cleaned_keywords = re.findall(r"\d+\.\s*([^\d]+)", raw_text)
    return [keyword.strip() for keyword in cleaned_keywords]


def query_dbpedia_for_keywords(keywords):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    results_dict = {}
    processed_keywords = set()

    for keyword in keywords:
        keyword_lower = keyword.lower()
        if any(kw in processed_keywords for kw in keyword_lower.split()):
            continue

        keyword = keyword.replace("\n", " ")
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
            processed_keywords.update(keyword_lower.split())
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
            processed_keywords.update(keyword_lower.split())
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
            processed_keywords.update(keyword_lower.split())

    return results_dict


def preprocess_keyword(keyword):
    tokens = word_tokenize(keyword)
    processed_tokens = [token.capitalize() for token in tokens]
    processed_phrase = " ".join(processed_tokens)
    return processed_phrase, tokens


def process_caption(caption):
    keywords = extract_keywords_openai(caption)
    print("Extracted keywords:", keywords)

    clean = clean_keywords(keywords)
    print("Cleaned keywords:", clean)

    expanded = expand_keywords(clean)
    print("Expanded keywords:", expanded)

    return query_dbpedia_for_keywords(expanded)


def answer_question(question, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions. I am giving you a bunch of strings as the context, please provide answer using your brain. Even if full context not there, you'll defo get some hint from the context so answer accordingly and make sense. Don't worry about the context being too long, just focus on the question and answer it.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\nContext: {context}\nAnswer:",
            },
        ],
        max_tokens=60,
        n=1,
        temperature=0.5,
    )
    print(context)
    return response.choices[0].message.content.strip()
