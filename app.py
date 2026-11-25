
import streamlit as st
import nltk
import spacy
import fitz
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="AI Notes Summarizer",
    layout="wide"
)

st.title(" AI Research Paper & Notes Summarizer")
st.write("Automated NLP that changes Research Papers and Study Notes into compact summary")

@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None

nlp = load_nlp_model()

def extract_pdf_text(pdf_file):
    text = ""

    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page in pdf:
        text += page.get_text() + "\n"

    pdf.close()

    return text

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))

    filtered_words = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    lemmatizer = WordNetLemmatizer()

    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in filtered_words
    ]

    return tokens, filtered_words, lemmatized_words

def summarize_text(text, summary_ratio=0.4):
    sentences = sent_tokenize(text)

    if len(sentences) <= 2:
        return text

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    stop_words = set(stopwords.words("english"))

    useful_words = [
        word for word in words
        if word not in stop_words
    ]

    word_frequencies = Counter(useful_words)

    if not word_frequencies:
        return text

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies:
        word_frequencies[word] /= max_frequency

    sentence_scores = {}

    for index, sentence in enumerate(sentences):
        sentence_words = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())

        score = sum(
            word_frequencies.get(word, 0)
            for word in sentence_words
        )

        sentence_scores[index] = score / max(len(sentence_words), 1)

    number_of_sentences = max(
        2, int(len(sentences) * summary_ratio)
    )

    number_of_sentences = min(
        number_of_sentences, len(sentences)
    )

    selected_indices = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )[:number_of_sentences]

    selected_indices.sort()

    return " ".join(
        sentences[index] for index in selected_indices
    )

def extract_keywords(text):
    sentences = sent_tokenize(text)

    if not sentences:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )

    try:
        matrix = vectorizer.fit_transform(sentences)
        scores = matrix.sum(axis=0).A1
        keywords = vectorizer.get_feature_names_out()

        ranked_keywords = sorted(
            zip(keywords, scores),
            key=lambda item: item[1],
            reverse=True
        )

        return [word for word, score in ranked_keywords[:10]]

    except ValueError:
        return []

def extract_entities(text):
    if nlp is None:
        return None

    doc = nlp(text)

    return [
        {"Text": entity.text, "Type": entity.label_}
        for entity in doc.ents
    ]

st.divider()

st.subheader("Choose Your Input Method")

input_method = st.radio(
    "How would you like to provide your document?",
    ["Paste Text", "Upload PDF"],
    horizontal=True
)

text = ""

if input_method == "Paste Text":

    text = st.text_area(
        "Enter your research paper or notes:",
        height=250,
        placeholder="Paste your text here..."
    )

else:

    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        try:
            text = extract_pdf_text(uploaded_file)

            if text.strip():
                st.success("PDF uploaded and text extracted successfully!")
                st.info(f"Extracted {len(text.split())} words from the PDF.")
            else:
                st.warning("No readable text found in this PDF.")

        except Exception:
            st.error("Unable to read this PDF. Please upload a valid PDF.")

st.divider()

summary_length = st.slider(
    "Choose Summary Length (%)",
    min_value=20,
    max_value=80,
    value=40,
    step=10
)

if st.button("Generate Summary", type="primary", use_container_width=True):

    if not text.strip():
        st.warning("Please provide some text or upload a readable PDF.")

    elif len(text.split()) < 10:
        st.warning("Please provide at least 10 words for meaningful analysis.")

    else:

        with st.spinner("Analyzing your document using NLP..."):

            summary = summarize_text(
                text,
                summary_length / 100
            )

            tokens, filtered_words, lemmatized_words = preprocess_text(text)

            keywords = extract_keywords(text)

            entities = extract_entities(text)

        st.success("Analysis completed successfully!")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Summary",
            "Keywords",
            "Text Preprocessing",
            "Named Entities"
        ])

        with tab1:

            st.subheader("Generated Summary")
            st.write(summary)

            original_words = len(text.split())
            summary_words = len(summary.split())

            col1, col2 = st.columns(2)

            col1.metric("Original Word Count", original_words)
            col2.metric("Summary Word Count", summary_words)

            reduction = (1 - summary_words / original_words) * 100

            st.metric("Word Reduction", f"{reduction:.1f}%")

        with tab2:

            st.subheader("Important Keywords")

            if keywords:
                st.write(", ".join(keywords))
            else:
                st.info("No keywords found.")

        with tab3:

            st.subheader("Text Preprocessing Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Tokens", len(tokens))
            col2.metric("After Stop Word Removal", len(filtered_words))
            col3.metric("After Lemmatization", len(lemmatized_words))

            st.write("**Tokenized Words**")
            st.write(tokens[:100])

            st.write("**After Stop Word Removal**")
            st.write(filtered_words[:100])

            st.write("**After Lemmatization**")
            st.write(lemmatized_words[:100])

        with tab4:

            st.subheader("Named Entity Recognition")

            if entities is None:
                st.error("spaCy model is missing. Run: python -m spacy download en_core_web_sm")

            elif entities:
                st.table(entities)

            else:
                st.info("No named entities detected.")

st.divider()

st.caption("AI Research Paper & Notes Summarizer | Python | Streamlit | NLTK | Scikit-learn | spaCy")
