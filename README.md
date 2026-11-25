python -m streamlit run app.py
# AI Research & Notes Summarizer

## Project Overview

AI Research & Notes Summarizer is a simple Natural Language Processing (NLP) application developed using Python.

It helps students convert lengthy research papers and study notes into shorter, meaningful summaries. It also extracts important keywords, performs text preprocessing, and identifies named entities.

## Features

1. Text Summarization - Generates concise summaries by identifying important sentences.
2. Text Preprocessing - Performs tokenization, stop word removal, and lemmatization.
3. Keyword Extraction - Uses TF-IDF to identify important keywords.
4. Named Entity Recognition - Identifies people, organizations, and locations using spaCy.

## Technologies Used

- Python
- Streamlit
- NLTK
- Scikit-learn
- spaCy

## NLP Concepts Implemented

- Tokenization
- Stop Word Removal
- Lemmatization
- TF-IDF
- Named Entity Recognition (NER)
- Extractive Text Summarization

## Project Structure

AI_Notes_Summarizer/
│
├── app.py
├── requirements.txt
└── README.md

## Installation

### Step 1: Install Required Libraries

Open Command Prompt inside the project folder and execute:

pip install -r requirements.txt

### Step 2: Download the spaCy English Model

python -m spacy download en_core_web_sm

### Step 3: Run the Application

streamlit run app.py

The application will open in your default web browser.

## How to Use

1. Open the application.
2. Paste your research paper or study notes.
3. Select the desired summary length.
4. Click Generate Summary.
5. Explore the generated summary, keywords, preprocessing results, and named entities.

## Objective

The main objective of this project is to demonstrate the practical application of Natural Language Processing techniques in simplifying lengthy educational and research documents.

## Developed Using

Python | Streamlit | NLTK | Scikit-learn | spaCy
