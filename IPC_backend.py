import re
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -----------------------------
# SIMPLE TOKENIZER (No NLTK Punkt Needed)
# -----------------------------
def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_text(text):
    words = simple_tokenize(text)

    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

# -----------------------------
# Load Data + Model
# -----------------------------
with open("preprocess_data.pkl", "rb") as f:
    new_ds = pickle.load(f)

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# -----------------------------
# Suggest IPC Sections
# -----------------------------
def suggest_sections(complaint, dataset, min_suggestions=3):

    preprocessed_complaint = preprocess_text(complaint)

    complaint_embedding = model.encode(preprocessed_complaint)
    section_embedding = model.encode(dataset["Combo"].tolist())

    similarities = util.pytorch_cos_sim(complaint_embedding, section_embedding)[0]

    similarity_threshold = 0.2
    relevant_indices = []

    while len(relevant_indices) < min_suggestions and similarity_threshold > 0:
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        similarity_threshold -= 0.05

    sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)

    suggestions = dataset.iloc[sorted_indices][:min_suggestions][[
        "Description", "Offense", "Punishment", "Cognizable", "Bailable", "Court"
    ]].to_dict(orient="records")

    return suggestions
