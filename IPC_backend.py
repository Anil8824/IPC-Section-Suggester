import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Ensure NLTK Data is Available
# -----------------------------
try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)

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

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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

    # Gradually lower threshold until suggestions found
    while len(relevant_indices) < min_suggestions and similarity_threshold > 0:
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        similarity_threshold -= 0.05

    sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)

    # Return clean dict list
    suggestions = dataset.iloc[sorted_indices][:min_suggestions][[
        "Description", "Offense", "Punishment", "Cognizable", "Bailable", "Court"
    ]].to_dict(orient="records")

    return suggestions
