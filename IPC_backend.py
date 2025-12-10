import re
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util

# -----------------------------------------
# SIMPLE CUSTOM PREPROCESSOR (NO NLTK)
# -----------------------------------------
def preprocess_text(text):
    text = text.lower()

    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Manually defined stopwords (very small list)
    stop_words = {
        "the", "is", "am", "are", "was", "were", "a", "an", "and", "or", "in",
        "on", "at", "to", "from", "of", "for", "by", "with", "about", "into",
        "that", "this", "it", "as", "be", "been", "have", "has", "had", "my"
    }

    words = [w for w in text.split() if w not in stop_words]

    return " ".join(words)


# -----------------------------------------
# LOAD DATA + MODEL
# -----------------------------------------
with open("preprocess_data.pkl", "rb") as f:
    new_ds = pickle.load(f)

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


# -----------------------------------------
# SUGGEST IPC SECTIONS
# -----------------------------------------
def suggest_sections(complaint, dataset, min_suggestions=3):

    processed = preprocess_text(complaint)

    complaint_embedding = model.encode(processed)
    section_embedding = model.encode(dataset["Combo"].tolist())

    similarities = util.pytorch_cos_sim(complaint_embedding, section_embedding)[0]

    similarity_threshold = 0.2
    relevant_indices = []

    while len(relevant_indices) < min_suggestions and similarity_threshold >= 0:
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        similarity_threshold -= 0.05

    sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)

    suggestions = dataset.iloc[sorted_indices][:min_suggestions][[
        "Description", "Offense", "Punishment", "Cognizable", "Bailable", "Court"
    ]].to_dict(orient="records")

    return suggestions
