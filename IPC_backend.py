import re
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util

# -------------------------------------------
# Custom Preprocessing (NO NLTK NEEDED)
# -------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # remove special characters
    words = text.split()

    stop_words = {"the","is","are","was","were","am","to","of","in","and","on","at","a","an","for"}
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


# -------------------------------------------
# Load Dataset + Model
# -------------------------------------------
with open("preprocess_data.pkl", "rb") as f:
    new_ds = pickle.load(f)

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


# -------------------------------------------
# RULE ENGINE (Very Powerful & Smart)
# -------------------------------------------
def rule_engine(complaint):
    text = complaint.lower()
    rules = []

    # 1) Theft (379)
    if any(word in text for word in ["steal", "stole", "theft", "robbed", "lost", "took"]):
        rules.append({
            "Description": "Theft",
            "Offense": "IPC 379",
            "Punishment": "As per IPC section",
            "Cognizable": "Cognizable",
            "Bailable": "Bailable",
            "Court": "Any Magistrate"
        })

    # 2) House Theft (380)
    if any(word in text for word in ["house", "home", "room", "inside my house"]):
        if any(word in text for word in ["stole", "theft", "steal", "robbed"]):
            rules.append({
                "Description": "House Theft",
                "Offense": "IPC 380",
                "Punishment": "As per IPC section",
                "Cognizable": "Cognizable",
                "Bailable": "Non-Bailable",
                "Court": "Magistrate"
            })

    # 3) House Breaking (454)
    if any(word in text for word in ["broke the lock", "break the lock", "forced entry", "break in", "door broken"]):
        rules.append({
            "Description": "House Breaking",
            "Offense": "IPC 454",
            "Punishment": "Up to 3 years + Fine",
            "Cognizable": "Cognizable",
            "Bailable": "Non-Bailable",
            "Court": "Any Magistrate"
        })

    # 4) House Trespass by Night (457)
    if "night" in text or "midnight" in text or "late night" in text:
        if any(word in text for word in ["break", "trespass", "entered", "broke", "forced entry"]):
            rules.append({
                "Description": "Lurking House Trespass by Night",
                "Offense": "IPC 457",
                "Punishment": "Up to 5 years + Fine",
                "Cognizable": "Cognizable",
                "Bailable": "Non-Bailable",
                "Court": "Magistrate"
            })

    return rules


# -------------------------------------------
# AI MODEL (Semantic Search)
# -------------------------------------------
def ai_model_suggestions(complaint, dataset, min_suggestions=3):

    processed = preprocess_text(complaint)

    try:
        complaint_emb = model.encode(processed)
        section_emb = model.encode(dataset["Combo"].tolist())
    except:
        return []

    sims = util.pytorch_cos_sim(complaint_emb, section_emb)[0]

    # Top matches
    top_idx = sims.topk(min_suggestions).indices.tolist()

    suggestions = dataset.iloc[top_idx][[
        "Description", "Offense", "Punishment", "Cognizable", "Bailable", "Court"
    ]].to_dict(orient="records")

    return suggestions


# -------------------------------------------
# FINAL FUNCTION (Hybrid Output)
# -------------------------------------------
def suggest_sections(complaint, dataset):

    final_output = []

    # 1️⃣ Rule Engine First
    rule_hits = rule_engine(complaint)
    final_output.extend(rule_hits)

    # 2️⃣ Then AI Suggestions
    ai_hits = ai_model_suggestions(complaint, dataset)
    final_output.extend(ai_hits)

    # Remove duplicates (optional)
    unique = []
    seen = set()

    for item in final_output:
        key = item["Offense"]
        if key not in seen:
            unique.append(item)
            seen.add(key)

    return unique[:5]  # return best 5 results
