import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ----------------------------------------------------
# LOAD NEW DATASET (USE THIS)
# ----------------------------------------------------
dataset = pd.read_csv("IPC_1000_dataset.csv")

# Create Combo column if not exists
if "Combo" not in dataset.columns:
    dataset["Combo"] = (
        dataset["Description"].astype(str) + " " +
        dataset["Offense"].astype(str) + " " +
        dataset["Punishment"].astype(str)
    )

# Load AI model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


# ----------------------------------------------------
# Custom Light Preprocessing
# ----------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    words = text.split()

    stop_words = {"the","is","are","was","were","am","to","of","in","and","on","at","a","an","for"}
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


# ----------------------------------------------------
# RULE ENGINE (Exact crime → exact IPC)
# ----------------------------------------------------
def rule_engine(complaint):
    t = complaint.lower()
    rules = []

    # Theft (379)
    if any(w in t for w in ["steal","stole","robbed","theft","snatched"]):
        rules.append({
            "Description": "Theft",
            "Offense": "IPC 379",
            "Punishment": "Up to 3 years or fine or both",
            "Cognizable": "Cognizable",
            "Bailable": "Bailable",
            "Court": "Any Magistrate"
        })

    # House Theft (380)
    if ("house" in t or "home" in t) and any(w in t for w in ["stole","theft","robbed"]):
        rules.append({
            "Description": "House Theft",
            "Offense": "IPC 380",
            "Punishment": "Up to 7 years + Fine",
            "Cognizable": "Cognizable",
            "Bailable": "Non-Bailable",
            "Court": "Magistrate"
        })

    # Criminal Intimidation / Blackmail (503 / 384)
    if any(w in t for w in ["threaten","threatening","blackmail","extort"]):
        rules.append({
            "Description": "Criminal Intimidation / Blackmail",
            "Offense": "IPC 503 / IPC 384",
            "Punishment": "Up to 3 years + Fine",
            "Cognizable": "Non-Cognizable",
            "Bailable": "Bailable",
            "Court": "Any Magistrate"
        })

    # Cyber Extortion (384 + IT Act Section 66D)
    if any(w in t for w in ["instagram","online","otp","cyber","leak"]):
        rules.append({
            "Description": "Cyber Fraud / Extortion",
            "Offense": "IPC 384 + IT Act 66D",
            "Punishment": "Up to 3 years + Fine",
            "Cognizable": "Cognizable",
            "Bailable": "Bailable",
            "Court": "Any Magistrate"
        })

    return rules


# ----------------------------------------------------
# AI Model (Semantic Matching)
# ----------------------------------------------------
def ai_model_suggestions(complaint, min_suggestions=3):

    text = preprocess_text(complaint)

    try:
        query_emb = model.encode(text)
        data_emb = model.encode(dataset["Combo"].tolist())
    except:
        return []

    sims = util.pytorch_cos_sim(query_emb, data_emb)[0]

    top_idx = sims.topk(min_suggestions).indices.tolist()

    return dataset.iloc[top_idx][[
        "Description", "Offense", "Punishment", "Cognizable", "Bailable", "Court"
    ]].to_dict(orient="records")


# ----------------------------------------------------
# FINAL HYBRID FUNCTION
# ----------------------------------------------------
def suggest_sections(complaint):

    final = []

    # 1) Rule Engine first
    final.extend(rule_engine(complaint))

    # 2) AI model suggestions
    ai_results = ai_model_suggestions(complaint)
    final.extend(ai_results)

    # Remove duplicates
    seen = set()
    unique = []
    for item in final:
        key = item["Offense"]
        if key not in seen:
            unique.append(item)
            seen.add(key)

    return unique[:5]
