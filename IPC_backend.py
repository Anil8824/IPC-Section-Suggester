import re
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# SIMPLE TOKENIZER (NO NLTK)
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    words = text.split()
    common_stopwords = {"the","is","and","to","of","in","for","on","a","an"}
    words = [w for w in words if w not in common_stopwords]
    return " ".join(words)

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
with open("preprocess_data.pkl", "rb") as f:
    new_ds = pickle.load(f)

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# -----------------------------
# RULE-ENGINE (HIGH PRIORITY)
# -----------------------------
RULES = {
    # FRAUD / SCAM / BANK OTP
    r"(otp|bank|fraud|scam|cheat|transaction|online|loan|impersonat)": [
        ("Cheating", "IPC 420"),
        ("Cheating by Personation", "IPC 419"),
    ],

    # THEFT / MOBILE STOLEN
    r"(stole|theft|mobile|cash|wallet|pickpocket|rob|snatch)": [
        ("Theft", "IPC 379"),
        ("House Theft", "IPC 380"),
    ],

    # HOUSE BREAKING
    r"(break|broken|house|night|entered|trespass)": [
        ("House Breaking by Night", "IPC 457"),
        ("Lurking House Trespass", "IPC 460"),
    ],

    # MURDER / ATTEMPT TO MURDER
    r"(killed|murder|stab|knife|gun|shot|blood|attack)": [
        ("Murder", "IPC 302"),
        ("Attempt to Murder", "IPC 307"),
    ],

    # SEXUAL ASSAULT / HARASSMENT
    r"(rape|sexual|harass|touch|molest|forced|assault)": [
        ("Rape", "IPC 376"),
        ("Sexual Harassment", "IPC 354A"),
        ("Outraging modesty", "IPC 354"),
    ],

    # CYBER CRIME
    r"(hack|hacked|password|facebook|instagram|email)": [
        ("Identity Theft", "IT Act 66C"),
        ("Online Impersonation", "IT Act 66D"),
    ]
}

# -----------------------------
# APPLY RULE ENGINE
# -----------------------------
def rule_based_suggestions(text):
    matched_rules = []
    for pattern, sections in RULES.items():
        if re.search(pattern, text.lower()):
            matched_rules.extend(sections)
    return matched_rules[:3]  # Only top 3

# -----------------------------
# AI-BASED SIMILARITY SEARCH
# -----------------------------
def ai_based_suggestions(complaint, dataset, count=3):
    preprocessed = preprocess_text(complaint)
    comp_emb = model.encode(preprocessed)
    sec_emb = model.encode(dataset["Combo"].tolist())

    sims = util.pytorch_cos_sim(comp_emb, sec_emb)[0]
    top_indices = sims.topk(count).indices.tolist()

    return dataset.iloc[top_indices][[
        "Description", "Offense", "Punishment", "Cognizable", "Bailable", "Court"
    ]].to_dict(orient="records")

# -----------------------------
# FINAL COMBINED SUGGESTION
# -----------------------------
def suggest_sections(complaint, dataset):

    # 1️⃣ RULE ENGINE - HIGH PRIORITY
    rules = rule_based_suggestions(complaint)

    # If rule-engine gives strong matches → return them + AI combined
    if rules:
        ai_suggestions = ai_based_suggestions(complaint, dataset, count=3)

        # Convert rules to structured format
        rule_output = []
        for title, section in rules:
            rule_output.append({
                "Description": title,
                "Offense": section,
                "Punishment": "As per IPC section",
                "Cognizable": "Depends on case",
                "Bailable": "Depends on section",
                "Court": "As per section"
            })

        return rule_output[:2] + ai_suggestions[:1]

    # 2️⃣ IF RULES DID NOT MATCH → AI ONLY
    return ai_based_suggestions(complaint, dataset, count=3)
