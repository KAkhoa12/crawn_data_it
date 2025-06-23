import re
import pandas as pd
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|#\S+|@\S+|[^\w\s]|[\*\•\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text 
def extract_primary_skills(text, primary_skills):
    # Xử lý trường hợp text không phải chuỗi
    text = clean_text(text)
    if not isinstance(text, str) or pd.isna(text):
        return []
    return [skill for skill in primary_skills if skill in text.lower()]

def extract_secondary_skills(text, secondary_skills):
    text = clean_text(text)
    if not isinstance(text, str) or pd.isna(text):
        return []
    return [skill for skill in secondary_skills if skill in text.lower()]

def extract_adjectives(text, nlp_en):
    text = clean_text(text)
    if not isinstance(text, str) or pd.isna(text):
        return []
    doc = nlp_en(text)
    return list(set([token.text for token in doc if token.pos_ == 'ADJ']))

def extract_adverbs(text, nlp_en):
    text = clean_text(text)
    if not isinstance(text, str) or pd.isna(text):
        return []
    doc = nlp_en(text)
    return list(set([token.text for token in doc if token.pos_ == 'ADV']))
