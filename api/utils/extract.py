import re
import pandas as pd
from collections import Counter
import spacy

adj_counter = Counter()
adv_counter = Counter()
def clean_text_for_matching(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    protected = []
    def _protect(match):
        protected.append(match.group(0))
        return f"__PROT_{len(protected)-1}__"
    
    # Bảo vệ các thuật ngữ IT như C#, C++, node.js trước mọi xử lý
    text = re.sub(r"\b\w+[#+.]\w*\b", _protect, text, flags=re.IGNORECASE)
    
    # Các bước xử lý văn bản
    text = text.lower()
    text = re.sub(r"(?<=\b\w)[-/](?=\w\b)", " ", text)
    text = re.sub(r"\s*&\s*", " ", text)
    text = re.sub(r"http\S+|@\s|\s@(?=\s)|\s#", " ", text)
    text = re.sub(r"[()]", " ", text)
    text = re.sub(r"(?<!\w)[.,!?;:–—](?!\w)", " ", text)
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"\b\w*['’‘]\w*\b", "", text)
    text = re.sub(r"\s*[-–—]{2,}\s*", " ", text)
    text = re.sub(r"[“”‘’«»„‟]", " ", text)
    text = re.sub(r"(?<=\w)[.,!:](?=\s|$)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"/", " ", text)
    text = re.sub(r"(?<=\w):(?=\w)", " ", text)
    text = re.sub(r"([^a-zA-Z0-9\s])\1{2,}", "", text)
    
    
    # Khôi phục các thuật ngữ được bảo vệ
    for i, word in enumerate(protected):
        text = text.replace(f"__prot_{i}__", word.lower())
    
    return text

def nomalize_skills(skills):
    return [clean_text_for_matching(skill).strip() for skill in skills]


def generate_ngrams(text, max_n=3):
    tokens = text.split()
    ngrams = []
    for n in range(1, max_n+1):
        ngrams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    return ngrams

def extract_skills(text, skills_list):
    if not isinstance(text.lower(), str) or pd.isna(text.lower()):
        return []

    ngrams = generate_ngrams(text.lower(), max_n=3)

    found = [skill for skill in skills_list if skill in ngrams]

    return list(dict.fromkeys(found))

def extract_adjectives(text, nlp):
    if not isinstance(text, str) or pd.isna(text):
        return []

    doc = nlp(text)
    adjs = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ == "ADJ"
        and len(token.lemma_) > 2            # loại từ ngắn như "as", "own"
        and token.is_alpha                   # bỏ từ chứa số, ký hiệu
        and not token.is_stop                # loại stop words mặc định của spaCy
    ]
    adj_counter.update(adjs)
    return list(set(adjs))

def extract_adverbs(text, nlp):
    if not isinstance(text, str) or pd.isna(text):
        return []

    doc = nlp(text)
    advs = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ == "ADV"
        and len(token.lemma_) > 2
        and token.is_alpha
        and not token.is_stop
    ]
    adv_counter.update(advs)
    return list(set(advs))
