import pandas as pd
import spacy
import re
import ast
from tqdm import tqdm

# Load mô hình NLP tiếng Anh
nlp = spacy.load("en_core_web_sm")

# Đọc file CSV
df = pd.read_csv("itviec_jobs_undetected.csv")  # đổi tên file nếu cần

# Danh sách từ không có nghĩa và sai chính tả cần loại bỏ
MEANINGLESS_WORDS = {
    # Từ không có nghĩa
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'what', 'when', 'where', 'why', 'how', 'who', 'which', 'whose',
    'if', 'then', 'else', 'than', 'as', 'so', 'too', 'very', 'much', 'many', 'more', 'most',
    'some', 'any', 'all', 'each', 'every', 'no', 'not', 'only', 'just', 'also', 'even',
    'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    # Từ thường gặp không có ý nghĩa trong context tuyển dụng
    'job', 'work', 'company', 'team', 'role', 'position', 'candidate', 'applicant',
    'good', 'great', 'excellent', 'strong', 'solid', 'proven', 'successful',
    # Từ sai chính tả phổ biến
    'teh', 'adn', 'nad', 'hte', 'taht', 'thier', 'recieve', 'seperate', 'definately',
    'occured', 'begining', 'untill', 'writting', 'comming', 'runing', 'geting',
    # Từ quá ngắn (1-2 ký tự)
    'a', 'i', 'to', 'of', 'in', 'it', 'is', 'be', 'as', 'at', 'so', 'we', 'he', 'by', 'or', 'on', 'do', 'if', 'me', 'my', 'up', 'an', 'go', 'no', 'us', 'am', 'ok'
}

# Làm sạch văn bản (giữ ký tự tiếng Việt)
def clean_text(text):
    if pd.isnull(text): return ""
    if isinstance(text, list):
        text = ' '.join(map(str, text))
    text = str(text)
    text = text.lower()
    text = re.sub(r'\n|\r', ' ', text)
    # Giữ ký tự tiếng Việt: a-z, A-Z, 0-9, và ký tự có dấu tiếng Việt
    text = re.sub(r'[^a-zA-Z0-9\s\u00C0-\u1EF9]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Lọc từ có nghĩa
def filter_meaningful_words(words):
    """Lọc bỏ từ không có nghĩa và sai chính tả"""
    filtered = []
    for word in words:
        word_clean = word.lower().strip()
        # Bỏ từ quá ngắn (< 3 ký tự) hoặc quá dài (> 30 ký tự)
        if len(word_clean) < 3 or len(word_clean) > 30:
            continue
        # Bỏ từ trong danh sách meaningless
        if word_clean in MEANINGLESS_WORDS:
            continue
        # Bỏ từ chỉ chứa số
        if word_clean.isdigit():
            continue
        # Bỏ từ có quá nhiều ký tự lặp lại (như 'aaaa', 'xxxx')
        if len(set(word_clean)) <= 2 and len(word_clean) > 3:
            continue
        filtered.append(word)
    return filtered

# Trích 4 cụm từ bằng spaCy với bộ lọc
def extract_clusters(text):
    doc = nlp(text)
    primary, secondary, adjectives, adverbs = set(), set(), set(), set()
    for token in doc:
        # Bỏ qua token nếu là stop word hoặc không phải chữ cái
        if token.is_stop or not token.is_alpha:
            continue

        word = token.lemma_.lower()

        if token.pos_ in ['NOUN', 'PROPN']:
            primary.add(word)
        elif token.pos_ == 'VERB':
            secondary.add(word)
        elif token.pos_ == 'ADJ':
            adjectives.add(word)
        elif token.pos_ == 'ADV':
            adverbs.add(word)

    # Áp dụng bộ lọc cho tất cả các cụm
    primary = set(filter_meaningful_words(primary))
    secondary = set(filter_meaningful_words(secondary))
    adjectives = set(filter_meaningful_words(adjectives))
    adverbs = set(filter_meaningful_words(adverbs))

    return primary, secondary, adjectives, adverbs

# Trích xuất từ tiếng Việt bằng underthesea với cải thiện
def extract_vi_clusters(text):
    """Trích xuất tính từ và trạng từ tiếng Việt bằng underthesea"""
    from underthesea import pos_tag, word_tokenize
    import re

    if not text.strip():
        return set(), set(), set(), set()

    try:
        # Làm sạch text trước khi xử lý
        text_clean = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)  # Giữ ký tự tiếng Việt
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()

        if not text_clean:
            return set(), set(), set(), set()

        # Word tokenization trước để tránh cắt từ sai
        words = word_tokenize(text_clean)

        # POS tagging từng từ
        primary, secondary, adjectives, adverbs = set(), set(), set(), set()

        # Xử lý từng câu để tránh lỗi
        sentences = re.split(r'[.!?]+', text_clean)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            try:
                pos_tags = pos_tag(sentence)

                for word, tag in pos_tags:
                    word_clean = word.lower().strip()

                    # Kiểm tra từ hợp lệ
                    if len(word_clean) < 2:
                        continue

                    # Kiểm tra có ký tự tiếng Việt hoặc tiếng Anh
                    if not re.match(r'^[a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+$', word_clean):
                        continue

                    # Debug: In ra để kiểm tra
                    # print(f"Word: {word_clean}, Tag: {tag}")

                    # Phân loại theo POS tag của underthesea
                    if tag in ['N', 'Np', 'Ny', 'Nc']:  # Danh từ
                        primary.add(word_clean)
                    elif tag in ['V', 'Vb', 'Vy', 'Va']:  # Động từ
                        secondary.add(word_clean)
                    elif tag in ['A', 'Ab', 'Aa']:  # Tính từ
                        adjectives.add(word_clean)
                    elif tag in ['R', 'Rb', 'Ra']:  # Trạng từ
                        adverbs.add(word_clean)

            except Exception as e:
                print(f"Error processing sentence: {sentence[:50]}... - {e}")
                continue

        # Không áp dụng bộ lọc quá nghiêm ngặt cho tiếng Việt
        # Chỉ lọc từ quá ngắn
        primary = {w for w in primary if len(w) >= 2}
        secondary = {w for w in secondary if len(w) >= 2}
        adjectives = {w for w in adjectives if len(w) >= 2}
        adverbs = {w for w in adverbs if len(w) >= 2}

        return primary, secondary, adjectives, adverbs

    except Exception as e:
        print(f"Error in Vietnamese POS tagging: {e}")
        return set(), set(), set(), set()

# Chuyển đổi skills thành primary_skills (giữ nguyên không lọc)
def process_skills_to_primary(skills_raw):
    """Chuyển đổi skills thành primary_skills giống y chang"""
    if isinstance(skills_raw, str) and skills_raw.startswith("["):
        try:
            skills = ast.literal_eval(skills_raw)
        except:
            skills = []
    elif isinstance(skills_raw, list):
        skills = skills_raw
    else:
        skills = [str(skills_raw)] if skills_raw else []

    # Chỉ làm sạch cơ bản, không áp dụng bộ lọc nghiêm ngặt
    cleaned_skills = []
    for skill in skills:
        if isinstance(skill, str):
            skill_clean = skill.strip()
            if skill_clean:  # Chỉ bỏ skill rỗng
                cleaned_skills.append(skill_clean)

    return cleaned_skills

# Phân tách văn bản tiếng Anh và tiếng Việt
def separate_sentences_by_lang(text):
    """Phân tách câu tiếng Anh và tiếng Việt"""
    import re

    # Tách câu
    sentences = re.split(r'[.!?]+', text)

    en_sentences = []
    vi_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Đếm ký tự tiếng Việt (có dấu)
        vietnamese_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', sentence.lower()))
        total_chars = len(re.findall(r'[a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', sentence.lower()))

        if total_chars > 0:
            vietnamese_ratio = vietnamese_chars / total_chars
            if vietnamese_ratio > 0.1:  # Nếu > 10% ký tự tiếng Việt
                vi_sentences.append(sentence)
            else:
                en_sentences.append(sentence)

    return ' '.join(en_sentences), ' '.join(vi_sentences)

# Gộp kết quả từ tiếng Anh và tiếng Việt
def merge_clusters(en_clusters, vi_clusters):
    """Gộp kết quả từ tiếng Anh và tiếng Việt"""
    en_primary, en_secondary, en_adj, en_adv = en_clusters
    vi_primary, vi_secondary, vi_adj, vi_adv = vi_clusters

    return {
        "primary_skills": list(set(en_primary) | set(vi_primary)),
        "secondary_skills": list(set(en_secondary) | set(vi_secondary)),
        "adjectives": list(set(en_adj) | set(vi_adj)),
        "adverbs": list(set(en_adv) | set(vi_adv))
    }

# Hàm xử lý từng dòng với debug
def process_row(row):
    desc = clean_text(row.get("description", ""))
    reqs = clean_text(row.get("requirements", ""))
    full_text = f"{desc} {reqs}"

    # Debug: In ra text để kiểm tra
    job_title = row.get("title", "Unknown")
    print(f"\n🔍 Processing: {job_title}")
    print(f"📝 Full text sample: {full_text[:100]}...")

    # Phân tách tiếng Anh và tiếng Việt
    en_part, vi_part = separate_sentences_by_lang(full_text)

    print(f"🇺🇸 English part: {en_part[:50]}..." if en_part else "🇺🇸 English part: (empty)")
    print(f"🇻🇳 Vietnamese part: {vi_part[:50]}..." if vi_part else "🇻🇳 Vietnamese part: (empty)")

    # Trích xuất từ tiếng Anh
    en_clusters = extract_clusters(en_part) if en_part.strip() else (set(), set(), set(), set())

    # Trích xuất từ tiếng Việt
    vi_clusters = extract_vi_clusters(vi_part) if vi_part.strip() else (set(), set(), set(), set())

    print(f"🇺🇸 EN adjectives: {list(en_clusters[2])[:5]}")
    print(f"🇺🇸 EN adverbs: {list(en_clusters[3])[:5]}")
    print(f"🇻🇳 VI adjectives: {list(vi_clusters[2])[:5]}")
    print(f"🇻🇳 VI adverbs: {list(vi_clusters[3])[:5]}")

    # Gộp kết quả
    text_analysis = merge_clusters(en_clusters, vi_clusters)

    # primary_skills sẽ giống y chang cột skills
    primary_skills = process_skills_to_primary(row.get("skills", ""))

    return pd.Series({
        "primary_skills": primary_skills,
        "secondary_skills": text_analysis["secondary_skills"],
        "adjectives": text_analysis["adjectives"],
        "adverbs": text_analysis["adverbs"]
    })

# Áp dụng xử lý
tqdm.pandas()
processed = df.progress_apply(process_row, axis=1)

# Ghép kết quả vào DataFrame và xóa cột skills để không bị dư
df = pd.concat([df.drop(columns=["description", "requirements", "skills"], errors="ignore"), processed], axis=1)

# Test underthesea trước khi lưu
def test_underthesea():
    """Test underthesea với một câu tiếng Việt đơn giản"""
    from underthesea import pos_tag

    test_text = "Công việc này rất thú vị và hấp dẫn. Tôi làm việc chăm chỉ và nhanh chóng."
    print(f"\n🧪 Testing underthesea with: {test_text}")

    try:
        pos_result = pos_tag(test_text)
        print(f"POS result: {pos_result}")

        adjectives = []
        adverbs = []

        for word, tag in pos_result:
            if tag in ['A', 'Ab', 'Aa']:
                adjectives.append(f"{word}({tag})")
            elif tag in ['R', 'Rb', 'Ra']:
                adverbs.append(f"{word}({tag})")

        print(f"Adjectives found: {adjectives}")
        print(f"Adverbs found: {adverbs}")

    except Exception as e:
        print(f"Error in test: {e}")

# Chạy test
test_underthesea()

# Lưu kết quả
df.to_csv("job_postings_processed.csv", index=False)
print("✅ Đã xử lý xong và lưu vào job_postings_processed.csv")
