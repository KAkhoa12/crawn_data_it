import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
if not hasattr(XGBClassifier, "__sklearn_tags__"):
    XGBClassifier.__sklearn_tags__ = lambda self: {}
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from py_eureka_client import eureka_client
import asyncio
import pdfplumber
import docx
from typing import List
import io
import re

# Khởi tạo FastAPI app
app = FastAPI()

# Khởi tạo spaCy model
nlp_en = spacy.load("en_core_web_sm")  # Dùng cho trích xuất CV
nlp_vi = spacy.load("vi_core_news_lg")  # Dùng cho recommendation
matcher = PhraseMatcher(nlp_vi.vocab, attr="LOWER")

# Danh sách SKILLS từ cv.py
SKILLS = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "php", "ruby",
    "kotlin", "swift", "dart", "scala", "r", "rust", "perl", "elixir", "haskell", "objective-c",
    "html", "css", "scss", "sass", "less", "bootstrap", "tailwind css", "javascript", "jquery", "pug", "ejs",
    "react", "reactjs", "angular", "angularjs", "vue", "svelte", "ember.js", "backbone.js", "next.js", "nuxt.js",
    "node.js", "express.js", "nestjs", "spring", "spring boot", "django", "flask", "fastapi",
    "laravel", "symfony", "rails", "asp.net", "adonisjs", "phoenix", "hapi", "koa", ".net", "wordpress",
    "flutter", "react native", "swift", "kotlin", "java (android)", "android studio", "xamarin",
    "cordova", "ionic", "objective-c", "swift (ios)", "xcode", "unity", "unreal engine", "cocos2d",
    "mysql", "postgresql", "sqlite", "oracle", "sql server", "mariadb", "mongodb", "firebase",
    "dynamodb", "cassandra", "neo4j", "elasticsearch", "couchbase", "couchdb", "influxdb", "hbase", "redis", "memcached", "riak", "arangodb",
    "docker", "kubernetes", "terraform", "ansible", "jenkins", "github actions", "gitlab ci", "circleci", "helm",
    "aws", "azure", "gcp", "cloudflare", "vercel", "netlify", "heroku", "firebase hosting", "elastic beanstalk",
    "nginx", "apache", "caddy", "haproxy", "traefik", "cloudfront", "cloudflare cdn", "fastly", "akamai",
    "apache spark", "hadoop", "kafka", "airflow", "flink", "beam", "hive", "hbase", "sqoop",
    "dbt", "presto", "delta lake", "iceberg", "snowflake", "bigquery", "redshift", "athena", "databricks",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "plotly", "bokeh", "altair", "ggplot", "shiny", "dash", "streamlit", "flask", "fastapi", "django",
    "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "tensorflow", "keras", "pytorch",
    "xgboost", "lightgbm", "huggingface", "transformers", "langchain", "llamaindex", "mlflow",
    "openai", "spacy", "nltk", "opencv", "yolov5", "detectron2", "fastai", "pycaret", "auto-sklearn",
    "scrapy", "beautifulsoup", "selenium", "puppeteer", "requests", "httpx", "axios", "fetch",
    "oop", "design patterns", "mvc", "microservices", "monolith", "event-driven", "soa", "clean architecture",
    "hexagonal architecture", "tdd", "bdd", "ddd", "agile", "scrum", "kanban", "ci/cd", "devops", "git",
    "jwt", "oauth2", "keycloak", "saml", "https", "ssl/tls", "xss", "csrf", "sql injection",
    "owasp top 10", "hashing", "encryption", "rbac", "abac", "oauth2", "openid connect", "sso", "saml",
    "rest api", "graphql", "grpc", "soap", "openapi", "swagger", "postman", "axios", "feign", "retrofit",
    "websockets", "mqtt", "amqp", "kafka", "rabbitmq", "zeromq", "grpc", "protobuf", "thrift",
    "junit", "testng", "jest", "mocha", "chai", "pytest", "unittest", "cypress", "selenium",
    "playwright", "postman tests", "karate", "robot framework", "cucumber", "load testing", "jmeter",
    "kafka", "rabbitmq", "mqtt", "socket.io", "websocket", "zeromq", "pubsub", "eventbridge", "sns", "sqs",
    "kinesis", "firehose", "dynamodb streams", "kafka streams", "spark streaming", "flink", "beam",
    "aws lambda", "azure functions", "google cloud functions", "vercel", "netlify", "heroku", "cloud run",
    "cloud functions", "cloudflare workers", "fastly", "akamai", "lambda@edge", "cloudfront",
    "prometheus", "grafana", "elk stack", "elasticsearch", "logstash", "kibana", "datadog",
    "new relic", "graylog", "sentry",
    "git", "github", "gitlab", "bitbucket", "jira", "trello", "confluence", "figma", "draw.io",
    "lucidchart", "vscode", "intellij", "eclipse", "slack", "notion",
    "solidity", "ethereum", "web3.js", "ethers.js", "nft", "defi", "metamask", "truffle", "hardhat", "ipfs",
    "blockchain", "smart contract", "dapp", "dao", "ipfs", "zk-snarks", "zk-rollups", "plasma",
    "decentralized", "centralized", "distributed", "consensus", "proof of work", "proof of stake",
    "arduino", "raspberry pi", "stm32", "esp32", "iot core", "rtos", "freertos", "mqtt", "can bus"
}

# Hàm khởi tạo Eureka Client
async def init_eureka():
    # await eureka_client.init_async(
    #     # eureka_server="http://discovery-service:9999",
    #     eureka_server="http://discovery-service:9999",
    #     app_name="recommendation-service",
    #     instance_port=8000,
    #     instance_host="recommendation-service"
    # )
    await eureka_client.init_async(
        eureka_server="http://localhost:9999",
        app_name="recommendation-service",
        instance_port=8000,
        instance_host="localhost"
    )

# Khởi tạo Eureka Client khi ứng dụng khởi động
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(init_eureka())

# Kết nối MySQL
# def get_db_connection():
#     return mysql.connector.connect(
#         host="host.docker.internal",
#         user="root",
#         password="",
#         database="jobs"
#     )
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="jobs"
    )
# Từ điển chuẩn hóa kỹ năng (giữ nguyên từ code gốc)
skills_dictionary = {
    "java script": "javascript",
    "node js": "node.js",
    "react js": "react.js",
    "reactjs": "react.js",
    "vue js": "vue.js",
    "vuejs": "vue.js",
    "angular js": "angular.js",
    "typescript.js": "typescript",
    "py": "python",
    "c sharp": "c#",
    "c plus plus": "c++",
    "go lang": "go",
    "ruby on rails": "ruby",
    "php hypertext preprocessor": "php",
    "swift programming": "swift",
    "kotlin lang": "kotlin",
    "rust lang": "rust",
    "dart lang": "dart",
    "html5": "html",
    "css3": "css",
    "scala lang": "scala",
    "perl script": "perl",
    "r lang": "r",
    "shell script": "shell scripting",
    "bash script": "bash",
    "powershell script": "powershell",
    "sql language": "sql",
    "pl sql": "pl/sql",
    "t sql": "t-sql",
    "java": "java",
    "python": "python",
    "javascript": "javascript",
    "dotnet": ".net",
    "net": ".net",
    "aws": "aws",
    "gcp": "gcp",
    "agile": "agile",
    "objective c": "objective-c",
    "flutter framework": "flutter",
    "bridge software engineer": "brse",
    "business analyst": "brse",
    "requirements analysis": "brse",
    "stakeholder communication": "brse",
    "mobile app development": "mobile development",
    "android development": "mobile development",
    "ios development": "mobile development",
    "react native": "mobile development",
    "flutter mobile": "mobile development",
    "xcode": "mobile development",
    "android studio": "mobile development",
    "devops engineer": "devops",
    "site reliability engineer": "sre",
    "ci/cd pipeline": "ci/cd",
    "continuous integration": "continuous integration",
    "continuous deployment": "continuous deployment",
    "ansible automation": "ansible",
    "terraform infrastructure": "terraform",
    "jenkins": "jenkins",
    "terraform": "terraform",
    "ci/cd": "ci/cd",
    "docker container": "docker",
    "kubernetes": "kubernetes",
    "prometheus": "prometheus",
    "grafana": "grafana",
    "artificial intelligence": "artificial intelligence",
    "machine learning": "machine learning",
    "deep learning": "machine learning",
    "natural language processing": "artificial intelligence",
    "computer vision": "artificial intelligence",
    "tensorflow ai": "tensorflow",
    "pytorch ai": "pytorch",
    "scikit-learn ml": "scikit-learn",
    "security engineer": "cybersecurity",
    "cyber security": "cybersecurity",
    "network security": "cybersecurity",
    "penetration testing": "cybersecurity",
    "ethical hacking": "cybersecurity",
    "firewall management": "cybersecurity",
    "intrusion detection": "cybersecurity",
    "siem": "cybersecurity",
    "palo alto": "cybersecurity",
    "fortinet": "cybersecurity",
    "network engineer": "networking",
    "network security": "networking",
    "lan configuration": "networking",
    "wan configuration": "networking",
    "vpn setup": "networking",
    "cisco networking": "networking",
    "juniper networking": "networking",
    "wireshark analysis": "networking",
    "nmap scanning": "networking",
    "spring boot": "springboot",
    "spring framework": "spring",
    "net": ".net",
    "net core": ".net core",
    "asp net": "asp.net",
    "asp net core": "asp.net core",
    "django framework": "django",
    "flask framework": "flask",
    "express js": "express.js",
    "laravel framework": "laravel",
    "flutter framework": "flutter",
    "jquery library": "jquery",
    "bootstrap framework": "bootstrap",
    "tensorflow library": "tensorflow",
    "pytorch library": "pytorch",
    "keras library": "keras",
    "hadoop framework": "hadoop",
    "spark framework": "spark",
    "hibernate framework": "hibernate",
    "ember js": "ember.js",
    "backbone js": "backbone.js",
    "vue js": "vue.js",
    "next js": "nextjs",
    "nuxt js": "nuxt.js",
    "svelte framework": "svelte",
    "nest js": "nestjs",
    "graphql js": "graphql",
    "opencv": "opencv",
    "torch": "torch",
    "scikit-learn": "scikit-learn",
    "adobe illustrator": "illustrator",
    "adobe photoshop": "photoshop",
    "adobe indesign": "indesign",
    "adobe creative suite": "adobe creative suite",
    "figma tool": "figma",
    "sketch tool": "sketch",
    "zeplin tool": "zeplin",
    "adobe xd": "xd",
    "my sql": "mysql",
    "postgres": "postgresql",
    "ms sql": "sql server",
    "oracle db": "oracle",
    "mongo db": "mongodb",
    "redis cache": "redis",
    "dynamodb aws": "dynamodb",
    "maria db": "mariadb",
    "elastic search": "elasticsearch",
    "agile methodology": "agile",
    "scrum framework": "scrum",
    "kanban board": "kanban",
    "ms project": "ms project",
    "jira software": "jira",
    "trello board": "trello",
    "waterfall methodology": "waterfall",
    "agile": "project management",
    "scrum": "project management",
    "kanban": "project management",
    "jira": "project management",
    "trello": "project management",
    "tiếng anh": "communication",
    "confluence": "confluence",
    "technical leadership": "technical lead",
    "tech leadership": "tech lead",
    "system architecture": "system architect",
    "solutions architect": "solution architect",
    "amazon web services": "aws",
    "microsoft azure": "azure",
    "google cloud platform": "gcp",
    "aws lambda": "aws lambda",
    "azure functions": "azure functions",
    "lập trình java": "java",
    "lập trình python": "python",
    "lập trình javascript": "javascript",
    "thiết kế web": "web design",
    "thiết kế giao diện": "ui/ux design",
    "kiểm thử phần mềm": "testing",
    "phát triển phần mềm": "software development",
    "quản trị hệ thống": "system administration",
    "an ninh mạng": "cybersecurity",
    "trí tuệ nhân tạo": "artificial intelligence",
    "học máy": "machine learning",
    "dữ liệu lớn": "big data",
    "đám mây": "cloud computing",
    "mạng máy tính": "networking",
    "quản lý dự án": "project management",
    "kiến trúc phần mềm": "software architecture",
    "dẫn dắt kỹ thuật": "tech lead",
    "hỗ trợ IT": "it support",
    "cơ sở dữ liệu": "database administration",
    "phát triển ứng dụng di động": "mobile development",
    "kỹ sư devops": "devops",
    "kỹ sư an ninh": "cybersecurity",
    "kỹ sư mạng": "networking",
    "kỹ sư phần mềm cầu nối": "brse",
    "perl": "perl",
    "groovy": "groovy",
    "kotlin": "kotlin",
    "lua": "lua",
    "go": "go",
    "typescript": "typescript",
    "scala": "scala",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    ".net": ".net",
    "react native": "react native",
    "angular": "angular",
    "node.js": "node.js",
    "vue.js": "vue.js",
    "spring": "spring",
    "django": "django",
    "flask": "flask",
    "jquery": "jquery",
    "vue": "vue",
    "react": "react",
    "html": "html",
    "css": "css",
    "bootstrap": "bootstrap",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "jenkins": "jenkins",
    "ansible": "ansible",
    "terraform": "terraform",
    "git": "git",
    "svn": "svn",
    "mercurial": "mercurial",
    "junit": "junit",
    "selenium": "selenium",
    "testng": "testng",
    "pytest": "pytest",
    "mojolicious": "mojolicious",
    "catalyst": "catalyst",
    "grails": "grails",
    "ktor": "ktor",
    "spring": "spring",
    "lapis": "lapis",
    "orbit": "orbit",
    "gin": "gin",
    "echo": "echo",
    "fiber": "fiber",
    "beego": "beego",
    "nestjs": "nestjs",
    "play framework": "play framework",
    "akka": "akka",
    "ruby on rails": "ruby on rails",
    "sinatra": "sinatra",
    "laravel": "laravel",
    "symfony": "symfony",
    "magento": "magento",
    "codeigniter": "codeigniter",
    "zend framework": "zend framework",
    "vapor": "vapor",
    "perfect": "perfect",
    "asp.net": "asp.net",
    "asp.net core": "asp.net core",
    "blazor": "blazor",
    "expo": "expo",
    "ngrx": "ngrx",
    "angular material": "angular material",
    "express.js": "express.js",
    "koa": "koa",
    "hapi": "hapi",
    "nuxt.js": "nuxt.js",
    "vuex": "vuex",
    "vuetify": "vuetify",
    "spring boot": "spring boot",
    "spring security": "spring security",
    "django rest framework": "django rest framework",
    "flask-restful": "flask-restful",
    "flask-login": "flask-login",
    "docker-compose": "docker-compose",
    "helm": "helm",
    "blue ocean": "blue ocean",
    "jenkins pipeline": "jenkins pipeline",
    "selenide": "selenide",
    "fastapi": "fastapi",
    "flask": "flask",
    "rabbitmq": "rabbitmq",
    "celery": "celery",
    "apache kafka": "apache kafka",
    "kafka": "kafka",
    "mysql": "mysql",
    "postgresql": "postgresql",
    "mongodb": "mongodb",
    "oracle": "oracle",
    "sql server": "sql server",
    "redis": "redis",
    "sql": "sql",
    "nosql": "nosql",
    "pgadmin": "pgadmin",
    "sql developer": "sql developer",
    "datagrip": "datagrip",
    "tcp/ip": "tcp/ip",
    "http": "http",
    "https": "https",
    "ftp": "ftp",
    "router": "router",
    "switch": "switch",
    "firewall": "firewall",
    "owasp": "owasp",
    "burp suite": "burp suite",
    "nmap": "nmap",
    "wireshark": "wireshark",
    "penetration testing": "penetration testing",
    "ethical hacking": "ethical hacking",
    "kali linux": "penetration testing",
    "metasploit": "penetration testing",
    "owasp zap": "penetration testing",
    "ida pro": "malware analysis",
    "ghidra": "malware analysis",
    "radare2": "malware analysis",
    "autopsy": "digital forensics",
    "ftk imager": "digital forensics",
    "encase": "digital forensics",
    "aws": "aws",
    "azure": "azure",
    "google cloud platform": "google cloud platform",
    "ec2": "ec2",
    "s3": "s3",
    "lambda": "lambda",
    "azure functions": "azure functions",
    "hadoop": "hadoop",
    "spark": "spark",
    "hive": "hive",
    "pig": "pig",
    "linux": "linux",
    "windows server": "windows server",
    "system administration": "system administration",
    "it support": "it support",
    "machine learning": "machine learning",
    "deep learning": "deep learning",
    "figma": "figma",
    "adobe xd": "adobe xd",
    "sketch": "sketch"
}

skill_patterns = [nlp_vi(text) for text in skills_dictionary.keys()]
matcher.add("SKILL", None, *skill_patterns)

# Hàm trích xuất từ cv.py
def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_skills_cv(text: str) -> List[str]:
    text_lower = text.lower()
    return sorted(list({skill for skill in SKILLS if skill in text_lower}))

def extract_experience_blocks(text: str) -> List[dict]:
    lines = [re.sub(r"[^\x00-\x7F]+", "", line.strip()) for line in text.splitlines() if line.strip()]
    experiences = []
    time_pattern = re.compile(r"(\d{1,2}/\d{4})\s*[-–to]+\s*(\d{1,2}/\d{4}|now|present)", re.IGNORECASE)

    i = 0
    while i < len(lines) - 2:
        line = lines[i]
        match = time_pattern.search(line.lower())
        if match:
            time = f"{match.group(1)} - {match.group(2)}"
            company = lines[i + 1].strip()
            position = lines[i + 2].strip()
            desc_lines = []
            i += 3
            while i < len(lines) and (lines[i].startswith("-") or lines[i].startswith("•")):
                desc_lines.append(lines[i])
                i += 1

            experiences.append({
                "position": position,
                "company": company,
                "time": time,
                "desc": "\n".join(desc_lines)
            })
        else:
            i += 1
    return experiences

def extract_experience_flexible(text: str) -> List[dict]:
    if "WORK EXPERIENCE" in text:
        text = text.split("WORK EXPERIENCE", 1)[1]

    lines = text.splitlines()
    section_keywords = ["PROJECT", "PROJECTS", "EDUCATION", "CERTIFICATION", "SKILLS", "REFERENCES"]
    final_lines = []
    for line in lines:
        if any(line.strip().upper() == keyword for keyword in section_keywords):
            break
        final_lines.append(line)
    text = "\n".join(final_lines)

    lines = [re.sub(r"[^\x00-\x7F]+", "", line.strip()) for line in text.splitlines() if line.strip()]
    experiences = []
    time_pattern = re.compile(r"(\d{1,2}/\d{4})\s*[-–to]+\s*(\d{1,2}/\d{4}|present|now)", re.IGNORECASE)
    position_keywords = ["engineer", "developer", "intern", "manager", "tester", "freelancer", "support"]

    for i, line in enumerate(lines):
        lower = line.lower()
        if any(keyword in lower for keyword in position_keywords) and time_pattern.search(line):
            time_match = time_pattern.search(line)
            raw_pos = line.split('//')[0].strip() if '//' in line else line.split(time_match.group(1))[0].strip()

            prefixes_to_remove = [
                "skills", "communicate in english", "in english", "objective", "summary",
                "about", "profile", "of rest api", "project in", "project", "certification", "personal"
            ]
            for prefix in prefixes_to_remove:
                if raw_pos.lower().startswith(prefix):
                    raw_pos = raw_pos[len(prefix):].strip()

            position = raw_pos
            time = f"{time_match.group(1)} - {time_match.group(2)}"

            # Tìm dòng gần nhất là tên công ty (nâng cấp thuật toán)
            company = None
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j].strip()
                if (
                    next_line
                    and not next_line.startswith("•")
                    and not next_line.startswith("-")
                    and not time_pattern.search(next_line.lower())
                    and not any(keyword in next_line.lower() for keyword in ["project", "tech stack", "experience", "description"])
                ):
                    if (
                        any(kw in next_line.lower() for kw in ["company", "technology", "inc", "co."])
                        or sum(1 for w in next_line.split() if w[0].isupper()) >= 2
                    ):
                        company = next_line
                        break

            desc_lines = []
            j = i + 2
            while j < len(lines) and (lines[j].startswith("-") or lines[j].startswith("•")):
                desc_lines.append(lines[j])
                j += 1

            experiences.append({
                "position": position,
                "company": company,
                "time": time,
                "desc": "\n".join(desc_lines)
            })

    if not experiences:
        return extract_experience_blocks(text)
    return experiences

# Hàm chuẩn hóa kỹ năng từ file chính
def extract_skills(text):
    if not isinstance(text, str):
        return set()
    skills = set()
    for skill in text.split(","):
        skill = skill.strip().lower()
        standardized = skills_dictionary.get(skill, skill.title())
        skills.add(standardized)
    return skills

# Hàm tính Jaccard similarity
def jaccard_similarity(cv_skills, job_skills):
    if not cv_skills or not job_skills:
        return 0.0
    relevant_cv_skills = cv_skills.intersection(job_skills)
    if not relevant_cv_skills:
        return 0.0
    intersection = len(relevant_cv_skills)
    union = len(job_skills)
    return intersection / union

# Hàm load dữ liệu từ database
def load_data_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    query_job = """
    SELECT id, title, description, required_skills 
    FROM job
    """
    df_job = pd.read_sql(query_job, conn)
    df_job.columns = ['Job_ID', 'Title', 'Job Description', 'Required_Skills']

    query_cv = """
    SELECT id, skills 
    FROM cv
    """
    df_cv = pd.read_sql(query_cv, conn)
    df_cv.columns = ['CV_ID', 'Skills']

    conn.close()
    return df_job, df_cv

# Hàm huấn luyện mô hình
def train_models(df_job):
    df_job['combined_text'] = df_job['Required_Skills'].apply(lambda x: str(x))
    df_job['extracted_skills'] = df_job['Required_Skills'].apply(extract_skills)

    job_tfidf_vectorizer = TfidfVectorizer()
    job_tfidf_matrix = job_tfidf_vectorizer.fit_transform(df_job['combined_text'])

    return job_tfidf_vectorizer, job_tfidf_matrix, df_job

# Hàm gợi ý công việc
def recommend_jobs_for_cv_model(cv_text, cv_skills, job_df, job_tfidf_vectorizer, job_tfidf_matrix, alpha=0.5, beta=0.5, threshold=0.1):
    cv_vec = job_tfidf_vectorizer.transform([cv_text])
    cosine_sim = cosine_similarity(cv_vec, job_tfidf_matrix).flatten()
    jaccard_scores = job_df['extracted_skills'].apply(lambda js: jaccard_similarity(cv_skills, js)).values
    hybrid_score = alpha * cosine_sim + beta * jaccard_scores

    sorted_indices = np.argsort(hybrid_score)[::-1]

    recommendations = []
    for idx in sorted_indices:
        if hybrid_score[idx] < threshold:
            continue
        job_skills = job_df.loc[idx, 'extracted_skills']
        matched = job_skills.intersection(cv_skills) if isinstance(job_skills, set) else set()
        if matched:
            recommendations.append((job_df.loc[idx, 'Job_ID'], ', '.join(matched)))

    return recommendations

# Hàm xóa các match không còn phù hợp
def remove_outdated_matches(cv_id, cv_skills, df_job):
    conn = get_db_connection()
    cursor = conn.cursor()

    query_matches = """
    SELECT cv_id, job_id, matched_skill
    FROM matches
    WHERE cv_id = %s
    """
    cursor.execute(query_matches, (cv_id,))
    matches = cursor.fetchall()

    records_to_delete = []

    for match in matches:
        cv_id, job_id, matched_skill = match
        job_skills = df_job[df_job['Job_ID'] == job_id]['extracted_skills'].iloc[0]
        matched = job_skills.intersection(cv_skills) if isinstance(job_skills, set) else set()
        if not matched:
            records_to_delete.append((cv_id, job_id))

    if records_to_delete:
        delete_query = """
        DELETE FROM matches
        WHERE cv_id = %s AND job_id = %s
        """
        cursor.executemany(delete_query, records_to_delete)
        conn.commit()

    conn.close()

# Endpoint từ file chính
@app.get("/python/hello")
def read_hello():
    return {"message": "Xin chào, đây là API của tôi!"}

@app.get("/python/bye")
def read_bye():
    return {"message": "Bye Bye!!"}

@app.post("/python/recommend")
async def recommend_jobs():
    df_job, df_cv = load_data_from_db()
    job_tfidf_vectorizer, job_tfidf_matrix, df_job = train_models(df_job)

    df_cv['combined_text'] = df_cv['Skills'].apply(lambda x: str(x))
    df_cv['extracted_skills'] = df_cv['Skills'].apply(extract_skills)

    recommendation_rows = []
    for idx, row in df_cv.iterrows():
        cv_id = int(row['CV_ID'])
        cv_text = row['combined_text']
        cv_skills = row['extracted_skills']

        remove_outdated_matches(cv_id, cv_skills, df_job)

        rec = recommend_jobs_for_cv_model(cv_text, cv_skills, df_job, job_tfidf_vectorizer, job_tfidf_matrix, alpha=0.5, beta=0.5, threshold=0.1)

        for job_id, matched in rec:
            recommendation_rows.append({
                'cv_id': cv_id,
                'job_id': int(job_id),
                'matched_skill': matched,
                'time_matches': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 1
            })

    conn = get_db_connection()
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO matches (cv_id, job_id, matched_skill, time_matches, status)
    VALUES (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        matched_skill = VALUES(matched_skill),
        time_matches = VALUES(time_matches),
        status = VALUES(status)
    """
    for rec in recommendation_rows:
        cursor.execute(insert_query, (
            rec['cv_id'],
            rec['job_id'],
            rec['matched_skill'],
            rec['time_matches'],
            rec['status']
        ))

    conn.commit()
    conn.close()

    return {
        'status': 'success',
        'message': f'Inserted or updated {len(recommendation_rows)} recommendations into database'
    }

# Endpoint từ cv.py
@app.post("/python/extract")
async def extract_cv(file: UploadFile = File(...)):
    ext = file.filename.lower()
    content = await file.read()

    if ext.endswith(".pdf"):
        text = extract_text_from_pdf(io.BytesIO(content))
    elif ext.endswith(".docx"):
        text = extract_text_from_docx(io.BytesIO(content))
    else:
        return {"error": "Chỉ hỗ trợ PDF hoặc DOCX"}

    skills = extract_skills_cv(text)
    experience = extract_experience_flexible(text)


    return {
        "skills": skills,
        "experience": experience
    }

@app.get("/hello")
async def hello():
    return "aaaa"


@app.post("/python/recommend-by-seeker")
async def recommend_jobs_by_seeker(seeker_id: int):
    # Load dữ liệu với seeker_id cụ thể
    df_job, df_cv = load_data_by_seeker_id(seeker_id)

    if df_cv.empty:
        return {
            'status': 'error',
            'message': f'No CV found for seeker_id {seeker_id}'
        }

    job_tfidf_vectorizer, job_tfidf_matrix, df_job = train_models(df_job)

    df_cv['combined_text'] = df_cv['Skills'].apply(lambda x: str(x))
    df_cv['extracted_skills'] = df_cv['Skills'].apply(extract_skills)

    recommendation_rows = []
    for idx, row in df_cv.iterrows():
        cv_id = int(row['CV_ID'])
        cv_text = row['combined_text']
        cv_skills = row['extracted_skills']

        remove_outdated_matches(cv_id, cv_skills, df_job)

        rec = recommend_jobs_for_cv_model(cv_text, cv_skills, df_job, job_tfidf_vectorizer, job_tfidf_matrix, alpha=0.5,
                                          beta=0.5, threshold=0.1)

        for job_id, matched in rec:
            recommendation_rows.append({
                'cv_id': cv_id,
                'job_id': int(job_id),
                'matched_skill': matched,
                'time_matches': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 1
            })

    conn = get_db_connection()
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO matches (cv_id, job_id, matched_skill, time_matches, status)
    VALUES (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        matched_skill = VALUES(matched_skill),
        time_matches = VALUES(time_matches),
        status = VALUES(status)
    """

    for rec in recommendation_rows:
        cursor.execute(insert_query, (
            rec['cv_id'],
            rec['job_id'],
            rec['matched_skill'],
            rec['time_matches'],
            rec['status']
        ))

    conn.commit()
    conn.close()

    return {
        'status': 'success',
        'message': f'Inserted or updated {len(recommendation_rows)} recommendations for seeker_id {seeker_id}'
    }


# Hàm load dữ liệu mới dựa trên seeker_id
def load_data_by_seeker_id(seeker_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Load tất cả jobs
    query_job = """
    SELECT id, title, description, required_skills 
    FROM job
    """
    df_job = pd.read_sql(query_job, conn)
    df_job.columns = ['Job_ID', 'Title', 'Job Description', 'Required_Skills']

    # Load CV theo seeker_id
    query_cv = """
    SELECT id, seeker_id, skills 
    FROM cv
    WHERE seeker_id = %s
    """
    df_cv = pd.read_sql(query_cv, conn, params=(seeker_id,))
    df_cv.columns = ['CV_ID', 'Seeker_ID', 'Skills']

    conn.close()
    return df_job, df_cv

# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)