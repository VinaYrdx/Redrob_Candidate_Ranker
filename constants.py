from datetime import date

TODAY = date.today()

# --- JD Text for embedding ---
JD_TEXT = """
Senior AI Engineer founding team product company. Production embeddings-based retrieval systems
sentence-transformers BGE E5 deployed real users at scale. Vector databases hybrid search
Pinecone Weaviate Qdrant Milvus FAISS Elasticsearch OpenSearch operational experience.
Strong Python production code quality. Ranking evaluation NDCG MRR MAP offline online A/B testing
evaluation frameworks information retrieval. Shipping ranking recommendation search retrieval
systems product companies applied ML AI NLP. LLM fine-tuning LoRA QLoRA PEFT.
Learning to rank XGBoost neural reranking. 5-9 years experience product company Pune Noida India.
"""

# --- Skill sets from JD ---
MUST_HAVE_SKILLS = {
    'embedding', 'embeddings', 'sentence-transformer', 'sentence transformer',
    'vector database', 'vector search', 'semantic search', 'retrieval',
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'elasticsearch',
    'opensearch', 'dense retrieval', 'hybrid search', 'bm25',
    'ranking', 'ndcg', 'mrr', 'map', 'information retrieval',
    'reranking', 're-ranking', 'recommendation', 'search',
    'bge', 'e5', 'python', 'nlp', 'natural language processing',
}

NICE_TO_HAVE_SKILLS = {
    'lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning', 'llm', 'rag',
    'xgboost', 'learning to rank', 'ltr', 'distributed systems',
    'kafka', 'spark', 'kubernetes', 'transformers', 'huggingface',
    'pytorch', 'tensorflow', 'a/b testing', 'open source',
    'langchain', 'openai', 'cohere', 'ann', 'approximate nearest neighbor',
}

# JD explicitly calls these out as disqualifiers
CONSULTING_FIRMS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl', 'tech mahindra', 'mphasis',
    'hexaware', 'ltimindtree', 'mindtree', 'ibm global services',
    'deloitte', 'kpmg', 'ey ', 'ernst young', 'pwc',
}

CV_SPEECH_PRIMARY = {
    'computer vision', 'image classification', 'object detection', 'yolo',
    'opencv', 'image segmentation', 'speech recognition', 'asr',
    'tts', 'text to speech', 'robotics', 'slam', 'lidar', 'point cloud',
}

NLP_IR_SIGNALS = {
    'nlp', 'retrieval', 'ranking', 'embedding', 'search',
    'information retrieval', 'recommendation', 'text classification',
    'language model', 'semantic', 'vector',
}

PURE_RESEARCH_TITLES = {
    'research scientist', 'research engineer', 'phd researcher',
    'postdoc', 'postdoctoral', 'research intern', 'research associate',
    'ai researcher', 'ml researcher',
}

PREFERRED_LOCATIONS = {
    'pune', 'noida', 'delhi', 'gurugram', 'gurgaon', 'new delhi',
}
ACCEPTABLE_LOCATIONS = {
    'mumbai', 'hyderabad', 'bangalore', 'bengaluru', 'chennai',
}

PROFICIENCY_MAP = {'beginner': 0.25, 'intermediate': 0.5, 'advanced': 0.75, 'expert': 1.0}
DEGREE_MAP = {
    'phd': 1.0, 'ph.d': 1.0, 'm.tech': 0.9, 'mtech': 0.9,
    'm.e.': 0.85, 'm.s.': 0.85, 'ms ': 0.85, 'mba': 0.55,
    'b.tech': 0.75, 'btech': 0.75, 'b.e.': 0.75, 'b.s.': 0.7,
}
TIER_MAP = {'tier_1': 1.0, 'tier_2': 0.75, 'tier_3': 0.5, 'tier_4': 0.3, 'unknown': 0.4}
RELEVANT_FIELDS = {
    'computer science', 'ai', 'machine learning', 'data science',
    'statistics', 'mathematics', 'electrical', 'electronics', 'information technology',
}

# Composite weights — justified by JD signal hierarchy:
# Skills+Career carry most discriminative power per JD's explicit must-haves/disqualifiers
# Semantic adds context beyond keyword presence
# YOE/Location/Notice/Education are tie-breakers
WEIGHTS = {
    'skill':    0.30,
    'career':   0.25,
    'semantic': 0.15,
    'yoe':      0.10,
    'location': 0.08,
    'education':0.07,
    'notice':   0.05,
}

DB_FILE = 'candidates.db'
FAISS_INDEX_FILE = 'faiss.index'
FEATURES_FILE = 'features.npy'
IDS_FILE = 'candidate_ids.pkl'
CANDIDATES_FILE = 'candidates.jsonl'
