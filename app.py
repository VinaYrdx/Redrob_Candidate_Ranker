"""
app.py — Streamlit sandbox for Redrob Hackathon.
Accepts a JSONL file (≤100 candidates) → runs full ranking pipeline inline → outputs CSV.
No precomputed artifacts needed. Downloads MiniLM on first run (~80MB).
"""
import json
import csv
import heapq
import io
import streamlit as st
import numpy as np
from datetime import date, datetime
from sentence_transformers import SentenceTransformer

# ── Inline constants (mirrors constants.py) ───────────────────────────────────
TODAY = date.today()

JD_TEXT = """
Senior AI Engineer founding team product company. Production embeddings-based retrieval systems
sentence-transformers BGE E5 deployed real users at scale. Vector databases hybrid search
Pinecone Weaviate Qdrant Milvus FAISS Elasticsearch OpenSearch operational experience.
Strong Python production code quality. Ranking evaluation NDCG MRR MAP offline online A/B testing.
Shipping ranking recommendation search retrieval systems product companies applied ML AI NLP.
LLM fine-tuning LoRA QLoRA PEFT. Learning to rank XGBoost neural reranking.
5-9 years experience product company Pune Noida India.
"""

MUST_HAVE_SKILLS = {
    'embedding', 'embeddings', 'sentence-transformer', 'vector database', 'vector search',
    'semantic search', 'retrieval', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss',
    'elasticsearch', 'opensearch', 'dense retrieval', 'hybrid search', 'bm25',
    'ranking', 'ndcg', 'mrr', 'map', 'information retrieval', 'reranking',
    'recommendation', 'search', 'bge', 'e5', 'python', 'nlp',
}
NICE_TO_HAVE_SKILLS = {
    'lora', 'qlora', 'peft', 'fine-tuning', 'llm', 'rag', 'xgboost',
    'learning to rank', 'ltr', 'transformers', 'huggingface', 'pytorch',
    'tensorflow', 'a/b testing', 'open source',
}
CONSULTING_FIRMS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'hcl', 'tech mahindra', 'mphasis', 'hexaware', 'ltimindtree',
    'mindtree', 'ibm global services', 'deloitte', 'kpmg', 'ey ', 'ernst young', 'pwc',
}
CV_SPEECH_PRIMARY = {
    'computer vision', 'image classification', 'object detection', 'yolo',
    'opencv', 'image segmentation', 'speech recognition', 'asr', 'tts',
    'text to speech', 'robotics', 'slam', 'lidar',
}
NLP_IR_SIGNALS = {
    'nlp', 'retrieval', 'ranking', 'embedding', 'search',
    'information retrieval', 'recommendation', 'semantic', 'vector',
}
PURE_RESEARCH_TITLES = {
    'research scientist', 'research engineer', 'phd researcher', 'postdoc',
    'postdoctoral', 'research intern', 'research associate', 'ai researcher',
}
PREFERRED_LOCATIONS = {'pune', 'noida', 'delhi', 'gurugram', 'gurgaon', 'new delhi'}
ACCEPTABLE_LOCATIONS = {'mumbai', 'hyderabad', 'bangalore', 'bengaluru', 'chennai'}
PROFICIENCY_MAP = {'beginner': 0.25, 'intermediate': 0.5, 'advanced': 0.75, 'expert': 1.0}
DEGREE_MAP = {
    'phd': 1.0, 'ph.d': 1.0, 'm.tech': 0.9, 'mtech': 0.9, 'm.e.': 0.85,
    'm.s.': 0.85, 'ms ': 0.85, 'mba': 0.55, 'b.tech': 0.75, 'btech': 0.75,
    'b.e.': 0.75, 'b.s.': 0.7,
}
TIER_MAP = {'tier_1': 1.0, 'tier_2': 0.75, 'tier_3': 0.5, 'tier_4': 0.3, 'unknown': 0.4}
RELEVANT_FIELDS = {
    'computer science', 'ai', 'machine learning', 'data science',
    'statistics', 'mathematics', 'electrical', 'electronics',
}
WEIGHTS = {
    'skill': 0.30, 'career': 0.25, 'semantic': 0.15,
    'yoe': 0.10, 'location': 0.08, 'education': 0.07, 'notice': 0.05,
}


# ── Scoring functions (identical logic to precompute.py) ─────────────────────

def is_honeypot(c):
    score = 0
    skills = c.get('skills', [])
    career = c.get('career_history', [])
    profile = c.get('profile', {})
    signals = c.get('redrob_signals', {})
    for s in skills:
        if s.get('proficiency') == 'expert' and s.get('duration_months', 1) == 0:
            score += 3
        if s.get('proficiency') in ('advanced', 'expert') and \
           s.get('endorsements', 0) == 0 and s.get('duration_months', 0) == 0:
            score += 1
    total_months = sum(r.get('duration_months', 0) for r in career)
    claimed_yoe = profile.get('years_of_experience', 0)
    if claimed_yoe > 3 and total_months > 0 and total_months / 12 < claimed_yoe * 0.45:
        score += 2
    for r in career:
        if r.get('is_current'):
            try:
                start = datetime.strptime(r['start_date'], '%Y-%m-%d').date()
                actual_months = (TODAY - start).days / 30.44
                if r.get('duration_months', 0) > actual_months + 4:
                    score += 2
            except Exception:
                pass
    expert_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
    if expert_count > 8 and len(signals.get('skill_assessment_scores', {})) == 0:
        score += 1
    if signals.get('profile_completeness_score', 0) > 90 and len(career) < 2 and len(skills) < 4:
        score += 2
    return score >= 2


def compute_skill_score(c):
    skills = c.get('skills', [])
    assessments = c.get('redrob_signals', {}).get('skill_assessment_scores', {})
    must_total, nice_total, matched, matched_names = 0.0, 0.0, 0, []
    for s in skills:
        name_l = s.get('name', '').lower()
        prof = PROFICIENCY_MAP.get(s.get('proficiency', 'beginner'), 0.25)
        endorse = min(s.get('endorsements', 0) / 50.0, 1.0)
        duration = min(s.get('duration_months', 0) / 36.0, 1.0)
        boost = 0.0
        for k, v in assessments.items():
            if k.lower() in name_l or name_l in k.lower():
                boost = (v / 100.0) * 0.15
                break
        quality = 0.50 * prof + 0.25 * endorse + 0.25 * duration + boost
        if any(kw in name_l for kw in MUST_HAVE_SKILLS):
            must_total += quality
            matched += 1
            if s.get('proficiency') in ('advanced', 'expert'):
                matched_names.append(s['name'])
        elif any(kw in name_l for kw in NICE_TO_HAVE_SKILLS):
            nice_total += quality * 0.5
    return 0.70 * min(must_total / 4.0, 1.0) + 0.30 * min(nice_total / 3.0, 1.0), matched, matched_names[:3]


def compute_career_score(c):
    career = c.get('career_history', [])
    if not career:
        return 0.0, ['no_career_history']
    total_months = sum(r.get('duration_months', 0) for r in career) or 1
    consulting_m, research_m, product_m, ai_prod_m = 0, 0, 0, 0
    flags = []
    prod_kw = {'deployed', 'production', 'shipped', 'real users', 'at scale', 'serving', 'product', 'launched'}
    ai_kw = {'embedding', 'retrieval', 'ranking', 'recommendation', 'nlp', 'llm', 'vector', 'search', 'machine learning', 'fine-tun', 'semantic'}
    for r in career:
        comp_l = r.get('company', '').lower()
        title_l = r.get('title', '').lower()
        desc_l = r.get('description', '').lower()
        dur = r.get('duration_months', 0)
        is_cons = any(f in comp_l for f in CONSULTING_FIRMS)
        is_res = any(rt in title_l for rt in PURE_RESEARCH_TITLES)
        has_prod = any(kw in desc_l for kw in prod_kw)
        has_ai = any(kw in desc_l for kw in ai_kw)
        if is_cons: consulting_m += dur
        elif is_res: research_m += dur
        else: product_m += dur
        if has_ai and has_prod: ai_prod_m += dur
    if consulting_m / total_months > 0.85 and total_months > 24: flags.append('consulting_lifer')
    if research_m / total_months > 0.70: flags.append('pure_research')
    if len(career) >= 3 and total_months / len(career) < 16: flags.append('job_hopper')
    score = 0.45 * (product_m / total_months) + 0.55 * min(ai_prod_m / total_months, 1.0)
    return min(score, 1.0), flags


def compute_yoe_score(yoe):
    if 5 <= yoe <= 9: return 1.0
    elif 4 <= yoe < 5: return 0.85
    elif 9 < yoe <= 12: return 0.80
    elif 3 <= yoe < 4: return 0.60
    elif 12 < yoe <= 15: return 0.60
    elif yoe < 3: return 0.25
    else: return 0.40


def compute_location_score(c):
    p = c.get('profile', {})
    sig = c.get('redrob_signals', {})
    loc = (p.get('location', '') + ' ' + p.get('country', '')).lower()
    rel = sig.get('willing_to_relocate', False)
    if any(pl in loc for pl in PREFERRED_LOCATIONS): return 1.0
    if any(al in loc for al in ACCEPTABLE_LOCATIONS): return 0.85 if rel else 0.70
    if 'india' in loc: return 0.65 if rel else 0.45
    return 0.35 if rel else 0.15


def compute_notice_score(c):
    days = c.get('redrob_signals', {}).get('notice_period_days', 90)
    if days <= 15: return 1.0
    elif days <= 30: return 0.95
    elif days <= 60: return 0.75
    elif days <= 90: return 0.50
    else: return 0.25


def compute_education_score(c):
    edu_list = c.get('education', [])
    if not edu_list: return 0.30
    best = 0.0
    for e in edu_list:
        tier = TIER_MAP.get(e.get('tier', 'unknown'), 0.4)
        deg_l = e.get('degree', '').lower()
        deg_val = max((v for k, v in DEGREE_MAP.items() if k in deg_l), default=0.50)
        field_l = e.get('field_of_study', '').lower()
        fb = 0.10 if any(f in field_l for f in RELEVANT_FIELDS) else 0.0
        best = max(best, min(0.45 * tier + 0.45 * deg_val + fb, 1.0))
    return best


def compute_behavioral_mult(c):
    sig = c.get('redrob_signals', {})
    try:
        last = datetime.strptime(sig['last_active_date'], '%Y-%m-%d').date()
        recency = max(0.0, 1.0 - (TODAY - last).days / 180.0)
    except Exception:
        recency = 0.3
    otw = 1.0 if sig.get('open_to_work_flag', False) else 0.4
    resp_rate = sig.get('recruiter_response_rate', 0.3)
    resp_time = max(0.0, 1.0 - sig.get('avg_response_time_hours', 48) / 200.0)
    interview = sig.get('interview_completion_rate', 0.5)
    offer = sig.get('offer_acceptance_rate', -1)
    offer_s = offer if offer >= 0 else 0.5
    completeness = sig.get('profile_completeness_score', 50) / 100.0
    github = sig.get('github_activity_score', -1)
    github_s = (github / 100.0) if github >= 0 else 0.3
    raw = (0.22 * recency + 0.18 * otw + 0.20 * resp_rate + 0.10 * resp_time +
           0.10 * interview + 0.05 * offer_s + 0.10 * completeness + 0.05 * github_s)
    return 0.50 + 0.50 * raw


def compute_cv_penalty(c):
    skill_text = ' '.join(s.get('name', '').lower() for s in c.get('skills', []))
    career_text = ' '.join(r.get('description', '').lower() for r in c.get('career_history', []))
    all_text = skill_text + ' ' + career_text
    cv_count = sum(1 for kw in CV_SPEECH_PRIMARY if kw in all_text)
    nlp_count = sum(1 for kw in NLP_IR_SIGNALS if kw in all_text)
    if cv_count >= 3 and nlp_count <= 1: return 0.25
    if cv_count >= 2 and nlp_count == 0: return 0.35
    return 1.0


def disq_mult(flags):
    m = 1.0
    if 'consulting_lifer' in flags: m *= 0.25
    if 'pure_research' in flags: m *= 0.35
    if 'job_hopper' in flags: m *= 0.70
    if 'no_career_history' in flags: m *= 0.40
    return m


def build_embedding_text(c):
    p = c.get('profile', {})
    skills = c.get('skills', [])
    career = c.get('career_history', [])
    summary = p.get('summary', '')[:250]
    roles = ' '.join(
        f"{r.get('title', '')} at {r.get('company', '')}: {r.get('description', '')[:200]}"
        for r in career[:3]
    )
    top_skills = ' '.join(
        s.get('name', '') for s in sorted(
            skills,
            key=lambda s: {'expert': 4, 'advanced': 3, 'intermediate': 2, 'beginner': 1}
                          .get(s.get('proficiency', 'beginner'), 1),
            reverse=True
        )[:10]
    )
    return f"{p.get('current_title', '')}. {p.get('headline', '')}. {summary}. {roles}. Skills: {top_skills}"


def generate_reasoning(c, score, matched_names, must_matched, flags):
    p = c.get('profile', {})
    sig = c.get('redrob_signals', {})
    parts = [f"{p.get('current_title', 'Engineer')} | {p.get('years_of_experience', 0):.1f} yrs | {p.get('current_company', '')} | {p.get('location', '')}"]
    if must_matched > 0 and matched_names:
        parts.append(f"{must_matched} core AI/retrieval skills ({', '.join(matched_names)})")
    if sig.get('open_to_work_flag'): parts.append("actively looking")
    if sig.get('recruiter_response_rate', 0) >= 0.65:
        parts.append(f"responsive ({sig['recruiter_response_rate']:.0%})")
    if sig.get('notice_period_days', 90) <= 30: parts.append("immediate joiner")
    if flags: parts.append(f"flags: {', '.join(flags)}")
    return '; '.join(parts)[:300]


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("🎯 Redrob Candidate Ranker — Sandbox")
st.caption("Upload a JSONL file with ≤100 candidates. Ranks them for the Senior AI Engineer JD.")

# Replace the hardcoded JD_TEXT usage in the scoring loop with:
custom_jd = st.text_area(
    "Job Description (edit or replace with your own JD)",
    value=JD_TEXT.strip(),
    height=150,
)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

uploaded = st.file_uploader("Upload candidates.jsonl (≤100 candidates)", type=['jsonl', 'json'])

if uploaded:
    raw = uploaded.read().decode('utf-8')
    candidates = []
    for line in raw.strip().split('\n'):
        line = line.strip()
        if line:
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                st.warning(f"Skipped malformed line.")

    if len(candidates) > 100:
        st.warning(f"Loaded {len(candidates)} candidates — truncating to first 100 for sandbox.")
        candidates = candidates[:100]

    st.info(f"Loaded **{len(candidates)} candidates**. Running pipeline...")

    with st.spinner("Loading model..."):
        model = load_model()

    with st.spinner("Computing features + embeddings..."):
        scored = []
        texts = []
        for c in candidates:
            texts.append(build_embedding_text(c))

        # Embed all at once
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        jd_emb = model.encode([custom_jd], normalize_embeddings=True)[0]

        for i, c in enumerate(candidates):
            hp = is_honeypot(c)
            if hp:
                scored.append((c['candidate_id'], 0.0, 0, [], []))
                continue

            sk, must_matched, matched_names = compute_skill_score(c)
            cs, flags = compute_career_score(c)
            yoe = c.get('profile', {}).get('years_of_experience', 0)
            ys = compute_yoe_score(yoe)
            ls = compute_location_score(c)
            ns = compute_notice_score(c)
            es = compute_education_score(c)
            bm = compute_behavioral_mult(c)
            cv = compute_cv_penalty(c)
            dm = disq_mult(flags)

            # Cosine similarity (both normalized)
            sem = float(np.dot(embs[i], jd_emb))
            sem = max(0.0, sem)  # clip negatives

            base = (
                WEIGHTS['skill'] * sk + WEIGHTS['career'] * cs +
                WEIGHTS['semantic'] * sem + WEIGHTS['yoe'] * ys +
                WEIGHTS['location'] * ls + WEIGHTS['education'] * es +
                WEIGHTS['notice'] * ns
            )
            final = base * bm * cv * dm
            scored.append((c['candidate_id'], final, must_matched, matched_names, flags, c))

    # Sort by score desc, tiebreak by candidate_id asc
    scored.sort(key=lambda x: (-x[1], x[0]))
    top_n = min(len(scored), 100)

    # Build output rows
    output_rows = []
    for rank, row in enumerate(scored[:top_n], start=1):
        cid, score, must_matched, matched_names, flags, c = row
        reasoning = generate_reasoning(c, score, matched_names, must_matched, flags)
        output_rows.append({
            'candidate_id': cid,
            'rank': rank,
            'score': round(score, 4),
            'reasoning': reasoning,
        })

    st.success(f"✅ Ranked {top_n} candidates.")

    # Display top 20
    st.subheader("Top 20 Results")
    import pandas as pd
    df = pd.DataFrame(output_rows[:20])
    st.dataframe(df, use_container_width=True)

    # Score distribution
    st.subheader("Score Distribution")
    all_scores = [r['score'] for r in output_rows]
    st.bar_chart(all_scores)

    # Download CSV
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
    writer.writeheader()
    writer.writerows(output_rows)

    st.download_button(
        label="⬇️ Download ranked_output.csv",
        data=buf.getvalue(),
        file_name="ranked_output.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Upload a JSONL file to get started. Format: one candidate JSON object per line.")
    st.markdown("""
    **JD Summary (Senior AI Engineer — Redrob AI)**
    - 5–9 years experience at product companies
    - Production embeddings/retrieval/vector DB systems
    - Strong Python + ranking evaluation (NDCG, MRR, MAP)
    - Location: Pune/Noida preferred, major Indian cities acceptable
    """)
