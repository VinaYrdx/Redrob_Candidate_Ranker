"""
precompute.py — Offline step. Run once.
Produces: candidates.db (with final_score + reasoning), embeddings.npy, candidate_ids.pkl
"""
import json
import sqlite3
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from constants import (
    JD_TEXT, WEIGHTS,
    DB_FILE, IDS_FILE, CANDIDATES_FILE,
)
from scoring import (
    is_honeypot, skill_score, career_score, yoe_score,
    location_score, notice_score, education_score,
    behavioral_multiplier, cv_speech_penalty, disqualifier_multiplier,
    build_embedding_text, generate_reasoning,
)

EMBEDDINGS_FILE = 'embeddings.npy'


def main():
    print("Loading candidates...")
    candidates = []
    with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"  {len(candidates)} candidates loaded.")

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("DROP TABLE IF EXISTS candidates")
    conn.execute("""
        CREATE TABLE candidates (
            candidate_id  TEXT PRIMARY KEY,
            data          TEXT NOT NULL,
            is_honeypot   INTEGER NOT NULL,
            disq_flags    TEXT NOT NULL,
            must_matched  INTEGER,
            matched_names TEXT,
            final_score   REAL,
            reasoning     TEXT
        )
    """)

    print("Computing features...")
    all_ids = []
    all_texts = []
    rows_buffer = []

    for i, c in enumerate(candidates):
        if i % 10000 == 0:
            print(f"  {i}/{len(candidates)}")

        cid = c['candidate_id']
        hp = is_honeypot(c)
        sk, mmatched, mnames = skill_score(c)
        cs, flags = career_score(c)
        yoe = c.get('profile', {}).get('years_of_experience', 0)
        ys = yoe_score(yoe)
        ls = location_score(c)
        ns = notice_score(c)
        es = education_score(c)
        bm = behavioral_multiplier(c)
        cv = cv_speech_penalty(c)
        dm = disqualifier_multiplier(flags)

        base_without_sem = (
            WEIGHTS['skill']     * sk +
            WEIGHTS['career']    * cs +
            WEIGHTS['yoe']       * ys +
            WEIGHTS['location']  * ls +
            WEIGHTS['education'] * es +
            WEIGHTS['notice']    * ns
        )

        rows_buffer.append({
            'cid': cid, 'data': json.dumps(c), 'hp': int(hp),
            'flags': flags, 'mmatched': mmatched, 'mnames': mnames,
            'base': base_without_sem, 'bm': bm, 'cv': cv, 'dm': dm, 'c': c,
        })
        all_ids.append(cid)
        all_texts.append(build_embedding_text(c))

    print("  Features done.")

    print("Computing embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_embeddings = []
    for i in range(0, len(all_texts), 1024):
        embs = model.encode(all_texts[i:i+1024], normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs)
        if i % 20000 == 0:
            print(f"  Embedded {i}/{len(all_texts)}")

    emb_matrix = np.vstack(all_embeddings).astype('float32')
    print(f"  Embedding matrix: {emb_matrix.shape}")

    print("Computing semantic scores (vectorized)...")
    jd_emb = model.encode([JD_TEXT], normalize_embeddings=True).astype('float32')[0]
    sem_scores = np.clip(np.dot(emb_matrix, jd_emb), 0, None)
    s_min, s_max = sem_scores.min(), sem_scores.max()
    if s_max > s_min:
        sem_scores = (sem_scores - s_min) / (s_max - s_min)

    print("Writing final scores to DB...")
    insert_data = []
    for i, row in enumerate(rows_buffer):
        final = 0.0 if row['hp'] else (
            (row['base'] + WEIGHTS['semantic'] * float(sem_scores[i]))
            * row['bm'] * row['cv'] * row['dm']
        )
        reasoning = generate_reasoning(row['c'], row['mmatched'], row['mnames'], row['flags'])
        insert_data.append((
            row['cid'], row['data'], row['hp'], json.dumps(row['flags']),
            row['mmatched'], json.dumps(row['mnames']), round(final, 6), reasoning
        ))
    conn.executemany("INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?,?,?)", insert_data)

    conn.commit()
    conn.close()

    np.save(EMBEDDINGS_FILE, emb_matrix)
    with open(IDS_FILE, 'wb') as f:
        pickle.dump(all_ids, f)

    print(f"Done. Artifacts: {DB_FILE}, {EMBEDDINGS_FILE}, {IDS_FILE}")


if __name__ == '__main__':
    main()
