"""
rank.py — Lightning-fast ranking. No model loading, no embeddings, no FAISS.
Just reads precomputed final_score from SQLite → outputs top-100 CSV.
Runtime: <10 seconds.

Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
import argparse
import csv
import sqlite3
from constants import DB_FILE


def run(output_file: str):
    conn = sqlite3.connect(DB_FILE)

    # Single query: sorted by score desc, tiebreak by candidate_id asc
    rows = conn.execute("""
        SELECT candidate_id, final_score, reasoning
        FROM candidates
        ORDER BY final_score DESC, candidate_id ASC
        LIMIT 100
    """).fetchall()
    conn.close()

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, (cid, score, reasoning) in enumerate(rows, start=1):
            writer.writerow([cid, rank, round(score, 4), reasoning])

    print(f"Done. Written {len(rows)} rows to {output_file}")
    print("Top-5 preview:")
    for rank, (cid, score, reasoning) in enumerate(rows[:5], start=1):
        print(f"  [{rank}] {cid} score={round(score,4)} | {reasoning[:90]}...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='candidates.jsonl')
    parser.add_argument('--out', default='submission.csv')
    args = parser.parse_args()
    run(args.out)
