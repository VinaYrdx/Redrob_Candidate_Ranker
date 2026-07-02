# Redrob Candidate Ranker — India.Runs Hackathon

Intelligent candidate ranking system for the Redrob AI × Hack2Skill **India.Runs** Track 1: Data & AI Challenge.

## What it does

Given 100,000 candidate profiles (`candidates.jsonl`) and a fixed Job Description (Senior AI Engineer, Redrob AI), produces a ranked top-100 shortlist with per-candidate reasoning. Goes beyond keyword matching — detects semantic fit, career quality signals, honeypot profiles, and behavioral engagement.

## Architecture

```
precompute.py         ← offline step (no time limit)
    ↓ candidates.jsonl
    ↓ MiniLM embeddings (all-MiniLM-L6-v2)
    ↓ feature engineering
    → candidates.db (SQLite) + faiss.index + features.npy + candidate_ids.pkl

rank.py               ← ranking step (≤5 min, CPU only, no network)
    ↓ loads precomputed artifacts
    ↓ encodes JD → cosine similarity via FAISS IndexFlatIP
    ↓ vectorized composite score
    ↓ O(N log K) min-heap for top-100
    → submission.csv
```

## Scoring Components

| Signal | Weight | Justification |
|--------|--------|---------------|
| Skill match (must-have + quality) | 30% | JD explicitly defines must-have skill clusters |
| Career quality (product co + AI/prod depth) | 25% | JD disqualifies consulting-lifers and pure research |
| Semantic similarity (MiniLM cosine) | 15% | Catches semantic fit beyond keyword presence |
| Years of experience (bell curve, peak 5–9) | 10% | JD target band |
| Location fit | 8% | Pune/Noida preferred, major Indian cities acceptable |
| Education | 7% | Tier + degree level + relevant field |
| Notice period | 5% | JD: ≤30 days ideal, buyable to 60 |

Final score = weighted_base × behavioral_multiplier × cv_speech_penalty × disqualifier_multiplier

### Disqualifier multipliers
- Consulting lifer (>85% career at TCS/Infosys/Wipro etc) → 0.25×
- Pure research (>70% career in research titles, no production) → 0.35×
- Job-hopper (avg tenure <16 months across 3+ companies) → 0.70×
- CV/Speech-only profile without NLP/IR signals → 0.25×

### Honeypot detection
Profiles with impossible characteristics (expert proficiency + 0 duration_months, claimed YOE >> sum of career durations, current role duration > time since start date) are zeroed out before the heap.

### Behavioral multiplier [0.5, 1.0]
Recency (last_active_date), open_to_work_flag, recruiter_response_rate, avg_response_time_hours, interview_completion_rate, offer_acceptance_rate, profile_completeness_score, github_activity_score.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce submission

```bash
# Step 1: Offline precomputation (~15-30 min, run once)
python precompute.py

# Step 2: Ranking step (≤5 min CPU, no network)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Step 3: Validate
python validate_submission.py submission.csv
```

## Run tests

```bash
pytest test_ranker.py -v
```

## Compute environment

- Python 3.11
- CPU only, 16GB RAM
- No GPU, no network during ranking step
- Runtime: precompute ~20 min | rank.py <5 min

## Sandbox

Live demo (≤100 candidates): [HuggingFace Space link here]

## Files

```
constants.py          # JD-derived constants, weights, skill sets
precompute.py         # Offline: features, embeddings, SQLite, FAISS
rank.py               # Online: fast ranking → CSV
app.py                # Streamlit sandbox
test_ranker.py        # pytest suite (honeypots, edge cases, tiebreaks)
requirements.txt
submission_metadata.yaml
validate_submission.py
```

## AI tools used
Claude (architecture discussion, code review), GitHub Copilot (autocomplete). No candidate data was fed to any LLM.
