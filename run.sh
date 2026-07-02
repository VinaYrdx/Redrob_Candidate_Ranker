#!/bin/bash
set -e
echo "=== Redrob Candidate Ranker ==="

if [ ! -f "candidates.db" ]; then
    echo "Step 1: Precomputing features + embeddings (~20 min)..."
    python precompute.py
else
    echo "Step 1: Artifacts found, skipping precompute. Delete candidates.db to rerun."
fi

echo "Step 2: Ranking (< 10 seconds)..."
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

echo "Step 3: Validating..."
python validate_submission.py submission.csv

echo "Done. Submit submission.csv to the portal."
