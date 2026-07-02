FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY constants.py scoring.py precompute.py rank.py validate_submission.py run.sh ./
RUN chmod +x run.sh

# Mount candidates.jsonl at runtime
CMD ["bash", "run.sh"]
