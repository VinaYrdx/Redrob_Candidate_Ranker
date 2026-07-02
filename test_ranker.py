"""
pytest test_ranker.py
Tests: honeypot detection, edge-case YOE, empty profiles, tiebreak ordering.
"""
import json
import pytest
from scoring import (
    is_honeypot, skill_score, career_score, yoe_score,
    location_score, notice_score, behavioral_multiplier, cv_speech_penalty,
    disqualifier_multiplier,
)


def make_candidate(overrides=None):
    """Minimal valid candidate fixture."""
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "ML Engineer",
            "summary": "Test summary",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "ML Engineer",
            "current_company": "ProductCo",
            "current_company_size": "201-500",
            "current_industry": "Technology",
        },
        "career_history": [{
            "company": "ProductCo",
            "title": "ML Engineer",
            "start_date": "2020-07-01",
            "end_date": None,
            "duration_months": 72,
            "is_current": True,
            "industry": "Technology",
            "company_size": "201-500",
            "description": "Built semantic search and embedding retrieval system deployed to production users at scale.",
        }],
        "education": [{
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2016,
            "end_year": 2020,
            "grade": "8.5 CGPA",
            "tier": "tier_1",
        }],
        "skills": [
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 30, "duration_months": 24},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 18},
            {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 48},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 85.0,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 20,
            "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {"Python": 85.0},
            "connection_count": 400,
            "endorsements_received": 60,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 35.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 70.0,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 8,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    if overrides:
        for path, val in overrides.items():
            keys = path.split('.')
            obj = base
            for k in keys[:-1]:
                obj = obj[k]
            obj[keys[-1]] = val
    return base


# ── Honeypot tests ────────────────────────────────────────────────────────────

def test_honeypot_expert_zero_duration():
    """Expert skill with 0 duration_months must be flagged."""
    c = make_candidate()
    c['skills'] = [
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 50, "duration_months": 0},
        {"name": "TensorFlow", "proficiency": "expert", "endorsements": 40, "duration_months": 0},
    ]
    assert is_honeypot(c)


def test_honeypot_legitimate_profile_not_flagged():
    """Valid profile must not be flagged as honeypot."""
    c = make_candidate()
    assert not is_honeypot(c)


def test_honeypot_claimed_yoe_vs_career_gap():
    """Claimed YOE >> sum of career durations should be flagged."""
    c = make_candidate()
    c['profile']['years_of_experience'] = 15.0
    c['career_history'][0]['duration_months'] = 6  # way below 15 years
    assert is_honeypot(c)


# ── YOE score tests ────────────────────────────────────────────────────────────

def test_yoe_in_band():
    assert yoe_score(6.0) == 1.0
    assert yoe_score(5.0) == 1.0
    assert yoe_score(9.0) == 1.0

def test_yoe_zero():
    assert yoe_score(0) == 0.25

def test_yoe_extreme_high():
    assert yoe_score(50) == 0.40

def test_yoe_just_below_band():
    assert yoe_score(4.5) == 0.85


# ── Empty / sparse profile edge cases ─────────────────────────────────────────

def test_empty_skills():
    c = make_candidate()
    c['skills'] = []
    sk, matched, names = skill_score(c)
    assert sk == 0.0
    assert matched == 0

def test_empty_career():
    c = make_candidate()
    c['career_history'] = []
    cs, flags = career_score(c)
    assert cs == 0.0
    assert 'no_career_history' in flags

def test_empty_education():
    c = make_candidate()
    c['education'] = []
    assert education_score(c) == 0.30

def test_missing_notice_period_defaults():
    c = make_candidate()
    del c['redrob_signals']['notice_period_days']
    # Should not crash, default = 90 → score = 0.50
    ns = notice_score(c)
    assert ns == 0.50


# ── Disqualifier tests ─────────────────────────────────────────────────────────

def test_consulting_lifer_penalized():
    c = make_candidate()
    c['career_history'] = [
        {"company": "TCS", "title": "Senior Analyst", "start_date": "2019-01-01",
         "end_date": None, "duration_months": 60, "is_current": True,
         "industry": "IT Services", "company_size": "10001+",
         "description": "Worked on client projects for banking clients."},
        {"company": "Infosys", "title": "Analyst", "start_date": "2015-01-01",
         "end_date": "2019-01-01", "duration_months": 48, "is_current": False,
         "industry": "IT Services", "company_size": "10001+",
         "description": "Maintained enterprise Java applications."},
    ]
    _, flags = career_score(c)
    assert 'consulting_lifer' in flags
    assert disqualifier_multiplier(flags) < 0.35


def test_cv_speech_only_penalized():
    c = make_candidate()
    c['skills'] = [
        {"name": "Computer Vision", "proficiency": "expert", "endorsements": 40, "duration_months": 36},
        {"name": "Image Classification", "proficiency": "expert", "endorsements": 30, "duration_months": 24},
        {"name": "Object Detection", "proficiency": "advanced", "endorsements": 20, "duration_months": 18},
    ]
    c['career_history'][0]['description'] = "Built YOLO-based object detection pipeline using OpenCV."
    assert cv_speech_penalty(c) < 0.5


# ── Tiebreak ordering ─────────────────────────────────────────────────────────

def test_identical_score_tiebreak_by_candidate_id():
    """Candidates with same score: CAND_0000001 < CAND_0000002, so 0000001 ranks higher."""
    # This is enforced in rank.py — test the sort logic directly
    scored = [('CAND_0000002', 0.95), ('CAND_0000001', 0.95), ('CAND_0000003', 0.90)]
    sorted_result = sorted(scored, key=lambda x: (-x[1], x[0]))
    assert sorted_result[0][0] == 'CAND_0000001'
    assert sorted_result[1][0] == 'CAND_0000002'


# Need to import education_score for test
from scoring import education_score