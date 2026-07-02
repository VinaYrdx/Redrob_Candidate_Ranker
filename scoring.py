"""
scoring.py — Single source of truth for all scoring logic.
Imported by precompute.py and app.py. No duplication.
"""
from datetime import date, datetime
from constants import (
    TODAY, MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, CONSULTING_FIRMS,
    CV_SPEECH_PRIMARY, NLP_IR_SIGNALS, PURE_RESEARCH_TITLES,
    PREFERRED_LOCATIONS, ACCEPTABLE_LOCATIONS,
    PROFICIENCY_MAP, DEGREE_MAP, TIER_MAP, RELEVANT_FIELDS,
)


def is_honeypot(c: dict) -> bool:
    score = 0
    skills = c.get('skills', [])
    career = c.get('career_history', [])
    profile = c.get('profile', {})
    signals = c.get('redrob_signals', {})

    # 1. Expert proficiency + 0 duration — impossible
    for s in skills:
        if s.get('proficiency') == 'expert' and s.get('duration_months', 1) == 0:
            score += 3
        if s.get('proficiency') in ('advanced', 'expert') and \
           s.get('endorsements', 0) == 0 and s.get('duration_months', 0) == 0:
            score += 1

    # 2. Claimed YOE >> sum of career durations
    total_months = sum(r.get('duration_months', 0) for r in career)
    claimed_yoe = profile.get('years_of_experience', 0)
    if claimed_yoe > 3 and total_months > 0 and total_months / 12 < claimed_yoe * 0.45:
        score += 2

    # 3. Current role duration > time since start date
    for r in career:
        if r.get('is_current'):
            try:
                start = datetime.strptime(r['start_date'], '%Y-%m-%d').date()
                actual_months = (TODAY - start).days / 30.44
                if r.get('duration_months', 0) > actual_months + 4:
                    score += 2
            except Exception:
                pass

    # 4. Many expert skills, zero assessments
    expert_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
    if expert_count > 8 and len(signals.get('skill_assessment_scores', {})) == 0:
        score += 1

    # 5. High completeness but sparse actual data
    if signals.get('profile_completeness_score', 0) > 90 and len(career) < 2 and len(skills) < 4:
        score += 2

    # 6. NEW: Low YOE but suspiciously many expert skills
    if claimed_yoe < 3 and expert_count > 5:
        score += 2

    # 7. NEW: Stale profile — high completeness, not open to work, inactive >1 year
    try:
        last = datetime.strptime(signals['last_active_date'], '%Y-%m-%d').date()
        days_inactive = (TODAY - last).days
        if (signals.get('profile_completeness_score', 0) > 95
                and not signals.get('open_to_work_flag', True)
                and days_inactive > 365):
            score += 2
    except Exception:
        pass

    return score >= 2


def skill_score(c: dict):
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

    return (0.70 * min(must_total / 4.0, 1.0) + 0.30 * min(nice_total / 3.0, 1.0),
            matched, matched_names[:3])


def career_score(c: dict):
    career = c.get('career_history', [])
    if not career:
        return 0.0, ['no_career_history']

    total_months = sum(r.get('duration_months', 0) for r in career) or 1
    consulting_m, research_m, product_m, ai_prod_m = 0, 0, 0, 0
    flags = []
    prod_kw = {'deployed', 'production', 'shipped', 'real users', 'at scale',
               'serving', 'product', 'launched', 'live system'}
    ai_kw = {'embedding', 'retrieval', 'ranking', 'recommendation', 'nlp', 'llm',
              'vector', 'search', 'machine learning', 'fine-tun', 'semantic', 'rag'}

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

    if consulting_m / total_months > 0.85 and total_months > 24:
        flags.append('consulting_lifer')
    if research_m / total_months > 0.70:
        flags.append('pure_research')
    if len(career) >= 3 and total_months / len(career) < 16:
        flags.append('job_hopper')

    score = 0.45 * (product_m / total_months) + 0.55 * min(ai_prod_m / total_months, 1.0)
    return min(score, 1.0), flags


def yoe_score(yoe: float) -> float:
    if 5 <= yoe <= 9:    return 1.0
    elif 4 <= yoe < 5:   return 0.85
    elif 9 < yoe <= 12:  return 0.80
    elif 3 <= yoe < 4:   return 0.60
    elif 12 < yoe <= 15: return 0.60
    elif yoe < 3:        return 0.25
    else:                return 0.40


def location_score(c: dict) -> float:
    p = c.get('profile', {})
    sig = c.get('redrob_signals', {})
    loc = (p.get('location', '') + ' ' + p.get('country', '')).lower()
    rel = sig.get('willing_to_relocate', False)
    if any(pl in loc for pl in PREFERRED_LOCATIONS): return 1.0
    if any(al in loc for al in ACCEPTABLE_LOCATIONS): return 0.85 if rel else 0.70
    if 'india' in loc: return 0.65 if rel else 0.45
    return 0.35 if rel else 0.15


def notice_score(c: dict) -> float:
    days = c.get('redrob_signals', {}).get('notice_period_days', 90)
    if days <= 15:   return 1.0
    elif days <= 30: return 0.95
    elif days <= 60: return 0.75
    elif days <= 90: return 0.50
    else:            return 0.25


def education_score(c: dict) -> float:
    edu_list = c.get('education', [])
    if not edu_list:
        return 0.30
    best = 0.0
    for e in edu_list:
        tier = TIER_MAP.get(e.get('tier', 'unknown'), 0.4)
        deg_l = e.get('degree', '').lower()
        deg_val = max((v for k, v in DEGREE_MAP.items() if k in deg_l), default=0.50)
        field_l = e.get('field_of_study', '').lower()
        fb = 0.10 if any(f in field_l for f in RELEVANT_FIELDS) else 0.0
        best = max(best, min(0.45 * tier + 0.45 * deg_val + fb, 1.0))
    return best


def behavioral_multiplier(c: dict) -> float:
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


def cv_speech_penalty(c: dict) -> float:
    skill_text = ' '.join(s.get('name', '').lower() for s in c.get('skills', []))
    career_text = ' '.join(r.get('description', '').lower() for r in c.get('career_history', []))
    all_text = skill_text + ' ' + career_text
    cv_count = sum(1 for kw in CV_SPEECH_PRIMARY if kw in all_text)
    nlp_count = sum(1 for kw in NLP_IR_SIGNALS if kw in all_text)
    if cv_count >= 3 and nlp_count <= 1: return 0.25
    if cv_count >= 2 and nlp_count == 0: return 0.35
    return 1.0


def disqualifier_multiplier(flags: list) -> float:
    m = 1.0
    if 'consulting_lifer' in flags:  m *= 0.25
    if 'pure_research' in flags:     m *= 0.35
    if 'job_hopper' in flags:        m *= 0.70
    if 'no_career_history' in flags: m *= 0.40
    return m


def build_embedding_text(c: dict) -> str:
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


def generate_reasoning(c: dict, must_matched: int, matched_names: list, flags: list) -> str:
    p = c.get('profile', {})
    sig = c.get('redrob_signals', {})
    parts = [
        f"{p.get('current_title', 'Engineer')} | "
        f"{p.get('years_of_experience', 0):.1f} yrs | "
        f"{p.get('current_company', '')} | "
        f"{p.get('location', '')}"
    ]
    if must_matched > 0 and matched_names:
        parts.append(f"{must_matched} core AI/retrieval skills ({', '.join(matched_names)})")
    if sig.get('open_to_work_flag'): parts.append("actively looking")
    if sig.get('recruiter_response_rate', 0) >= 0.65:
        parts.append(f"responsive ({sig['recruiter_response_rate']:.0%})")
    if sig.get('notice_period_days', 90) <= 30: parts.append("immediate joiner")
    if flags: parts.append(f"flags: {', '.join(flags)}")
    return '; '.join(parts)[:300]
