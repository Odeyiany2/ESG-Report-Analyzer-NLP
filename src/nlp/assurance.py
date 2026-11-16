"""Assurance-readiness screen.

Question answered: would this report's claims survive third-party assurance?
The LLM only extracts claims and rates each evidence check; the scoring and
readiness levels are computed here in code so results are repeatable.

This is a text-level screen. A real assurance engagement tests underlying data
and controls, which this tool cannot see.
"""
import json
import re
from typing import Dict, List

from src.nlp.retriever import Retriever
from src.utils.logging import retriever_logger

# What an assurer needs to see before they can test a claim.
CHECKS = {
    "measurable": "Has a specific number, unit and reporting period (not just qualitative wording).",
    "methodology": "Names how it was calculated or measured (e.g. GHG Protocol, emission factors, definitions).",
    "boundary": "States scope: which entities, sites or operations, and the consolidation approach.",
    "comparatives": "Gives a prior-period figure or base year so movement can be verified.",
    "traceability": "Names the data source or system (meters, HR system, supplier data, third-party).",
    "controls": "Mentions internal review, controls or prior verification over the data.",
}
POINTS = {"pass": 1.0, "partial": 0.5, "fail": 0.0}

# Retrieval queries so claims are sampled across the whole report, not one topic.
SAMPLING_QUERIES = [
    "greenhouse gas emissions Scope 1 Scope 2 Scope 3 tonnes CO2e",
    "energy consumption renewable electricity targets",
    "water waste biodiversity environmental metrics",
    "workforce employees diversity health and safety injury rate training",
    "board oversight governance ethics anti-corruption compliance",
    "climate risks and opportunities financial impact scenario analysis",
    "assurance verification methodology boundary reporting period data sources",
]

PROMPT = """You are screening a sustainability report for assurance readiness.
Use ONLY the report excerpts below. Quote claims exactly as written.

Task:
1. Extract up to {max_claims} distinct factual or quantitative claims (numbers, targets achieved, policy assertions that could be tested). Skip slogans.
2. For each claim rate every check as "pass", "partial" or "fail", judging only from the excerpts:
{checks}
3. For each claim list "assurer_asks": 1-3 short, concrete evidence requests an assurer would make. Empty list if all checks pass.
4. Set "standard_ref" to the closest IFRS S1/S2 or SASB metric/section if the standards excerpts support it, else "".
5. Under "inconsistencies" list any case where the same metric appears with conflicting values in the excerpts (quote both). Empty list if none.

Return ONLY valid JSON, no prose, in this shape:
{{"claims":[{{"claim":"...","topic":"Environmental|Social|Governance","standard_ref":"","checks":{{{check_keys}}},"assurer_asks":["..."]}}],"inconsistencies":["..."]}}

Standards excerpts:
{standards}

Report excerpts:
{reports}
"""


def _gather_context(retriever: Retriever, per_query: int = 3, max_chunks: int = 24):
    reports, standards, seen_r, seen_s = [], [], set(), set()
    for q in SAMPLING_QUERIES:
        ctx = retriever.retrieve_context(q, k=per_query)
        for t in ctx["reports"]:
            if t not in seen_r:
                seen_r.add(t)
                reports.append(t)
        for t in ctx["standards"]:
            if t not in seen_s:
                seen_s.add(t)
                standards.append(t)
    return reports[:max_chunks], standards[:8]


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output.")
    return json.loads(text[start:end + 1])


def score_claim(claim: Dict) -> Dict:
    checks = {k: str(claim.get("checks", {}).get(k, "fail")).lower() for k in CHECKS}
    checks = {k: (v if v in POINTS else "fail") for k, v in checks.items()}
    pct = round(100 * sum(POINTS[v] for v in checks.values()) / len(CHECKS))
    # A claim with no number/period or no methodology cannot be assured regardless of the rest.
    hard_gap = checks["measurable"] == "fail" or checks["methodology"] == "fail"
    if pct >= 75 and not hard_gap:
        level = "Assurance-ready"
    elif pct >= 40 and checks["measurable"] != "fail":
        level = "Needs evidence"
    else:
        level = "Unlikely to survive"
    return {**claim, "checks": checks, "score": pct, "level": level}


def summarize(claims: List[Dict], inconsistencies: List[str]) -> Dict:
    scored = [score_claim(c) for c in claims]
    if not scored:
        return {"overall_score": 0, "overall_level": "Not enough testable claims found",
                "claims": [], "inconsistencies": inconsistencies, "top_gaps": [], "evidence_requests": []}
    overall = round(sum(c["score"] for c in scored) / len(scored))
    if inconsistencies:
        overall = max(0, overall - 5 * len(inconsistencies))
    level = ("Assurance-ready" if overall >= 75 else "Needs evidence" if overall >= 40 else "Unlikely to survive")

    gap_counts = {k: sum(1 for c in scored if c["checks"][k] != "pass") for k in CHECKS}
    top_gaps = [{"check": k, "description": CHECKS[k], "claims_affected": n}
                for k, n in sorted(gap_counts.items(), key=lambda kv: -kv[1]) if n][:5]
    asks, seen = [], set()
    for c in sorted(scored, key=lambda c: c["score"]):
        for a in c.get("assurer_asks", []):
            if a.lower() not in seen:
                seen.add(a.lower())
                asks.append({"claim": c["claim"], "request": a})
    return {"overall_score": overall, "overall_level": level, "claims": scored,
            "inconsistencies": inconsistencies, "top_gaps": top_gaps,
            "evidence_requests": asks[:25],
            "disclaimer": "Text-level screen only; not an assurance opinion."}


def run_assurance_readiness(retriever: Retriever, max_claims: int = 15) -> Dict:
    reports, standards = _gather_context(retriever)
    if not reports:
        raise ValueError("No report content available for assurance screening.")
    prompt = PROMPT.format(
        max_claims=max_claims,
        checks="\n".join(f'   - {k}: {v}' for k, v in CHECKS.items()),
        check_keys=",".join(f'"{k}":"pass|partial|fail"' for k in CHECKS),
        standards="\n---\n".join(standards),
        reports="\n---\n".join(reports),
    )
    raw = retriever.llm.chat.completions.create(
        model="openai/gpt-oss-120b:fireworks-ai",
        messages=[{"role": "system", "content": "You output only valid JSON."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4000,
    ).choices[0].message.content
    try:
        data = _parse_json(raw)
    except Exception as e:
        retriever_logger.error(f"Assurance JSON parse failed: {e}")
        raise
    return summarize(data.get("claims", []), data.get("inconsistencies", []))
