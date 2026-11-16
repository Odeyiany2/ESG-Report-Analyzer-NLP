import os
os.environ.setdefault("HUGGINGFACE_API_KEY", "test")

from src.nlp.assurance import score_claim, summarize

GOOD = {"claim": "Scope 1 emissions were 4,210 tCO2e in FY2024, down 12% from FY2023.",
        "checks": {"measurable": "pass", "methodology": "pass", "boundary": "pass",
                   "comparatives": "pass", "traceability": "partial", "controls": "pass"},
        "assurer_asks": []}
VAGUE = {"claim": "We are committed to reducing our footprint.",
         "checks": {"measurable": "fail", "methodology": "fail", "boundary": "fail",
                    "comparatives": "fail", "traceability": "fail", "controls": "fail"},
         "assurer_asks": ["Provide quantified footprint and method."]}
NO_METHOD = {"claim": "Emissions fell 10%.",
             "checks": {"measurable": "pass", "methodology": "fail", "boundary": "pass",
                        "comparatives": "pass", "traceability": "pass", "controls": "pass"}}


def test_levels():
    assert score_claim(GOOD)["level"] == "Assurance-ready"
    assert score_claim(VAGUE)["level"] == "Unlikely to survive"
    # high raw score but no methodology must not be called assurance-ready
    assert score_claim(NO_METHOD)["level"] != "Assurance-ready"


def test_bad_check_values_default_to_fail():
    c = score_claim({"claim": "x", "checks": {"measurable": "maybe"}})
    assert c["checks"]["measurable"] == "fail"


def test_summary_and_inconsistency_penalty():
    base = summarize([GOOD, VAGUE], [])
    penal = summarize([GOOD, VAGUE], ["Scope 1: 4,210 vs 4,512"])
    assert penal["overall_score"] == base["overall_score"] - 5
    assert base["evidence_requests"] and base["top_gaps"]


def test_empty():
    assert summarize([], [])["claims"] == []
