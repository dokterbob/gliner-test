import json
from pathlib import Path

import pytest

from gliner_test import relations as R

SCHEMA = R.load_schema(Path(__file__).resolve().parent.parent / "schema" / "ontonotes.json")
CATS = SCHEMA["categories"]


def ep(text, start):
    return {"text": text, "start": start, "end": start + len(text)}


def ent(label, text, start, score=0.9):
    return {"label": label, "text": text, "start": start, "end": start + len(text), "score": score}


def rel(name, h, t, score):
    return {"name": name, "head": h, "tail": t, "score": score}


def typed(relations, entities):
    return R.snap_and_type(relations, entities)


def test_snap_largest_overlap_and_preserves_offsets():
    ents = [ent("ORG", "Port Authority of Rotterdam", 48, 0.9), ent("GPE", "Rotterdam", 66, 0.95)]
    r = [rel("located_in", ep("Port Authority", 48), ep("Rotterdam", 66), 0.8)]
    out = R.snap_and_type(r, ents)
    assert out[0]["head"]["type"] == "ORG"
    assert out[0]["tail"]["type"] == "GPE"  # overlap tie (9 chars each): higher entity score wins
    ents[1]["score"] = 0.9
    assert R.snap_and_type(r, ents)[0]["tail"]["type"] == "ORG"  # still tied: earlier entity wins
    # larger overlap beats higher score
    ents2 = [ent("GPE", "Rotterdam", 66, 0.99), ent("ORG", "Port Authority of Rotterdam", 48, 0.1)]
    wide = [rel("located_in", ep("Authority of Rotterdam", 53), ep("x", 0), 0.8)]
    assert R.snap_and_type(wide, ents2)[0]["head"]["type"] == "ORG"
    # offsets/text untouched
    assert out[0]["head"]["start"] == 48 and out[0]["head"]["end"] == 62
    assert out[0]["head"]["text"] == "Port Authority"


def test_snap_unmatched_is_none():
    out = R.snap_and_type([rel("family", ep("Bob", 0), ep("Alice", 10), 0.5)], [ent("PERSON", "Bob", 0)])
    assert out[0]["head"]["type"] == "PERSON"
    assert out[0]["tail"]["type"] is None
    assert R.categorize(out[0], CATS) is None


def test_directed_place_keeps_orientation_and_rejects_reverse():
    ents = [ent("PERSON", "Ann", 0), ent("GPE", "Paris", 10)]
    fwd = typed([rel("resident_of", ep("Ann", 0), ep("Paris", 10), 0.7)], ents)[0]
    cz = R.categorize(fwd, CATS)
    assert cz["category"] == "place" and cz["directed"] is True
    assert cz["head"]["text"] == "Ann"
    rev = typed([rel("resident_of", ep("Paris", 10), ep("Ann", 0), 0.7)], ents)[0]
    assert R.categorize(rev, CATS) is None  # GPE -> PERSON fits nothing


def test_directed_pairs_do_not_merge_across_orientation():
    ents = [ent("GPE", "Paris", 0), ent("GPE", "France", 10)]
    rs = typed([rel("located_in", ep("Paris", 0), ep("France", 10), 0.9),
                rel("located_in", ep("France", 10), ep("Paris", 0), 0.4)], ents)
    pairs = R.collapse_pairs(rs, CATS, "w")
    assert len(pairs) == 2
    assert {p["head"]["text"] for p in pairs} == {"Paris", "France"}


def test_per_org_canonical_order_person_first():
    ents = [ent("ORG", "Acme", 0), ent("PERSON", "Ann", 10)]
    r = typed([rel("employee_of", ep("Acme", 0), ep("Ann", 10), 0.8)], ents)[0]
    cz = R.categorize(r, CATS)
    assert cz["category"] == "per_org" and cz["directed"] is False
    assert cz["head"]["text"] == "Ann" and cz["tail"]["text"] == "Acme"


def test_per_per_canonical_order_by_start():
    ents = [ent("PERSON", "Ann", 20), ent("PERSON", "Bob", 0)]
    r = typed([rel("family", ep("Ann", 20), ep("Bob", 0), 0.8)], ents)[0]
    cz = R.categorize(r, CATS)
    assert cz["category"] == "per_per"
    assert cz["head"]["text"] == "Bob" and cz["tail"]["text"] == "Ann"


def test_collapse_max_vs_noisy_or_and_qualifiers():
    ents = [ent("PERSON", "Ann", 0), ent("PERSON", "Bob", 10)]
    rs = typed([
        rel("family", ep("Ann", 0), ep("Bob", 10), 0.6),
        rel("business_associate", ep("Bob", 10), ep("Ann", 0), 0.5),   # reversed: same undirected pair
        rel("employee_of", ep("Ann", 0), ep("Bob", 10), 0.9),            # not a per_per qualifier
    ], ents)
    (mx,) = R.collapse_pairs(rs, CATS, "w", mode="max")
    assert mx["score"] == pytest.approx(0.9)  # score_names=all includes employee_of
    assert [q["name"] for q in mx["qualifiers"]] == ["family", "business_associate"]
    (q_only,) = R.collapse_pairs(rs, CATS, "w", mode="max", score_names="qualifiers")
    assert q_only["score"] == pytest.approx(0.6)
    (no,) = R.collapse_pairs(rs, CATS, "w", mode="noisy-or", score_names="qualifiers")
    assert no["score"] == pytest.approx(1 - 0.4 * 0.5)
    assert [q["score"] for q in no["qualifiers"]] == sorted((q["score"] for q in no["qualifiers"]), reverse=True)


def test_collapse_keeps_best_score_per_name():
    ents = [ent("PERSON", "Ann", 0), ent("ORG", "Acme", 10)]
    rs = typed([rel("employee_of", ep("Ann", 0), ep("Acme", 10), 0.4),
                rel("employee_of", ep("Ann", 0), ep("Acme", 10), 0.7)], ents)
    (p,) = R.collapse_pairs(rs, CATS)
    assert p["score"] == pytest.approx(0.7)
    assert p["qualifiers"] == [{"name": "employee_of", "score": 0.7}]


def test_threshold_fallback_and_override():
    cats = [dict(c) for c in CATS]
    for c in cats:
        if c["id"] == "place":
            c["threshold"] = 0.8
    mk = lambda cat, s: {"category": cat, "score": s}
    pairs = [mk("place", 0.7), mk("place", 0.85), mk("per_per", 0.55), mk("per_per", 0.45)]
    kept = R.apply_thresholds(pairs, cats, 0.5)
    assert [(p["category"], p["score"]) for p in kept] == [("place", 0.85), ("per_per", 0.55)]
    kept = R.apply_thresholds(pairs, cats, 0.5, overrides={"place": 0.6, "per_per": 0.0})
    assert len(kept) == 4


def test_isotonic_monotone_and_pools():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 1, 0, 0, 1, 1, 0, 1, 1]
    t = R.fit_isotonic(scores, labels)
    assert all(a < b for a, b in zip(t["x"], t["x"][1:]))
    assert all(a <= b for a, b in zip(t["y"], t["y"][1:]))
    grid = [i / 100 for i in range(0, 101)]
    ys = [R.apply_isotonic(t, g) for g in grid]
    assert all(a <= b + 1e-12 for a, b in zip(ys, ys[1:]))
    assert 0.0 <= min(ys) and max(ys) <= 1.0
    # perfectly separable data stays 0 then 1
    t2 = R.fit_isotonic([0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
    assert R.apply_isotonic(t2, 0.1) == 0.0 and R.apply_isotonic(t2, 0.9) == 1.0


def test_calibration_fit_apply_roundtrip_json_and_raw_score():
    samples = {"place": [(0.5, 0), (0.6, 0), (0.7, 1), (0.9, 1)]}
    table = json.loads(json.dumps(R.fit(samples, engine="x")))
    pairs = [{"category": "place", "score": 0.8}, {"category": "per_per", "score": 0.8}]
    out = R.apply(pairs, table)
    assert out[0]["raw_score"] == 0.8 and 0.0 <= out[0]["score"] <= 1.0
    assert out[1]["score"] == 0.8 and "raw_score" not in out[1]  # category not fitted


def test_offsets_preserved_through_pipeline():
    text = "Ann works at Acme"
    ents = [ent("PERSON", "Ann", 0), ent("ORG", "Acme", 13)]
    rec = {"window_id": "w1", "entities": ents,
           "relations": [rel("employee_of", ep("Ann", 0), ep("Acme", 13), 0.9)]}
    (p,) = R.predictions_to_pairs(rec, CATS)
    for side in ("head", "tail"):
        e = p[side]
        assert text[e["start"]:e["end"]] == e["text"]
    assert p["head"]["type"] == "PERSON" and p["tail"]["type"] == "ORG"


def test_aggregate_document_merges_overlapping_windows():
    windows = {
        "c:0-30": {"chunk_id": "c", "char_start": 0},
        "c:10-40": {"chunk_id": "c", "char_start": 10},
        "c:60-90": {"chunk_id": "c", "char_start": 60},
    }

    def pair(wid, h, t, score, q="employee_of"):
        return {"window_id": wid, "category": "per_org", "directed": False,
                "head": {**h, "type": "PERSON"}, "tail": {**t, "type": "ORG"},
                "score": score, "qualifiers": [{"name": q, "score": score}]}

    pairs = [
        pair("c:0-30", ep("Ann", 10), ep("Acme", 20), 0.7),
        pair("c:10-40", ep("Ann", 0), ep("Acme", 10), 0.8),    # same mention pair at doc offsets 10/20
        pair("c:60-90", ep("ann", 2), ep("ACME", 12), 0.5),     # different mention, same normalised text
    ]
    (mx,) = R.aggregate_document(pairs, windows, "max")
    assert mx["score"] == pytest.approx(0.8)
    assert mx["evidence"] == 2 and mx["n_windows"] == 3
    assert mx["head"]["start"] == 10  # best mention's chunk-level offset
    (no,) = R.aggregate_document(pairs, windows, "noisy-or")
    assert no["score"] == pytest.approx(1 - 0.2 * 0.5)  # duplicates across windows not double counted


def test_aggregate_keeps_different_categories_apart():
    windows = {"w": {"chunk_id": "c", "char_start": 0}}
    a = {"window_id": "w", "category": "place", "directed": True, "head": {**ep("X", 0), "type": "ORG"},
         "tail": {**ep("Y", 5), "type": "GPE"}, "score": 0.9, "qualifiers": []}
    b = {**a, "category": "org_org", "directed": False, "tail": {**ep("Y", 5), "type": "ORG"}}
    assert len(R.aggregate_document([a, b], windows)) == 2


def test_detectors_score_names():
    cats = [dict(c, detectors=["related_to"]) if c["id"] == "per_per" else c for c in CATS]
    ents = [ent("PERSON", "Ann", 0), ent("PERSON", "Bob", 10)]
    rs = typed([
        rel("related_to", ep("Ann", 0), ep("Bob", 10), 0.4),
        rel("family", ep("Ann", 0), ep("Bob", 10), 0.9),
    ], ents)
    (p,) = R.collapse_pairs(rs, cats, "w", score_names="detectors")
    assert p["score"] == pytest.approx(0.4)                  # only the detector scores the pair
    assert [q["name"] for q in p["qualifiers"]] == ["family"]  # fine names stay qualifiers
    assert R.collapse_pairs(rs, cats, "w", score_names="all")[0]["score"] == pytest.approx(0.9)


def test_temperature_identity_monotone_and_fit():
    assert R.apply_temperature_score(0.7, 1.0) == pytest.approx(0.7, abs=1e-9)
    xs = [0.05, 0.2, 0.5, 0.8, 0.95]
    for t in (0.5, 2.0):
        ys = [R.apply_temperature_score(x, t) for x in xs]
        assert ys == sorted(ys)
    assert R.apply_temperature_score(0.9, 3.0) < 0.9 < R.apply_temperature_score(0.9, 0.5)
    # over-confident scores (0.95 but right only half the time) need T > 1; under-confident need T < 1
    over = [(0.95, 1), (0.95, 0)] * 20
    under = [(0.6, 1)] * 19 + [(0.6, 0)] + [(0.4, 0)] * 20
    assert R.fit_temperature(over) > 1.5
    assert R.fit_temperature(under) < 1.0
    assert R.fit_temperature([]) == 1.0


def test_temperature_table_apply_and_dispatch():
    samples = {"a": [(0.95, 1), (0.95, 0)] * 10, "b": [(0.9, 1)]}
    table = R.fit_temperature_table(samples, engine="x", min_samples=8)
    assert "a" in table["categories"] and "b" not in table["categories"]   # b falls back to the pooled T
    table = json.loads(json.dumps(table))
    pairs = [{"category": "a", "score": 0.95}, {"category": "b", "score": 0.9}]
    out = R.apply_calibration(pairs, table)
    assert out[0]["raw_score"] == 0.95 and out[0]["score"] < 0.95
    assert out[1]["score"] == pytest.approx(R.apply_temperature_score(0.9, table["global"]["T"]))


def test_suppress_overlapping_keeps_best_and_other_categories():
    ents = [ent("PERSON", "Robert Habeck", 0), ent("ORG", "Gruene-Fraktion im Bundestag", 20), ent("ORG", "Gruene-Fraktion", 20)]
    rs = typed([
        rel("member_of", ep("Robert Habeck", 0), ep("Gruene-Fraktion im Bundestag", 20), 0.9),
        rel("member_of", ep("Robert Habeck", 0), ep("Gruene-Fraktion", 20), 0.6),
    ], ents)
    pairs = R.collapse_pairs(rs, CATS, "w")
    assert len(pairs) == 2
    (kept,) = R.suppress_overlapping(pairs)
    assert kept["score"] == pytest.approx(0.9)
    # disjoint tails are not duplicates
    ents2 = [ent("PERSON", "A", 0), ent("ORG", "X", 10), ent("ORG", "Y", 20)]
    rs2 = typed([rel("employee_of", ep("A", 0), ep("X", 10), 0.9), rel("employee_of", ep("A", 0), ep("Y", 20), 0.8)], ents2)
    assert len(R.suppress_overlapping(R.collapse_pairs(rs2, CATS, "w"))) == 2
