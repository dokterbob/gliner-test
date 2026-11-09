# Relation ontology (schema v2.2)

Entities follow OntoNotes 5.0 (8-type subset). OntoNotes defines no relations, so relations
follow a small **ACE 2005** subset (LDC2006T06; type table in
`references/ace2005-C12-2086.pdf`, Table 1). ACE 2005 is a good fit: its argument types (PER, ORG,
GPE, LOC, FAC) are the same entity classes we already extract, and its relation types are directed
(Arg1 → Arg2), with a few symmetric ones (PER-SOC). Each relation is also mapped to **Wikidata
properties** (`references/fewrel-pid2name.json`). That keeps output linkable to
FewRel/T-REx/DocRED-style data and Wikidata, and gives a path to more relations later.

Scope is deliberately small: people ↔ collective entities, people ↔ people, and where things are.

| id                   | prompt          | ACE 2005                                               | Wikidata                     | head → tail                |
| -------------------- | --------------- | ------------------------------------------------------ | ---------------------------- | -------------------------- |
| `employee_of`        | "works for"     | ORG-AFF.Employment                                     | P108 employer                | PERSON → ORG, GPE          |
| `member_of`          | "member of"     | ORG-AFF.Membership (+Sports-Affiliation)               | P463, P102, P54              | PERSON, ORG, GPE → ORG     |
| `founder_of`         | "founder of"    | ORG-AFF.Founder                                        | P112 founded by (inverse)    | PERSON, ORG → ORG          |
| `subsidiary_of`      | "subsidiary of" | PART-WHOLE.Subsidiary                                  | P749 parent organization     | ORG → ORG, GPE             |
| `family`             | "family"        | PER-SOC.Family (symmetric)                             | P26, P40, P22, P25, P3373, P1038 | PERSON ↔ PERSON        |
| `business_associate` | "colleague"     | PER-SOC.Business (symmetric)                           | P1327                        | PERSON ↔ PERSON            |
| `personal_relation`  | "friend"        | PER-SOC.Lasting-Personal (symmetric)                   | none                         | PERSON ↔ PERSON            |
| `located_in`         | "located in"    | PHYS.Located + GEN-AFF.Org-Location + PART-WHOLE.Geographical | P131, P159, P276      | ORG, FAC, GPE, LOC → GPE, LOC, FAC |
| `resident_of`        | "lives in"      | GEN-AFF.Citizen-Resident-Religion-Ethnicity            | P551 residence, P27 citizenship | PERSON → GPE, LOC       |

## Pair categories (v2.2)

The output is **pair-centric**: which two entities are related, with a confidence score. The
relation name is a secondary *qualifier*. `pair_categories` in the schema defines four categories
by endpoint type:

| category  | directed | head                        | tail          | qualifiers                                  |
| --------- | -------- | --------------------------- | ------------- | ------------------------------------------- |
| `place`   | yes      | PERSON, ORG, FAC, GPE, LOC  | GPE, LOC, FAC | located_in, resident_of                     |
| `per_per` | no       | PERSON                      | PERSON        | family, business_associate, personal_relation |
| `org_org` | no       | ORG                         | ORG           | subsidiary_of, member_of, founder_of        |
| `per_org` | no       | PERSON                      | ORG           | employee_of, member_of, founder_of          |

Each category has a `threshold`. It is `null` until tuned on the gold dev split (see
`docs/relation-accuracy.md`); `null` falls back to the global `threshold`.

## Design notes

- **Two passes of four (the v2.1 grouping, kept in v2.2).** The CoreML relation head is R4 and
  the bucket allows Q ≤ 8, so relations run in passes of at most four:
  - `affiliation`: employee_of, member_of, founder_of, subsidiary_of;
  - `social_place`: family, business_associate, personal_relation, located_in.

  With the entity pass, that is **three model calls per window**. All engines run the same
  passes, so they stay comparable.

  We also tried regrouping by pair category. The `per_org` / `org_place` / `per_per` passes,
  three relations each and four calls per window, lost on every dev metric: pair F1 .634 vs
  .672, AUPRC .498 vs .576, and +33% latency. See `docs/relation-accuracy.md`, experiment 01.

  Measured prompt cost with `windowing.py::measure_schema_cost` (model collator, empty text):

  | pass | tokens |
  |---|---|
  | entities | 59 |
  | `affiliation` | 60 |
  | `social_place` | 57 |
  | rejected v2.2 passes: `per_org` / `org_place` / `per_per` | 45 / 46 / 42 |

  Every pass is within the 65-token budget that `data/windows.jsonl` was sized for.
- **`name` vs `prompt`.** `name` is the canonical output id. `prompt` is the text sent to the
  model, and its wording is an experiment knob to tune on `eval/gold.jsonl`.
- **Typed endpoints.** `head`/`tail` hold OntoNotes types. GLiNER2.5's `JointIE` decoder
  (`references/gliner2-tutorial-15-joint-ie.md`) takes typed relation signatures and decodes a
  globally consistent graph. Independent `extract_relations` does not, so typing is then applied
  as a post-filter.
- **Dropped `occurred_on`.** Neither ACE 2005 nor Wikidata models "entity has date" as a binary
  relation; ACE anchors time through TIMEX2 and event arguments, Wikidata through qualifiers
  (P580 start time, P582 end time). DATE/TIME stay entities. Time-anchored facts (e.g.
  employment with a start date) are a better fit for GLiNER2.5 records/structures later.
- **Folded subtypes.** `located_in` merges three ACE subtypes that differ mainly by argument type.
  The specific Wikidata property (P131/P159/P276) can be derived from the endpoint types.
- **`resident_of` (v2.2) is a qualifier, not a prompt.** It is in the schema with prompt "lives in",
  but its `pass` is `unassigned`, so it is not sent to the model.
  - The `located_in` prompt already fires for "X lives in Y": 4 of the 6 gold resident pairs are
    found that way.
  - A separate "lives in" pass found no extra pair, added 3 false positives on dev and doubled
    latency (experiment 13a).
  - Instead, the `place` category's `qualifier_remap` relabels `located_in` as `resident_of` when
    the head is a PERSON. On test this lifts qualifier top-1 from 26/36 to 29/36 and leaves pair
    scores unchanged.
- **Optional extensions** (`relations_optional` in the schema, not run): `owner_of`
  (ORG-AFF.Ownership/Investor-Shareholder, P127 inverse) and `alumnus_of` (ORG-AFF.Student-Alum,
  P69).
