# Relation ontology (schema v2)

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

## Design notes

- **Two passes of four.** The CoreML relation head is R4 and the bucket allows Q ≤ 8, so the
  relations run as passes `affiliation` and `social_place`, on top of the entity pass: three model
  calls per window. All engines run the same passes so they stay comparable. Measured prompt cost
  is 57–64 subword tokens per pass, within the 65-token budget `data/windows.jsonl` was sized for.
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
- **Optional extensions** (`relations_optional` in the schema, not run): `owner_of`
  (ORG-AFF.Ownership/Investor-Shareholder, P127 inverse), `alumnus_of` (ORG-AFF.Student-Alum,
  P69), `resident_of` (GEN-AFF.Citizen-Resident, P27/P551).
