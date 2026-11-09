# Annotation guidelines for gold v3 (entities and related pairs)

These guidelines define what counts as a gold entity and a gold related pair for
`eval/gold.jsonl` (v3). They are written for the annotators (humans or models) and for the
adjudicator. They apply to English, Dutch, German and French text from a personal document
store (emails, notes, project docs, chat logs).

The question we answer is: **which two entities are related?** The relation name is a secondary
*qualifier*. The schema is `schema/ontonotes.json` v2.2; the ontology is in
`docs/relation-ontology.md`.

## 1. Entities

Mark every mention of the 8 OntoNotes types. Mark the **name itself**: not titles, honorifics,
determiners, possessive `'s`, or trailing punctuation.

| type | what | examples |
|---|---|---|
| `PERSON` | people, real or fictional (names, not roles like "the CEO") | Satya Nadella; Dr. **Fatima Al-Sayed**; Sanne de Vries; Claire Dubois |
| `ORG` | companies, agencies, institutions, teams that are named organisations, parties, committees, bands, museums as institutions | Microsoft; Port Authority of Rotterdam; NATO; D66-fractie; Deutsche Bahn; LVMH |
| `GPE` | countries, states, cities, municipalities, and a city/state/country used as a political actor | Berlin; Texas; Nederland; Marseille |
| `LOC` | non-GPE places: mountains, rivers, seas, regions | Rocky Mountains; Rhein; la Seine |
| `FAC` | man-made facilities: buildings, bridges, airports, stations, named campuses | Heathrow Airport; Golden Gate Bridge; Kölner Dom; Tour Eiffel |
| `NORP` | nationalities, religious and political groups | German; katholieke; evangelische; Democratic |
| `DATE` | absolute or relative dates and periods | March 4, 2025; Tuesday; 2026-01-14; volgende week zaterdag; 1998 |
| `TIME` | times of day, and ISO datetimes | 3pm; 09:15; 15.00 uur; 2026-09-06T15:34:07Z |

Rules:

1. **Longest span, no nesting.** If a place name is part of an organisation's name, only the
   whole organisation is an entity: `Port Authority of Rotterdam` is one `ORG`; `Rotterdam` inside
   it is **not** a separate `GPE`. Same for `Cairo Institute for Peace Studies`, `Van Gogh Museum`,
   `Universität Heidelberg`. (Adjectival forms such as *Rotterdamse haven* are not GPE.)
2. **Product, model, library, file and version names are not entities** unless the name is also an
   organisation: `sqlite-vec`, `all-MiniLM-L6-v2`, `onnxruntime`, `v1.23.0`, `Boeing 747`,
   `src/ingest/fetch.rs`, `parse_config()` are not marked. `Cloudflare`, `Anthropic`, `SAP` are `ORG`.
3. **Metonymy follows context.** "Berlin signed an accord" is a `GPE` actor; "a meeting in Berlin"
   is a `GPE` place. Mark it `GPE` either way.
4. A mention can have **one type**. If unsure between `GPE` and `LOC` (a city), choose `GPE`.
5. Repeated names are separate mentions; each is an entity (use the occurrence index).
6. Pronouns and bare roles ("the CEO", "de premier", "le président") are not entities.

## 2. Pair categories

A **pair** is two distinct entity mentions in one text that the text relates. There are four
categories, defined by endpoint types. Annotate only entities you marked in section 1.

| category | directed | head | tail | qualifiers |
|---|---|---|---|---|
| `place` | yes | PERSON, ORG, FAC, GPE, LOC | GPE, LOC, FAC | `located_in`, `resident_of` |
| `per_per` | no | PERSON | PERSON | `family`, `business_associate`, `personal_relation` |
| `org_org` | no | ORG | ORG | `subsidiary_of`, `member_of`, `founder_of` |
| `per_org` | no | PERSON | ORG | `employee_of`, `member_of`, `founder_of` |

For each pair give the category and **all qualifiers that hold** (one or more).

### 2.1 The evidence rule

Annotate a pair **only when the text states or clearly entails it.** Use only the passage itself.

- Counts: explicit statements ("X works for Y", "X, the CEO of Y"), appositions and
  comma-/parenthesis-groupings ("Austin, Texas", "Veldhoven (NL)"), possessives and genitives
  ("Anna's brother Tom", "de Rotterdamse haven" is not a pair, but "the port of Rotterdam" is),
  and constructions with an obvious reading ("X, founder of Y").
- Does not count: world knowledge ("Tim Cook" and "Apple" in a text that does not connect them;
  Paris is in France but only "Paris, France" or "Paris in France" states it), co-occurrence in
  the same sentence, shared event, or "both attended".
- **Clear entailment is allowed**: "Dr. Weber leads Cloudflare's storage team" entails
  `employee_of(Weber, Cloudflare)`. "Mr Smith, who retired from the bank last year" does not tell
  us the bank's name, so no pair.
- When the passage is genuinely ambiguous, **leave the pair out**. Precision matters more than
  recall for borderline cases; the pooling pass recovers misses.

### 2.2 `place` (directed: head is inside / lives in; tail is the place)

- `located_in`: head ORG/FAC/GPE/LOC/PERSON-as-organiser is physically or administratively
  inside the tail. Examples: "Rijksmuseum in Amsterdam"; "Austin, Texas" (Austin -> Texas);
  "ASML, gevestigd in Veldhoven"; "Kölner Dom am Rhein" (Dom -> Rhein); "WHO ... Geneva".
  Organisation HQ/office/base: "based in", "headquartered in", "gevestigd in", "basé à", "mit
  Sitz in" count.
- `resident_of`: a PERSON lives, resides, is based or is from-and-lives in the tail:
  "Anna lives in Utrecht", "Pieter woont in Gent", "Claire habite à Lyon", "Jan wohnt in Köln".
  `resident_of` is for PERSON heads only. Citizenship/nationality adjectives are NORP, not pairs.
- **Direction**: the contained / living thing is the **head**, the place is the **tail**.
  Never the reverse.
  `Austin -> Texas`, `Rijksmuseum -> Amsterdam`, `Anna -> Utrecht`; not `Texas -> Austin`.
- Place chains: "Austin, Texas, USA" gives `Austin -> Texas` and `Texas -> USA`. Add
  `Austin -> USA` only if the text makes it explicit; otherwise leave it out.
- A person *at* a place for an event is **not** a pair (section 3, hard negatives).

### 2.3 `per_per` (undirected)

- `family`: kinship words (father, mother, son, daughter, wife, brother, vader, tante, Schwester,
  épouse, ...), including "his/her" + kin word where the kin is named.
- `business_associate`: colleagues, business partners, co-founders-of-the-same-venture as a
  stated partnership, "works with", "reports to", clients if stated as a working relationship.
- `personal_relation`: friends, partners (romantic), lasting personal ties.
- Annotate the pair once (no mirrored pair). Give every qualifier that holds, e.g. a couple who
  are also co-founders: `["family", "business_associate"]` only if the text says both.
- Two people who are only listed together, or who attended the same event, have **no** pair.

### 2.4 `per_org` (undirected, but recorded with the PERSON as head)

- `employee_of`: works for, is CEO / employee / manager / engineer at, leads, "joins".
- `member_of`: member of a party, committee, club, band, association.
- `founder_of`: founded, co-founded, "founder of".
- If someone founded and leads the company, both qualifiers hold only if the text states both
  (founded -> `founder_of`; "founded and runs" -> both).

### 2.5 `org_org` (undirected)

- `subsidiary_of`: subsidiary, division, brand of, "part of the group", "dochteronderneming",
  "Tochtergesellschaft", "filiale".
- `member_of`: an organisation is a member of another (a club in a federation, a company in a
  consortium).
- `founder_of`: one organisation founded the other.
- A contract, deal, accord, lawsuit, or partnership announcement between two ORGs
  ("Google and the European Commission settled the case") is **not** a pair.

## 3. Hard negatives (no pair)

1. **Co-mentioned people** with no stated relation: "Taylor Swift performed at the same festival
   where Dua Lipa also played." No `per_per` pair. Being on the same bill, at the same meeting,
   or in the same sentence is not a relation. If they are "friends", "colleagues", "siblings",
   it is.
2. **Code identifiers and file paths** in technical text: `ingest::fetch`, `crates/ingest/src/lib.rs`,
   `config.yaml`, `localdb`, `sqlite-vec`. These are not entities and cannot form pairs. If a
   technical sentence also names a real person and organisation ("maintained by Jane Park at
   Anthropic"), those are marked normally.
3. **Event locations are not residence or location facts.** "We met in Berlin", "attended the
   premiere in Los Angeles", "visited the United Nations headquarters in New York", "conference
   in Lyon". The `GPE` is an entity but there is **no** `resident_of` or `located_in` pair
   between the person and it. For an organisation, "held its meeting in Brussels" is also an event
   location. Only "based in", "headquartered in", "lives in", a postal address, or an apposition
   attached to the organisation counts. "Founded in Seattle in 1994" is an event (founding
   place) and is **not** enough for `located_in`.
4. **Negation and hypotheticals.** "Switzerland is not part of the European Union" has no pair.
   "Might join", "plans to acquire" are not pairs.
5. **Dealings and meetings.** Agreement, accord, signed with, met with, talked to, sued: not a
   relation by themselves.
6. **Nearby is not inside.** "Near", "close to", "across from", "bij", "à côté de" do not give
   `located_in` (the Golden Gate Bridge "connects San Francisco to Marin County" gives no
   `located_in` for the bridge; use only explicit containment).
7. **Endpoint type must fit.** A `DATE`, `TIME`, `NORP` is never an endpoint. A pair needs both
   endpoints to be entities from section 1 and to fit one of the four categories.
8. **Same entity twice.** No self-pairs. An alias of the same entity is not another entity.

## 4. Worked examples

Entities are in `[brackets]`; pairs are written `category | head -> tail | qualifiers`.

### English

1. `[Satya Nadella], the CEO of [Microsoft], met [German] Chancellor officials in [Berlin] on [Tuesday] at [3pm].`
   - `per_org | Satya Nadella -> Microsoft | employee_of`
   - No pair for Berlin: it is the meeting place (3.3).
2. `[Elon Musk], who leads [Tesla] and [SpaceX], announced a new factory near [Austin], [Texas] yesterday.`
   - `per_org | Elon Musk -> Tesla | employee_of`, `per_org | Elon Musk -> SpaceX | employee_of`
   - `place | Austin -> Texas | located_in` (apposition). No pair between Musk and Austin
     ("near", and a factory announcement).
3. `[Tim Cook] and [Jony Ive] worked closely together as business partners at [Apple].`
   - `per_per | Tim Cook <-> Jony Ive | business_associate`
   - `per_org | Tim Cook -> Apple | employee_of`, `per_org | Jony Ive -> Apple | employee_of`
4. `[Taylor Swift] performed at the same festival where [Dua Lipa] also played a set.` No pairs (3.1).
5. `The ingest crate calls fetch_all() in src/ingest/mod.rs; owner: [Jane Park] ([Anthropic]).`
   - `per_org | Jane Park -> Anthropic | employee_of`. Nothing for `fetch_all` or the path (3.2).
6. `[Anna Clarke] lives in [Leeds], but met [Tom Rey] in [Prague] last [May].`
   - `place | Anna Clarke -> Leeds | resident_of`. No pair for Prague (3.3). No pair between the
     two people (they only met).

### Dutch

7. `[Sanne de Vries] woont in [Utrecht] en werkt bij [Philips].`
   - `place | Sanne de Vries -> Utrecht | resident_of`
   - `per_org | Sanne de Vries -> Philips | employee_of`
8. `[ASML], gevestigd in [Veldhoven], werd opgericht door [Martin van den Brink] in [1984].`
   - `place | ASML -> Veldhoven | located_in`; `per_org | Martin van den Brink -> ASML | founder_of`
9. `[Frank de Boer] was op dezelfde persconferentie als [Ronald Koeman].` No pair (3.1).
10. `We spraken [Pieter] en zijn broer [Joris] gisteren in [Gent].` Single first names are people
    when they are names. `per_per | Pieter <-> Joris | family`. No pair for Gent (3.3).
11. `De Rotterdamse haven tekende een deal met [Amsterdam].` No entity "Rotterdam" (adjective); no pair (3.5).

### German

12. `Die [Siemens AG] wurde [1847] von [Werner von Siemens] in [München] gegründet.`
    - `per_org | Werner von Siemens -> Siemens AG | founder_of`. No `located_in`: "gegründet in München"
      is the founding place (3.3).
13. `[Lukas Weber] und [Anna Hoffmann] sind Geschäftspartner bei [SAP].`
    - `per_per | Lukas Weber <-> Anna Hoffmann | business_associate`; two `per_org ... | employee_of` pairs.
14. `Der [Kölner Dom] liegt am [Rhein], nicht weit von der [Hohenzollernbrücke] entfernt.`
    - `place | Kölner Dom -> Rhein | located_in`. "Nicht weit entfernt" gives no pair (3.6).
15. `Das Backend nutzt die Funktion load_index() aus config/db.toml; Ansprechpartnerin ist [Eva Braun] von der [Deutschen Bahn].`
    - `per_org | Eva Braun -> Deutschen Bahn | employee_of`. Nothing for identifiers or the path.

### French

16. `La [Tour Eiffel], construite par [Gustave Eiffel], se trouve à [Paris], non loin de la [Seine].`
    - `place | Tour Eiffel -> Paris | located_in`. No pair to the Seine ("non loin de", 3.6).
    - Gustave Eiffel -> Tour Eiffel is a person and a facility: not a category. No pair.
17. `[Claire Dubois] habite à [Lyon] et [Marc Lefevre] est son associé chez [Dassault].`
    - `place | Claire Dubois -> Lyon | resident_of`; `per_per | Claire Dubois <-> Marc Lefevre | business_associate`;
      `per_org | Marc Lefevre -> Dassault | employee_of`. Claire is not tied to Dassault.
18. `[Moët Hennessy] est une filiale du groupe [LVMH].`
    - `org_org | Moët Hennessy <-> LVMH | subsidiary_of`
19. `[Kylian Mbappé] et [Karim Benzema] ont grandi dans la même ville sans jamais se rencontrer.` No pair (3.1).
20. `On s'est vus à [Bruxelles] [mardi]; [Julie] est la fille de [Paul Martin].` Julie and Paul Martin:
    `per_per | Julie <-> Paul Martin | family`. Bruxelles is an event location (3.3).

## 5. Qualifier checklist

| situation | category | qualifiers |
|---|---|---|
| "CEO of", "works at", "leads" | per_org | employee_of |
| "founded", "co-founder" | per_org | founder_of |
| "member of the party/band/committee" | per_org | member_of |
| "subsidiary / brand / division of" | org_org | subsidiary_of |
| "joined the consortium" (org in org) | org_org | member_of |
| "lives in", "woont in", "wohnt in", "habite à" | place | resident_of |
| "based in", "in" + city after an org / facility | place | located_in |
| kinship terms | per_per | family |
| colleagues, business partners | per_per | business_associate |
| friends, partners | per_per | personal_relation |

## 6. Decisions recorded for the existing 84 sentences

- `gold-0002` ("Port Authority of Rotterdam"): the earlier `located_in` pair used `Rotterdam` as a
  tail while `Rotterdam` is inside the `ORG` span. Under rule 1.1 `Rotterdam` is not a separate
  entity, so **that pair is dropped**; the sentence has no pair. The `Amsterdam`/Port Authority
  accord is a dealing (3.5).
- `gold-0041` ("founded Amazon in Seattle in 1994; ... headquartered near the Space Needle"): the
  v2.1 pair `Amazon located_in Seattle` is **dropped**: "founded in Seattle" is a founding place
  (3.3) and "near the Space Needle" is nearness (3.6).
- Conventions fixed during adjudication of v3 (apply to all items):
  - a time expression includes its unit word (`10:00 Uhr`, `15.00 uur`); durations and frequencies
    ("jaren", "nightly", "de longue date") are `DATE`;
  - the German genitive `-s` stays inside the token (`Volkswagen-Konzerns`);
  - generic descriptors are not part of a name (`Start-up Vektorlabor` -> `Vektorlabor`); a library or
    museum named after a city (`médiathèque de Charleroi`) is one `ORG` (rule 1.1);
  - names of events and conferences (`Taalwerk Congres`) are not `ORG`;
  - "headquarters/HQ/office in X" and a postal address attached to an organisation give `located_in`;
    an anaphor that clearly refers to the organisation ("Their HQ is still in Ostrava") counts;
  - "sits on the board of X" is `member_of`;
  - kin words with a signed or named speaker (`Onkel Stefan` in a mail signed `Hannes`) give `family`
    with the signature name, consistent with `Meine Schwester Judith`;
  - the same fact may appear for coreferent mentions (`Pelgrim` / `Pelgrim Techniek`); the evidence
    mention is annotated by the annotators and a full-name alias may be added by pooling.
- Any other pair recorded in v2.1 is re-judged from scratch against these rules; v2.1 is frozen at
  `eval/annotation/gold-v2.1.jsonl` and was never shown to annotators.

## 7. Output format for annotators

Return results for every text, as strict JSONL (one JSON object per line, no prose, no code fence)
in the final message. Do **not** write files. Per text:

```
{"id": "<id>",
 "entities": [{"text": "<exact surface string>", "occurrence": 0, "type": "PERSON"}],
 "pairs": [{"category": "place",
            "head": {"text": "<surface>", "occurrence": 0},
            "tail": {"text": "<surface>", "occurrence": 0},
            "qualifiers": ["located_in"]}]}
```

- `text` must be copied exactly from the passage (same characters, case, accents).
- `occurrence` is the 0-based index of that exact string among all its occurrences in the passage
  (0 if it appears once). Substring matches count: if "Rotterdam" occurs inside another string,
  count it in the index.
- Pair endpoints must be strings you also listed under `entities`.
- Categories: `place`, `per_per`, `org_org`, `per_org`. For undirected categories, any order.
- Qualifiers must belong to the category (section 2).
- A text with nothing to annotate gets empty lists.

## 8. Process and provenance

Gold v3 is produced by blind double annotation (two independent model annotators, A1 and A2, who see
only the guidelines and the texts), adjudicated against these guidelines, and then completed by a
pooling pass: model predictions at a low threshold (>= 0.3) that the gold lacks are reviewed against
the guidelines and, if accepted, added with `source: "pool"`. All other pairs carry
`source: "annotator"`. Inter-annotator agreement and pooling statistics are in
`eval/annotation/README.md` and `eval/annotation/agreement.json`.
