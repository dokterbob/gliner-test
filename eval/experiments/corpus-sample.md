# Corpus sample review (40 pairs, by eye)

Source: CoreML fp16-w8 (CPU) over `data/windows.jsonl` (3450 windows of localdb's own technical docs), final schema,
`--relation-threshold 0.0`; pipeline = max collapse over qualifier names, NMS, temperature calibration
(`eval/calibration/coreml-fp16-w8-cpu.json`), then document aggregation (`--aggregate`, max). 307 document-level pairs
exist with no threshold; 20 survive the calibrated threshold 0.5. Sample drawn by `eval/experiments/corpus_sample.py`
(stratified: per category up to 6 pairs at score >= 0.5, 4 in [0.2, 0.5), 4 below 0.2; ranks evenly spaced). Verdicts are
mine, judged on the pair as an entity relation of the kind the schema asks for ("correct" = a reader of the passage would
agree the two are related in that category; "unsure" = related in some loose sense, or right pair with the wrong kind of
relation; "wrong" otherwise).

Totals: correct 6, unsure 7, wrong 27 of 40. Strict precision 6/40 = 0.15, lenient (correct + unsure) 13/40 = 0.33.
Pairs at calibrated score >= 0.5 (the production threshold): 13 in the sample, correct 2, unsure 2, wrong 9.
By category (strict): see table. The corpus is technical prose: the entity layer tags code identifiers, tools and
product names as ORG/PERSON/GPE, and the pair layer inherits that, so precision here is far below the gold set's. Only
the genuine people (Alice/Bob/Carol in the sharing explainer) come out right.

| # | category | calibrated score | pair | verdict | reason |
|---|---|---|---|---|---|
| 1 | place | 0.61 | 'enum' <-> 'core' | unsure | code identifiers, not places; "lives in core" is a crate location |
| 2 | place | 0.53 | 'Dublin Core date' <-> 'Dublin' | wrong | "Dublin Core" is a metadata standard, not a place |
| 3 | place | 0.48 | 'entry loop' <-> 'floor' | wrong | code identifier / generic noun |
| 4 | place | 0.39 | 'schema' <-> 'store' | unsure | schema belongs to a store, but neither is a place |
| 5 | place | 0.33 | 'workspace' <-> 'workspace' | wrong | same noun twice |
| 6 | place | 0.31 | 'YAML front matter' <-> 'Dublin' | wrong | "Dublin Core" again |
| 7 | per_per | 0.99 | 'Alice' <-> 'Bob' | correct | people sharing libraries |
| 8 | per_per | 0.84 | 'Alice' <-> 'Bob' | correct | Alice shares with Bob |
| 9 | per_per | 0.52 | 'chunker' <-> 'embedder' | wrong | software components, not people |
| 10 | per_per | 0.46 | 'Carol' <-> 'Bob' | correct | Carol shares content that reaches Bob's node |
| 11 | per_per | 0.39 | 'Carol' <-> 'Bob' | correct | Bob's computer talks to Carol's |
| 12 | per_per | 0.35 | 'Carol' <-> 'Alice' | correct | Carol shared a library with Alice |
| 13 | per_per | 0.32 | 'Rationale' <-> 'chunker' | wrong | "Rationale" is not a person |
| 14 | per_per | 0.17 | 'Docker' <-> 'Docker Compose' | wrong | products, not people |
| 15 | per_per | 0.09 | 'archivists' <-> 'writers' | wrong | job groups merely co-listed |
| 16 | per_per | 0.05 | 'Khoj' <-> 'AnythingLLM' | wrong | products co-listed as alternatives |
| 17 | per_per | 0.03 | 'ingestor' <-> 'locator' | wrong | code identifiers |
| 18 | org_org | 0.97 | 'daemon' <-> 'store' | wrong | software components typed as ORG |
| 19 | org_org | 0.89 | 'HTTP' <-> 'fetch' | wrong | protocol/function names |
| 20 | org_org | 0.70 | 'CI' <-> 'cargo llvm-cov' | wrong | CI/tool names |
| 21 | org_org | 0.61 | 'aarch64-unknown-linux-gnu' <-> 'ONNX' | wrong | target triple and runtime |
| 22 | org_org | 0.54 | 'Turso' <-> 'localdb' | unsure | Turso (product) is relevant to localdb, but the qualifier is wrong |
| 23 | org_org | 0.53 | 'source add' <-> 'voyage' | wrong | CLI command and provider co-listed |
| 24 | org_org | 0.48 | 'localdb' <-> 'CLI' | correct | CLI is part of localdb (part-whole) |
| 25 | org_org | 0.39 | 'NATs' <-> 'UCAN' | wrong | co-listed protocols |
| 26 | org_org | 0.31 | 'CLI' <-> 'HTTP' | wrong | interfaces, not orgs |
| 27 | org_org | 0.24 | 'migrations' <-> 'Migrations' | wrong | same word, link text |
| 28 | org_org | 0.19 | 'localdb' <-> 'GPT4All' | unsure | competitor comparison, not membership/subsidiary |
| 29 | org_org | 0.04 | 'fetch' <-> 'embed' | unsure | crates sharing a limiter, typed as ORG |
| 30 | org_org | 0.02 | 'CLI' <-> 'cli' | wrong | same token, different case |
| 31 | org_org | 0.01 | 'chunk_prose' <-> 'chunk_prose' | wrong | same identifier |
| 32 | per_org | 0.60 | 'human' <-> 'source list' | wrong | "human" is an output mode |
| 33 | per_org | 0.55 | 'assistant' <-> 'library' | wrong | generic nouns |
| 34 | per_org | 0.41 | 'agent' <-> 'localdb' | unsure | agent knowledge vs localdb index, not employment/membership |
| 35 | per_org | 0.31 | 'contributor' <-> 'publisher' | wrong | schema fields |
| 36 | per_org | 0.28 | 'Claude' <-> 'markdown' | wrong | import source, not employment |
| 37 | per_org | 0.24 | 'user' <-> 'LibsqlDb::open' | wrong | generic noun + function |
| 38 | per_org | 0.19 | 'author' <-> 'Atom' | wrong | XML element names |
| 39 | per_org | 0.17 | 'Claude' <-> 'basic_memory' | wrong | product features |
| 40 | per_org | 0.14 | 'AnythingLLM' <-> 'localdb' | unsure | competitor mention, not an organisational relation |
