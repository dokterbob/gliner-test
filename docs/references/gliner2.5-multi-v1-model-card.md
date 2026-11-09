---
library_name: transformers
license: apache-2.0
language:
- multilingual
- en
tags:
- gliner2
- Text classification
- Named Entity Recognition
- Relation Extraction
- Intent classification
- Sentiment Analysis
- Topic classification
- Structured extraction
- Json extraction
- information-extraction
- boundary-extraction
---
<div align="center">
  <a href="https://agent.fastino.ai" target="_blank" rel="noopener noreferrer">
    <img src="GLiNER-2.5-Multi-HF-Banner.png" alt="Fastino Agent - Fine-tune GLiNER with a single prompt" width="100%"/>
  </a>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;">
  <a href="https://agent.fastino.ai" target="_blank" rel="noreferrer" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/Finetune-GLiNER2.5-EA4335" alt="Fine-tune and Deploy GLiNER2.5 with Fastino Agent" style="vertical-align:middle;">
  </a>
  <a href="https://arxiv.org/abs/2507.18546" target="_blank" rel="noreferrer" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/arXiv-2507.18546-b31b1b.svg?logo=arxiv" alt="arXiv Paper" style="vertical-align:middle;">
  </a>
  <a href="https://github.com/fastino-ai/GLiNER2" target="_blank" rel="noreferrer" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/GitHub-GLiNER2-black?logo=github" alt="GitHub" style="vertical-align:middle;">
  </a>
  <a href="https://x.com/fastinoAI" target="_blank" rel="noreferrer" style="text-decoration:none;">
    <img src="https://img.shields.io/twitter/follow/:fastinoAI" alt="Follow @fastinoAI" style="vertical-align:middle;">
  </a>
</div>

# GLiNER2.5 Multi: Unified Schema-Based Information Extraction

> *Extract entities, classify text, parse structured records, score span attributes, and extract relations — all in one boundary architecture.*

GLiNER2.5 Multi is the multilingual boundary checkpoint. It is built on mDeBERTa-v3-base and is the default choice when you need entities, classification, records, and relations in one model across languages. Load it with `AutoExtractor`: the checkpoint's `architecture` field selects `BoundaryExtractor` automatically.

Fine-tune via [Fastino](https://agent.fastino.ai). Join discussions on [Reddit](https://www.reddit.com/r/GLiNER/).

## ✨ Why GLiNER2.5?

- **🎯 One model, many tasks**: entities, classification, structured records, relations, and span attributes in a single schema
- **📐 Boundary architecture**: sparse start/end pairing instead of a fixed span-width grid — any span length that fits in the encoded window
- **🔗 Constrained decoding**: `Classifier` for cross-task label constraints, `JointIE` for typed entity–relation graphs
- **💻 Local inference**: CPU, CUDA, or MPS through `gliner2[local]` — no external API required

## GLiNER2.5 family

| Model | Parameters | Encoder | Language | Use case |
|-------|------------|---------|----------|----------|
| [`fastino/gliner2.5-small-v1`](https://huggingface.co/fastino/gliner2.5-small-v1) | 74M | DeBERTa-v3-xsmall | English | Fast CPU extraction / classification |
| [`fastino/gliner2.5-base-v1`](https://huggingface.co/fastino/gliner2.5-base-v1) | 194M | DeBERTa-v3-base | English | Default English multi-task checkpoint |
| [`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1) | 287M | mDeBERTa-v3-base | Multilingual | Default multilingual multi-task checkpoint |

This card is for **`fastino/gliner2.5-multi-v1`**. All three checkpoints share the same public API.

## Installation

```bash
pip install "gliner2[local]"
```

Python 3.10 or newer is required. The `[local]` extra pulls in PyTorch so you can load Hub checkpoints.

## Load the model

Always use `AutoExtractor` for GLiNER2.5. `GLiNER2.from_pretrained(...)` is the legacy **span** loader and will not dispatch this checkpoint.

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")

print(type(model).__name__)
print(model.config.architecture)
# BoundaryExtractor
# boundary
```

Optional device, fp16, and compile flags:

```python
model = AutoExtractor.from_pretrained(
    "fastino/gliner2.5-multi-v1",
    map_location="cuda",   # or "cpu" / "mps"
    quantize=True,         # fp16 weights on GPU
    compile=True,          # torch.compile after the first tracing call
)
print(type(model).__name__, next(model.parameters()).device)
# BoundaryExtractor cuda:0
```

## Usage

### Entity extraction

```python
text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."

result = model.extract_entities(
    text,
    ["company", "person", "product", "location"],
    include_confidence=True,
    include_spans=True,
)
print(result)
# {
#     "entities": {
#         "company": [{"text": "Apple", "start": 0, "end": 5, "confidence": 0.98}],
#         "person": [{"text": "Tim Cook", "start": 10, "end": 18, "confidence": 0.97}],
#         "product": [{"text": "iPhone 15", "start": 29, "end": 38, "confidence": 0.96}],
#         "location": [{"text": "Cupertino", "start": 42, "end": 51, "confidence": 0.95}],
#     }
# }
```

Returned offsets are half-open character spans into the original string: `text[start:end] == entity["text"]`.

Add descriptions when labels are domain-specific:

```python
result = model.extract_entities(
    "Patient received 400mg ibuprofen for severe headache at 2 PM.",
    {
        "medication": "Names of drugs or pharmaceutical substances",
        "dosage": "Amounts such as 400mg, 2 tablets, or 5ml",
        "symptom": "Reported symptoms or conditions",
        "time": "Clock times or relative times",
    },
    include_spans=True,
)
print(result)
# {
#     "entities": {
#         "medication": [{"text": "ibuprofen", "start": 23, "end": 32}],
#         "dosage": [{"text": "400mg", "start": 17, "end": 22}],
#         "symptom": [{"text": "severe headache", "start": 37, "end": 52}],
#         "time": [{"text": "2 PM", "start": 56, "end": 60}],
#     }
# }
```

### Text classification

Independent per-task decoding with `classify_text`:

```python
result = model.classify_text(
    "This laptop has amazing performance but terrible battery life!",
    {"sentiment": ["positive", "negative", "neutral"]},
)
print(result)
# {"sentiment": "negative"}

result = model.classify_text(
    "Great camera quality, decent performance, but poor battery life.",
    {
        "aspects": {
            "labels": ["camera", "performance", "battery", "display", "price"],
            "multi_label": True,
            "cls_threshold": 0.4,
        }
    },
)
print(result)
# {"aspects": ["camera", "performance", "battery"]}
```

### Constrained classification

Use `gliner2.classification.Classifier` when labels on one task legally constrain another. `classify_text` will not enforce those rules.

```python
from gliner2.classification import (
    Classifier,
    ClassificationSchema,
    ClassificationConfig,
)
from gliner2.classification import constraints as C

clf = Classifier.from_pretrained("fastino/gliner2.5-multi-v1")

schema = (
    ClassificationSchema()
    .single("intent", ["read", "write", "delete"])
    .multi("effects", ["read_only", "create", "modify", "delete"], min_labels=1)
    .constrain(
        C.implies(("intent", "delete"), ("effects", "delete")),
        C.excludes(("intent", "read"), ("effects", "delete")),
    )
)

result = clf.classify("Delete the temporary file from /tmp", schema)
print(result.value("intent"))
print(result.value("effects"))
print(result.feasible)
print(result.to_dict())
# delete
# ['delete']
# True
# {
#     "intent": {
#         "value": "delete",
#         "confidence": 0.93,
#         "probabilities": {"read": 0.02, "write": 0.05, "delete": 0.93},
#     },
#     "effects": {
#         "value": ["delete"],
#         "confidence": 0.88,
#         "probabilities": {
#             "read_only": 0.04, "create": 0.03, "modify": 0.05, "delete": 0.88
#         },
#     },
#     "_meta": {"feasible": True, "decoder": "exact"},
# }
```

Prediction knobs belong in `ClassificationConfig` on the call, not in `from_pretrained`:

```python
result = clf.classify(
    "Preview the report",
    schema,
    config=ClassificationConfig(decoder="beam", beam_size=16),
)
print(result.value("intent"), result.value("effects"), result.feasible)
# read ['read_only'] True
```

### Relation extraction

This checkpoint was trained with `enable_relations=True`. Independent decoding:

```python
text = "Alice works for Acme in Paris."
result = model.extract_relations(
    text,
    ["works_for", "located_in"],
    include_spans=True,
    include_confidence=True,
)
print(result)
# {
#     "relation_extraction": {
#         "works_for": [{
#             "head": {"text": "Alice", "start": 0, "end": 5, "confidence": 0.91},
#             "tail": {"text": "Acme", "start": 16, "end": 20, "confidence": 0.91},
#         }],
#         "located_in": [{
#             "head": {"text": "Acme", "start": 16, "end": 20, "confidence": 0.87},
#             "tail": {"text": "Paris", "start": 24, "end": 29, "confidence": 0.87},
#         }],
#     }
# }
```

Or through a schema:

```python
schema = model.create_schema().relations(
    {"works_for": {"threshold": 0.6}, "located_in": {"threshold": 0.6}}
)
result = model.extract(text, schema, include_spans=True)
print(result)
# {
#     "relation_extraction": {
#         "works_for": [{
#             "head": {"text": "Alice", "start": 0, "end": 5},
#             "tail": {"text": "Acme", "start": 16, "end": 20},
#         }],
#         "located_in": [{
#             "head": {"text": "Acme", "start": 16, "end": 20},
#             "tail": {"text": "Paris", "start": 24, "end": 29},
#         }],
#     }
# }
```

Independent extraction does **not** guarantee that `works_for` heads are people and tails are organizations.

### Joint information extraction

`JointIE` scores mention and relation candidates, then searches a globally consistent graph with typed endpoints and uniqueness constraints.

```python
from gliner2.joint_ie import JointIE, JointIEConfig

joint = JointIE.from_pretrained("fastino/gliner2.5-multi-v1")

schema = (
    joint.create_schema()
    .entities(["person", "organization", "location"])
    .relation("works_for", "person", "organization", unique_head=True)
    .relation("located_in", "organization", "location")
    .no_self_loops()
)

result = joint.extract(
    "Alice works for Acme in Paris. Bob joined Acme last year.",
    schema,
    config=JointIEConfig(optimizer="beam", beam_size=32),
)

print(result.feasible)
print(result.to_dict())
# True
# {
#     "entities": [
#         {"id": "e1", "type": "person", "text": "Alice", "start": 0, "end": 5, "confidence": 0.94},
#         {"id": "e2", "type": "organization", "text": "Acme", "start": 16, "end": 20, "confidence": 0.92},
#         {"id": "e3", "type": "location", "text": "Paris", "start": 24, "end": 29, "confidence": 0.90},
#         {"id": "e4", "type": "person", "text": "Bob", "start": 31, "end": 34, "confidence": 0.91},
#     ],
#     "relations": [
#         {"type": "works_for", "head": "e1", "tail": "e2", "confidence": 0.88},
#         {"type": "works_for", "head": "e4", "tail": "e2", "confidence": 0.81},
#         {"type": "located_in", "head": "e2", "tail": "e3", "confidence": 0.86},
#     ],
# }
```

Always check `result.feasible`. `False` means the hard constraints could not be satisfied (distinct from “the text contains no facts”).

```python
for rel in result.relations:
    head = result.entity(rel.head)
    tail = result.entity(rel.tail)
    print(f"{head.text} -{rel.type}-> {tail.text}")
# Alice -works_for-> Acme
# Bob -works_for-> Acme
# Acme -located_in-> Paris
```

### Span attributes: people with sentiment

Attributes are **span-conditioned**. The model finds entities first, then scores attribute labels at those exact spans. They are not extra entity types and they are not document-level classification.

```python
from gliner2 import AutoExtractor, AttributeGroup

model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")

text = (
    "Alice was delighted with the promotion, "
    "but Bob sounded frustrated about the delay."
)

schema = (
    model.create_schema()
    .entities(["person"])
    .entity_attributes({
        "sentiment": AttributeGroup(
            ["positive", "negative", "neutral"],
            applies_to=["person"],
            qualify_labels=True,
        )
    })
)

result = model.extract(
    text,
    schema,
    include_spans=True,
    include_confidence=True,
)
print(result)
# {
#     "entities": {
#         "person": [
#             {
#                 "text": "Alice",
#                 "start": 0,
#                 "end": 5,
#                 "confidence": 0.96,
#                 "sentiment": {"label": "positive", "confidence": 0.89},
#             },
#             {
#                 "text": "Bob",
#                 "start": 44,
#                 "end": 47,
#                 "confidence": 0.95,
#                 "sentiment": {"label": "negative", "confidence": 0.84},
#             },
#         ]
#     }
# }
```

`applies_to=["person"]` keeps sentiment off other entity types. `qualify_labels=True` encodes model-facing queries as `sentiment: positive` while returning the short label `positive`.

Restrict sentiment to people while still extracting companies:

```python
schema = (
    model.create_schema()
    .entities(["person", "organization"])
    .entity_attributes({
        "sentiment": AttributeGroup(
            ["positive", "negative", "neutral"],
            applies_to=["person"],
            qualify_labels=True,
        )
    })
)

result = model.extract(
    "Alice praised Microsoft, but Bob criticized OpenAI.",
    schema,
    include_spans=True,
    include_confidence=True,
)
print(result)
# {
#     "entities": {
#         "person": [
#             {
#                 "text": "Alice",
#                 "start": 0,
#                 "end": 5,
#                 "confidence": 0.96,
#                 "sentiment": {"label": "positive", "confidence": 0.88},
#             },
#             {
#                 "text": "Bob",
#                 "start": 29,
#                 "end": 32,
#                 "confidence": 0.95,
#                 "sentiment": {"label": "negative", "confidence": 0.86},
#             },
#         ],
#         "organization": [
#             {"text": "Microsoft", "start": 14, "end": 23, "confidence": 0.97},
#             {"text": "OpenAI", "start": 44, "end": 50, "confidence": 0.96},
#         ],
#     }
# }
```

Organization spans have no `sentiment` field. Person spans do.

### Structured records

Record mode keeps instance identity (who bought what) instead of flattening fields into unrelated lists. Enable `natural` mode with an anchor field:

```python
schema = (
    model.create_schema()
    .structure("purchase", mode="natural", anchor="buyer")
    .field("buyer", dtype="str", cardinality="required_one")
    .field("item", dtype="str", cardinality="required_one")
)

result = model.extract(
    "Alice bought apples and Bob bought oranges.",
    schema,
)
print(result)
# {
#     "purchase": [
#         {"buyer": "Alice", "item": "apples"},
#         {"buyer": "Bob", "item": "oranges"},
#     ]
# }
```

This checkpoint was trained with `enable_records=True`.

### Task combination

Compose entities, span attributes, classification, relations, and structures in **one** `extract` call:

```python
from gliner2 import AttributeGroup

schema = (
    model.create_schema()
    .entities({
        "person": "Named people",
        "organization": "Companies or teams",
        "product": "Named products or services",
    })
    .entity_attributes({
        "sentiment": AttributeGroup(
            ["positive", "negative", "neutral"],
            applies_to=["person"],
            qualify_labels=True,
        )
    })
    .classification("topic", ["technology", "business", "sports", "politics"])
    .relations(["works_for", "announced"])
    .structure("announcement", mode="natural", anchor="product")
        .field("company", dtype="str")
        .field("product", dtype="str", cardinality="required_one")
)

text = "Apple CEO Tim Cook unveiled the iPhone 15 Pro for $999."
result = model.extract(text, schema, include_spans=True, include_confidence=True)
print(result)
# {
#     "entities": {
#         "person": [{
#             "text": "Tim Cook",
#             "start": 10,
#             "end": 18,
#             "confidence": 0.97,
#             "sentiment": {"label": "positive", "confidence": 0.82},
#         }],
#         "organization": [{"text": "Apple", "start": 0, "end": 5, "confidence": 0.98}],
#         "product": [{"text": "iPhone 15 Pro", "start": 32, "end": 45, "confidence": 0.96}],
#     },
#     "topic": {"label": "technology", "confidence": 0.94},
#     "relation_extraction": {
#         "works_for": [{
#             "head": {"text": "Tim Cook", "start": 10, "end": 18, "confidence": 0.86},
#             "tail": {"text": "Apple", "start": 0, "end": 5, "confidence": 0.86},
#         }],
#         "announced": [{
#             "head": {"text": "Tim Cook", "start": 10, "end": 18, "confidence": 0.84},
#             "tail": {"text": "iPhone 15 Pro", "start": 32, "end": 45, "confidence": 0.84},
#         }],
#     },
#     "announcement": [{
#         "company": "Apple",
#         "product": "iPhone 15 Pro",
#     }],
# }
```

Document-level `topic` is independent of per-person `sentiment`.

### Batch inference

```python
texts = [
    "Google hired Jane Doe in London.",
    "Tesla launched the Model 3 in California.",
]
results = model.batch_extract_entities(
    texts,
    ["company", "person", "product", "location"],
    batch_size=8,
    include_spans=True,
)
print(results)
# [
#     {
#         "entities": {
#             "company": [{"text": "Google", "start": 0, "end": 6}],
#             "person": [{"text": "Jane Doe", "start": 13, "end": 21}],
#             "product": [],
#             "location": [{"text": "London", "start": 25, "end": 31}],
#         }
#     },
#     {
#         "entities": {
#             "company": [{"text": "Tesla", "start": 0, "end": 5}],
#             "person": [],
#             "product": [{"text": "Model 3", "start": 19, "end": 26}],
#             "location": [{"text": "California", "start": 30, "end": 40}],
#         }
#     },
# ]
```

`batch_extract` accepts one schema or a list of schemas (one per document).

### Long documents

`extract(...)` with `max_len` **truncates**. Long-context helpers scan overlapping word chunks and remap spans to document offsets.

```python
long_text = ("Quarterly overview. " * 40) + "Satya Nadella spoke in Redmond about Microsoft."

result = model.extract_entities_long(
    long_text,
    ["person", "organization", "location"],
    chunk_size=384,
    chunk_overlap=64,
    include_spans=True,
)
print(result)
# {
#     "entities": {
#         "person": [{"text": "Satya Nadella", "start": 800, "end": 813}],
#         "organization": [{"text": "Microsoft", "start": 837, "end": 846}],
#         "location": [{"text": "Redmond", "start": 823, "end": 830}],
#     }
# }

result = model.extract_long(long_text, schema, chunk_size=384, chunk_overlap=64)
print(result["topic"])
# technology
```

The same idea applies to `Classifier.classify_long` and `JointIE.extract_long`.

Limits:

- A span is kept only if its start and end fall in the **same chunk**.
- A relation is kept only if both endpoints were extracted in the same chunk.
- Boundary models can represent arbitrarily long spans **inside one encoded window**; they do not stitch a mention whose endpoints never co-occur.

## Model details

- **Architecture:** GLiNER2 **boundary** extractor (`BoundaryExtractor`)
- **Candidate search:** sparse start/end pairing (not a dense `[L, W]` width grid)
- **Span length:** any length that fits in the encoded window (`max_len=4096`)
- **Encoder:** `microsoft/mdeberta-v3-base`
- **Parameters:** 287M
- **Weights:** ~594 MB (mostly FP16)
- **Language:** Multilingual
- **Heads enabled:** classification, records (`enable_records=True`), relations (`enable_relations=True`)
- **Overlap default:** `flat` (weighted interval scheduling); override per call with `overlap_policy`
- **Input / output:** text → entities, labels, span attributes, records, and relation edges

Do not load this checkpoint with `GLiNER2` / `SpanExtractor`. Those classes expect the legacy span architecture.

## Fine-tuning

This model can be fine-tuned locally with the `gliner2` trainer (see the [training guide](https://github.com/fastino-ai/GLiNER2#-training-custom-models)). To fine-tune or serve it without managing GPUs, the [Fastino API](https://agent.fastino.ai) offers hosted training and inference, usually faster than training locally.

## Citation

If you use this model, please cite:

```bibtex
@misc{zaratiana2025gliner2efficientmultitaskinformation,
      title={GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface},
      author={Urchade Zaratiana and Gil Pasternak and Oliver Boyd and George Hurn-Maloney and Ash Lewis},
      year={2025},
      eprint={2507.18546},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.18546},
}
```

## License

Apache License 2.0.

## Links

- **Repository:** https://github.com/fastino-ai/GLiNER2
- **Paper:** https://arxiv.org/abs/2507.18546
- **Docs:** [boundary architecture](https://github.com/fastino-ai/GLiNER2/blob/main/docs/boundary_architecture.md) · [span attributes](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/13-span_attributes.md) · [constrained classification](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/14-constrained_classification.md) · [joint IE](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/15-joint_ie.md)
- **Organization:** [Fastino AI](https://fastino.ai)
