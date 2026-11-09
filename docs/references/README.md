# Third-party reference material

These files are vendored for reference. They are **not** covered by this project's AGPL-3.0
license; each keeps its original license and attribution.

| File | Source | License |
| --- | --- | --- |
| `gliner2-arxiv-2507.18546.pdf` | Zaratiana et al., *GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface*, [arXiv:2507.18546](https://arxiv.org/abs/2507.18546). Cited by the GLiNER2.5 model cards; there is no separate GLiNER2.5 paper. | CC BY 4.0 |
| `gliner2.5-multi-v1-model-card.md` | [fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1) model card | Apache-2.0 |
| `gliner2.5-boundary-baseline.md`, `gliner2-RELEASE.md`, `gliner2-tutorial-6-relation-extraction.md`, `gliner2-tutorial-15-joint-ie.md` | [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) (`docs/`, `RELEASE.md`, `tutorial/`) | Apache-2.0 |
| `ace2005-C12-2086.pdf` | Nguyen & Poesio, *Relational Structures and Models for Coreference Resolution*, COLING 2012 ([C12-2086](https://aclanthology.org/C12-2086/)). Table 1 summarises the ACE 2005 relation types. | CC BY-NC-SA 3.0 |
| `fewrel-D18-1514.pdf` | Han et al., *FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset*, EMNLP 2018 ([D18-1514](https://aclanthology.org/D18-1514/)) | CC BY 4.0 |
| `fewrel-pid2name.json` | [thunlp/FewRel](https://github.com/thunlp/FewRel) `data/pid2name.json` (Wikidata property id → name, description) | MIT |
| `gliner2.5-blog-span-free-ie.md` | Notes (our own words, no verbatim reproduction) on Newhauser & Zaratiana, [*Introducing GLiNER2.5: Efficient Span-Free Information Extraction*](https://fastino.ai/blog/gliner2-5-span-free-information-extraction), Fastino Labs blog, 2026 (retrieved 2026-11-09) | Notes: AGPL-3.0-or-later (this project); the original post © Fastino Labs |

Upstream source is also pinned as git submodules under `third_party/`: `GLiNER2` at tag `v2.0.0` (Apache-2.0; `v2.1.0` exists upstream with the same weights) and `gliner25-rs` at commit `580c57b` (crate 0.5.6, Apache-2.0; no tags upstream).
