Corpus: 33 Python stdlib docstrings (40 chunks). 42 scored questions (28 paraphrased, 14 keyword). Embeddings: `hash`. top_k=12. Python 3.13.18.

| Retrieval | Hit@1 | Hit@5 | MRR@10 | Hit@1 (paraphrased) | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.76 | 0.95 | 0.84 | 0.68 | 0.9 ms | 1.0 ms |
| Vector only | 0.52 | 0.88 | 0.66 | 0.50 | 4.8 ms | 5.8 ms |
| Hybrid (no self-heal) | 0.76 | 0.91 | 0.83 | 0.71 | 4.4 ms | 5.4 ms |
| Hybrid + self-heal (full pipeline) | 0.76 | 0.91 | 0.83 | 0.71 | 3.8 ms | 4.5 ms |

Full pipeline answer match (extractive answer contains the expected phrase): 0.52. Self-heal triggered on 0 of 42 scored questions and 2/3 vague ones.
