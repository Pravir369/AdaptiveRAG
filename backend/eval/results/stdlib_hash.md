Corpus: 33 Python stdlib docstrings (40 chunks). 42 scored questions (28 paraphrased, 14 keyword). Embeddings: `hash`. Generator: `extractive`. top_k=12. Python 3.14.6.

| Retrieval | Hit@1 | Hit@5 | MRR@10 | Hit@1 (paraphrased) | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.76 | 0.95 | 0.84 | 0.68 | 1.0 ms | 1.1 ms |
| Vector only | 0.52 | 0.88 | 0.66 | 0.50 | 3.6 ms | 4.5 ms |
| Hybrid (no self-heal) | 0.76 | 0.91 | 0.83 | 0.71 | 3.5 ms | 4.3 ms |
| Hybrid + self-heal (full pipeline) | 0.76 | 0.91 | 0.83 | 0.71 | 3.8 ms | 4.7 ms |

Full pipeline answer match (extractive answer contains the expected phrase): 0.52 (answer latency p50 1 ms, p95 4 ms). Self-heal triggered on 0 of 42 scored questions and 2/3 vague ones.
