Corpus: 33 Python stdlib docstrings (40 chunks). 42 scored questions (28 paraphrased, 14 keyword). Embeddings: `sbert`. Generator: `extractive`. top_k=12. Python 3.14.6.

| Retrieval | Hit@1 | Hit@5 | MRR@10 | Hit@1 (paraphrased) | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.76 | 0.95 | 0.84 | 0.68 | 1.0 ms | 1.0 ms |
| Vector only | 0.98 | 1.00 | 0.98 | 0.96 | 116.2 ms | 153.8 ms |
| Hybrid (no self-heal) | 0.95 | 1.00 | 0.97 | 0.93 | 116.9 ms | 144.3 ms |
| Hybrid + self-heal (full pipeline) | 0.95 | 1.00 | 0.97 | 0.93 | 117.9 ms | 148.1 ms |

Full pipeline answer match (extractive answer contains the expected phrase): 0.64 (answer latency p50 1 ms, p95 4 ms). Self-heal triggered on 0 of 42 scored questions and 2/3 vague ones.
