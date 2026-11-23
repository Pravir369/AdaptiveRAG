Corpus: 33 Python stdlib docstrings (40 chunks). 42 scored questions (28 paraphrased, 14 keyword). Embeddings: `sbert`. Generator: `ollama` (llama3.2:3b). top_k=12. Python 3.14.6.

| Retrieval | Hit@1 | Hit@5 | MRR@10 | Hit@1 (paraphrased) | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.76 | 0.95 | 0.84 | 0.68 | 1.0 ms | 1.0 ms |
| Vector only | 0.98 | 1.00 | 0.98 | 0.96 | 124.7 ms | 161.9 ms |
| Hybrid (no self-heal) | 0.95 | 1.00 | 0.97 | 0.93 | 124.3 ms | 159.7 ms |
| Hybrid + self-heal (full pipeline) | 0.95 | 1.00 | 0.97 | 0.93 | 734.1 ms | 852.5 ms |

Full pipeline answer match (ollama answer contains the expected phrase): 0.62 (answer latency p50 21564 ms, p95 41398 ms). Self-heal triggered on 0 of 42 scored questions and 2/3 vague ones.
