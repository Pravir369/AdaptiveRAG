# Eval JSONL schema

Each line is a JSON object with:

- **query** (string): Question to run against the RAG API.
- **expected_doc_ids** (array of strings): Doc IDs that should appear in the retrieved set for a "hit". For HitRate@10 we check whether any of these doc_ids appear in the top-10 retrieved doc_ids (from trace payload).

Optional:

- **id** (string): Optional example id for logging.
- **expected_answer_contains** (string): Optional substring to check in the answer (not used in v0 metrics).
