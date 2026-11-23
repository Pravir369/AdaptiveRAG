# Eval sets

There are two ways to evaluate AdaptiveRAG.

## 1. Offline retrieval eval (recommended, reproducible)

`python eval/run_eval.py` builds a fresh index in a temp directory from `eval/corpus/`
and scores four retrieval setups (BM25, vector, hybrid, hybrid + self-heal) on
`eval/stdlib_qa.jsonl`. No server, no LLM, no network. Results go to `eval/results/`.

Each line of `stdlib_qa.jsonl`:

- **id** (string): example id.
- **query** (string): the question.
- **expected_doc_titles** (array of strings): titles of the docs that answer it. A doc's title
  is its corpus file name without `.txt`. Empty means unscored (used for vague queries).
- **expected_answer_contains** (string): a phrase from the source doc; the full pipeline's
  extractive answer is checked for it (whitespace and case insensitive).
- **style** (string): `keyword` (shares words with the doc), `paraphrase` (mostly
  doesn't), or `vague` (no single right doc).

To evaluate your own documents, put `.txt` files in a folder, write a JSONL in this format,
and run `python eval/run_eval.py --corpus path/to/folder --qa path/to/questions.jsonl`.

## 2. Live API eval

`python eval/eval_runner.py` calls a running backend (`/api/chat` then `/api/trace/{id}`)
for each line of `sample_eval.jsonl` and reports HitRate@10 using **expected_doc_ids**
(the UUIDs the server assigned at ingest). The sample file ships without ground truth, so
fill in `expected_doc_ids` for your own ingested docs before reading its numbers.
