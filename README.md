# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two-stage retrieval over the **FIDE Laws of Chess (2023)** PDF (`data/document.pdf`).

- **Pipeline 1 (baseline):** `all-MiniLM-L6-v2` bi-encoder → FAISS inner product (cosine) → Top 10 → Top 3 to the LLM
- **Pipeline 2 (re-rank):** same Top 10 → `cross-encoder/ms-marco-MiniLM-L-6-v2` → re-sorted Top 3 to the LLM
- **LLM:** Gemini (`gemini-3.8-flash`), identical prompt for both pipelines

Everything is in one notebook: **`RAG_Baseline_vs_Reranking.ipynb`** (chunking, indexing, both pipelines,
failure-mode tests, rank shifts, answer comparison, retrieval metrics, latency profiling, scaling experiment).

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your key: GEMINI_API_KEY=...
jupyter notebook RAG_Baseline_vs_Reranking.ipynb   # or open in VS Code and "Run All"
```

LLM answers are cached in `results/llm_cache.json`, so re-running does not use API quota again
(delete that file to regenerate).

## Outputs (`results/`)

| File | Content |
|---|---|
| `rank_shifts.csv` | gold-chunk rank before/after re-ranking for each failure query |
| `answers.csv`, `answers.md` | Answer 1 (baseline) vs. Answer 2 (re-rank) |
| `retrieval_metrics.csv`, `eval_ranks.csv` | Hit@3 / MRR@10 over 36 labelled questions |
| `latency.csv` | per-stage latency (retrieval, re-ranking, LLM) |
| `cross_encoder_scaling.csv` | cross-encoder time vs. number of chunks scored |
| `chunks.csv` | chunk list with clause labels and pages |
