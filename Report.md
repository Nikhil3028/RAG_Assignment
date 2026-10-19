# RAG Assignment: Baseline Retrieval vs Re-Ranking

## Setup

| Item | Choice |
|---|---|
| Document | FIDE Laws of Chess (2023), 41 pages |
| Chunking | Split by clause number, about 700 characters per chunk, 124 chunks in total |
| Bi-encoder | sentence-transformers/all-MiniLM-L6-v2 with a FAISS inner product index (cosine similarity) |
| Cross-encoder | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Gemini (gemini-3.8-flash), temperature 0, same prompt for both pipelines |
| Retrieval | Top 10 from the bi-encoder, Top 3 sent to the LLM |

Pipeline 1 sends bi-encoder ranks 1 to 3 to the LLM. Pipeline 2 takes the same 10 candidates, scores all 10 pairs with the cross-encoder, sorts them again and sends the new top 3. The only difference between Answer 1 and Answer 2 is which 3 chunks are in the prompt.

## 1. Architecture Comparison

A bi-encoder turns the query and each chunk into a vector separately. The chunk vectors are computed once, before any question is asked, and stored in the index. At query time only the query is encoded and the closest vectors are found with cosine similarity. This is very fast, but the query and the chunk never see each other inside the model. The whole meaning of a chunk is squeezed into one vector, so small but important words like "not", "except" or "automatic" have very little effect on the score, and chunks that share many words with the query score high even when they do not answer it.

A cross-encoder takes the query and the chunk together as one input. Self-attention runs across the tokens of both texts, so every word of the query can be compared directly with every word of the chunk. The output is one relevance score for that pair. This is much better at judging whether the chunk actually answers the question, but nothing can be computed in advance. The model has to run once for every (query, chunk) pair at query time.

## 2. Qualitative and Rank Shift Analysis

### Rank shifts

The gold chunk is the chunk that contains the clause with the correct answer.

| ID | Type | Query | Gold clause | Baseline rank | Re-ranked rank |
|---|---|---|---|---|---|
| KT1 | Keyword trap | Can a player stop the clock to go to the toilet? | 6.11.2 | 8 | 1 |
| KT2 | Keyword trap | Can a player use two hands to castle? | 7.5.4 | 8 | 1 |
| EX1 | Exception | If a player runs out of time, do they always lose the game? | 6.9 | 7 | 2 |
| EX2 | Exception | Is the draw automatic after 50 moves without capture? | 9.6.2 | 4 | 3 |
| NG1 | Negation | What is the arbiter not allowed to tell a player during the game? | 12.6 | 4 | 3 |
| BU1 | Buried chunk | Can a player be penalised for knocking pieces over? | 7.4.3 | 8 | 3 |
| CTRL | Control | Can the time on the clock be corrected after an illegal move? | 7.1 | 3 | 7 |

In the first six queries the gold chunk was outside the baseline top 3, so the LLM never saw it. After re-ranking it was inside the top 3 every time.

Why the baseline failed:

* KT1: the word "clock" pulled in all the clock handling clauses (6.2.x). The answer uses different words: a player may "pause the chessclock only to seek the arbiter's assistance".
* KT2: "castle" matched the castling rules in Articles 3 and 4. The rule about using two hands is in Article 7 (illegal moves).
* EX1: the general time and flag rules were retrieved. The exception (the game is drawn if the opponent cannot checkmate) is in 6.9.
* EX2: the 50 move claim rule (9.3) ranked first. The automatic draw only happens at 75 moves (9.6.2).
* NG1: "not allowed" matched clauses about what players are allowed to do (11.2.x). Clause 12.6 says what the arbiter "must not" do.
* BU1: "knocking over" only appears for the clock (6.2.4). The rule for pieces uses the word "displaces".

To check that this is not only true for hand picked queries, both rankings were scored on 36 labelled questions:

| Metric | Baseline | Re-rank |
|---|---|---|
| Hit@3 (gold chunk reaches the LLM) | 0.611 | 0.889 |
| MRR@10 | 0.567 | 0.718 |
| Recall@10 (same 10 candidates for both) | 0.972 | 0.972 |

Recall@10 is the same because both pipelines start from the same 10 chunks. The gain comes only from putting the right chunk inside the top 3.

### Answer 1 vs Answer 2

* KT2, EX1, NG1, BU1: Answer 1 was "The provided context does not contain this information." Answer 2 gave the correct rule with the right clause (7.5.4, 6.9, 12.6, 7.4.3). The baseline omitted the essential clause because it sat outside the top 3.
* EX2: Answer 1 was not wrong (the draw is not automatic, it must be claimed) but it was incomplete. Only Answer 2 mentioned the automatic draw at 75 moves from 9.6.2.
* KT1: Answer 1 had no information. Answer 2 had the right clause and correctly said the clock can only be paused to call the arbiter, but it still started with "does not contain this information" before giving the rule. The context was fixed but the LLM was overly careful.
* CTRL: here re-ranking made things worse. Clause 7.1 (the arbiter uses best judgement to set the clock times) was at rank 3 in the baseline and Answer 1 was correct. The cross-encoder moved it to rank 7, and Answer 2 said the context does not contain the answer.

In this run the baseline did not hallucinate. Because the prompt tells the model to use only the context, the main failure was a missing answer or a missing clause, not an invented rule. Re-ranking fixed six of the seven cases and broke one, so it helps a lot but is not always right.

## 3. Latency Profiling

Measured on a Google Colab CPU runtime. Retrieval was timed 5 times over all 36 questions (180 runs per stage). LLM times are from the 7 failure queries.

| Stage | Mean (ms) | Median (ms) | p95 (ms) |
|---|---|---|---|
| Stage 1: embed query and FAISS top 10 | 18.9 | 17.5 | 26.8 |
| Stage 2: cross-encoder on 10 pairs | 825.4 | 770.6 | 1169.9 |
| Retrieval total, Pipeline 1 | 18.9 | 17.5 | 26.8 |
| Retrieval total, Pipeline 2 | 844.3 | 787.2 | 1196.1 |
| LLM generation, Pipeline 1 | 4281.9 | 2812.2 | 10321.6 |
| LLM generation, Pipeline 2 | 3184.1 | 2119.0 | 7574.2 |
| End to end, Pipeline 1 | 4299.4 | 2829.7 | 10339.1 |
| End to end, Pipeline 2 | 3971.3 | 2906.3 | 8361.4 |

Re-ranking makes retrieval about 45 times slower (17.5 ms to 787 ms median). Still, the Gemini call takes 2 to 3 seconds, so the full answer takes almost the same time in both pipelines. The LLM time changes a lot from call to call (network and API load), which is why Pipeline 2 looks faster end to end. That difference is noise, not a real speed up. On a GPU the cross-encoder step would be much faster than on this CPU.

## 4. Scaling Question

The cross-encoder was timed on more and more chunks for one query:

| Chunks scored | Time |
|---|---|
| 10 | 0.69 s |
| 25 | 1.84 s |
| 50 | 4.19 s |
| 124 (whole document) | 13.75 s |

The time grows in a straight line, about 0.11 s per chunk on this CPU. Scoring the whole 41 page document already takes almost 14 seconds for a single question. For 100,000 chunks it would be about 3 hours per question, and for 10 million chunks about 13 days.

We do not run the cross-encoder on everything because its work cannot be done in advance. Every new query needs a full model pass for every chunk. The bi-encoder does the expensive part (encoding all chunks) once, offline, and at query time it only encodes the query and does a fast vector search, which stays in milliseconds even for millions of chunks with an approximate index. So the practical design is two stages: the cheap bi-encoder narrows the collection down to a small set (here 10), and the accurate cross-encoder is only used on that small set.
