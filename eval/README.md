# Evaluation harness (task 4.3)

Three scripts, all runnable with no API key:

```powershell
python eval/run_eval.py                 # the 22-question set -> results.json / results.csv / summary.json + traces
python eval/run_eval.py --suffix _llm   # same set via a hosted LLM (needs GROQ_API_KEY) -> results_llm.json / summary_llm.json + *_llm traces
python eval/improvement_experiment.py   # chunk-size A/B and retry-policy A/B -> improvement_*.json + improvement.png
python eval/local_tracer.py             # render one trace on its own (used as a module by the above)
```

If `LANGSMITH_API_KEY` is set, `run_eval.py` exports the tracing environment **before importing
LangChain**, so the same run also lands in LangSmith. With no key, every node still records a span
in `state["trace"]` and `local_tracer.py` renders it. The results never depend on which tracer is
active.

## Files

| File | What it is |
|---|---|
| `testset.json` | 22 labelled questions: expected route, expected source document, expected answer keywords, and `in_scope` |
| `run_eval.py` | runs the set, scores it, writes the results and a few trace images |
| `results.json` / `results.csv` | per-question outcome + the metrics block |
| `summary.json` | just the metrics |
| `results_llm.json` / `summary_llm.json` | the same set run through the hosted LLM (`--suffix _llm`) |
| `summary_llm_naive.json` | the first hosted run, before the router prompt was aligned to the agent's policy |
| `local_tracer.py` | turns a `state["trace"]` list into `eval/traces/*.json` and `docs/traces/*.png` |
| `improvement_experiment.py` | the two ablations below |
| `improvement_chunks.json`, `improvement_retry.json`, `improvement.png` | their outputs |

## Metric definitions

| Metric | Definition | Applies to |
|---|---|---|
| `routing_accuracy` | graph's route == the labelled route | all 22 |
| `retrieval_hit_rate` | the labelled source document is among the retrieved sources | in-scope `retrieve` questions |
| `answer_keyword_rate` | the answer contains at least one labelled keyword | in-scope `retrieve` questions |
| `tool_success_rate` | tool route, the tool reported success, and the answer contains the expected value | `tool` questions |
| `direct_ok_rate` | small talk answered without retrieval | `direct` questions |
| `out_of_scope_refusal_rate` | the answer contains an explicit refusal (a "could not find" / "does not contain" / "cannot answer" phrasing, matched after normalising typographic dashes and LaTeX markup) | questions labelled `in_scope: false` |
| `grounded_confidence_rate` | the run ended with confidence `grounded` | in-scope `retrieve` questions |
| `latency_ms` | wall-clock per question, after one warm-up call excluded as model-load | all |

## How to read the headline numbers honestly

`routing_accuracy`, `retrieval_hit_rate` and `tool_success_rate` are at 100% and that is *expected,
not impressive*: the questions were written by the same person who wrote the corpus, and the
offline grader is lexical. The two metrics that carry information are:

- **`answer_keyword_rate` 93% (13/14)** - one in-scope answer missed its keyword (q22, FM bandwidth:
  the right document was retrieved but the extractive generator quoted a neighbouring sentence
  about FM noise robustness instead of Carson's rule).
- **`out_of_scope_refusal_rate` 67% (2/3)** - the one failure (q20, Friis equation) is the most
  interesting result in the whole evaluation, because the question *sounds* in-domain ("transmission
  ... formula") so the grader sees enough lexical overlap to pass it, and the agent answers with a
  real-but-irrelevant passage on transmission lines. See `docs/4.3_evaluation.md`.

Both failures are analysed in `docs/4.3_evaluation.md`; neither is papered over.

## Hosted-LLM run

With `GROQ_API_KEY` set, `python eval/run_eval.py --suffix _llm` runs the identical set with the
hosted model doing the routing, grading and generation. The first pass (kept as `summary_llm_naive.json`)
scored 45% routing / 0% refusal because the model answered general-knowledge and out-of-scope
questions from memory; after the router prompt was constrained to the agent's "answer from the notes"
policy it matches the offline metrics while answering more questions correctly, at a median ~17 s per
question. The full before/after table and the two graders' opposite failure modes are in
`docs/4.3_evaluation.md`.

## The two ablations

**Chunking** (`improvement_chunks.json`). The corpus is re-indexed at 400/100, 800/150 and
1200/240 and retrieval is re-scored.

**Retry policy** (`improvement_retry.json`). The low-coverage floor in `graph.py` (`RAG_LOW_COVER_FLOOR`,
default 0.15) is set to 0 and to 0.15 and the unanswerable questions are run both ways. Setting it
to 0 restores the naive behaviour in which a failed grade always rewrites and re-retrieves.
