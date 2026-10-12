# ECE Study Assistant - a LangGraph RAG agent

A retrieval-augmented agent that answers questions over eight ECE lecture-note documents, does
calculations and formula look-ups as tool calls, and shows its full reasoning trace. It runs with
**no API key** (deterministic extractive generation) and switches to Groq or Gemini automatically
when one is present.

This is the Week 4 capstone: tasks 4.1 (RAG), 4.2 (multi-step workflow), 4.3 (observability &
evaluation) and 4.4 (demo + report) are all implemented in this one repository.

**Live demo:** _pending deployment_ · **Repository:** <https://github.com/crahulece2028-lab/week4-study-assistant-agent>

```
round 1  route -> retrieve -> grade -+-> generate (grounded, cited)
                                     +-> rewrite -> retrieve   (bounded retry, up to 2)
tool     route -> tool ------> generate
meta     route -> direct ------> end
```

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python data/build_corpus.py     # generate the 8 lecture-note PDFs
python src/ingest.py            # chunk + embed + FAISS  -> index/ and docs/ingest_stats.json

# 1) command line
python src/graph.py "What is the Nyquist rate and why does it matter?"

# 2) the demo
streamlit run src/app.py

# 3) the evaluation
python eval/run_eval.py
python eval/improvement_experiment.py
```

### Optional: turn on a real LLM and real tracing

Copy `.env.example` to `.env` and set any of:

| Variable | Effect |
|---|---|
| `GROQ_API_KEY` | generation via `openai/gpt-oss-120b` (free tier) |
| `GOOGLE_API_KEY` | generation via `gemini-2.0-flash` (free tier) |
| `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` | remote traces in the `LANGSMITH_PROJECT` |

Precedence is Groq -> Gemini -> offline. Everything, including the evaluation, runs without any of
them.

## Deploy (Streamlit Community Cloud)

The app is deploy-ready: `src/app.py` mirrors host secrets into the environment and **builds the FAISS
index itself** on first run, so nothing has to be committed.

1. Push this repo to GitHub (already at
   <https://github.com/crahulece2028-lab/week4-study-assistant-agent>).
2. Go to <https://share.streamlit.io> -> **New app** -> pick the repo/branch (`main`).
3. **Main file path:** `src/app.py`.
4. **Advanced settings -> Secrets** - paste (TOML):

   ```toml
   GROQ_API_KEY = "gsk_..."          # optional: enables the hosted LLM
   LANGSMITH_API_KEY = "lsv2_pt_..."  # optional: remote traces
   LANGSMITH_TRACING = "true"
   LANGSMITH_PROJECT = "week4-study-assistant"
   ```

   With no secrets the app still runs on the offline extractive model.
5. Deploy. The first load shows "building the FAISS index...", then serves the chat UI.

After it is live, replace `_pending deployment_` with the URL in `README.md`, section 5 of
`docs/4.4_capstone_report.md`, and the `LIVE_DEMO_URL` constant in `docs/build_report.py` (then
`python docs/build_report.py`).

**Hugging Face Spaces** works the same way: create a Streamlit-SDK Space, set **app file** to
`src/app.py`, and put the same keys in **Settings -> Variables and secrets**.

## Deliverables

| Task | Deliverable | Where |
|---|---|---|
| 4.1 | RAG prototype + architecture notes | `src/ingest.py`, `src/rag.py`, `src/store.py`, `docs/4.1_rag_architecture.md` |
| 4.2 | Multi-step workflow + diagram | `src/graph.py`, `src/tools.py`, `docs/4.2_langgraph_workflow.md`, `docs/4.2_workflow.png` |
| 4.3 | Evaluation report, traces, ablations | `eval/`, `docs/4.3_evaluation.md`, `docs/traces/` |
| 4.4 | Demo, report, slides | `src/app.py`, `docs/4.4_capstone_report.md`, `docs/4.4_slides.html` |

## Evaluation at a glance

22-question labelled set, both configurations:

| metric | offline | llm (Groq `gpt-oss-20b`) |
|---|---|---|
| routing | 100% | 100% |
| retrieval hit | 100% | 100% |
| answer keyword | 93% | 100% |
| tool | 100% | 100% |
| refusal (out-of-scope) | 67% | 100% |
| latency (median) | 56 ms | 4.4 s |

The 100%s are a ceiling from a small self-authored set; the honest story is in the two that are not
perfect (and which the hosted model fixes). Both failures are analysed in `docs/4.3_evaluation.md`:

- **q20 Friis equation** - a question that *sounds* in-domain gets a fluent but wrong cited answer,
  because relevance is graded lexically. This is the case for a generative grader.
- **q22 FM bandwidth** - retrieval was correct; the extractive generator quoted the adjacent
  sentence. A generation-ranking failure, not a retrieval one.

Two ablations back the design choices: chunk size (hit@4 is saturated; 800/150 keeps the sharpest
non-degenerate default) and the low-coverage retry floor (8 -> 4 nodes and 49 -> 22 ms on
unanswerable questions, same correctness).

## Repository layout

```
week4-study-assistant-agent/
  data/build_corpus.py      generates the 8-source corpus (reportlab)
  src/
    config.py               paths + backend selection (Groq/Gemini/offline)
    embeddings.py           all-MiniLM-L6-v2 wrapped as a LangChain Embeddings
    store.py                FAISS save/load via byte serialisation (Unicode-safe paths)
    ingest.py               load -> chunk -> embed -> index        (task 4.1)
    rag.py                  retrieve_docs / retrieve_pairs          (task 4.1)
    llm.py                  AgentLLM: chat + offline extractive     (tasks 4.1/4.2)
    tools.py                calculator (safe AST) + formula_lookup
    graph.py                the LangGraph state machine             (task 4.2)
    app.py                  Streamlit demo                          (task 4.4)
  eval/
    testset.json            22 labelled questions
    run_eval.py             scoring + trace capture                 (task 4.3)
    langsmith_eval.py       dataset + evaluators on the hosted suite (task 4.3)
    local_tracer.py         state trace -> waterfall PNG
    improvement_experiment.py  chunk-size + retry-policy ablations
  docs/                     4.1-4.4 write-ups, diagram, traces
  index/                    built FAISS store (git-ignored, auto-built on first run)
```

## Design decisions worth stating

- **Decisions, not prompts, behind the LLM interface.** `AgentLLM` exposes `route / grade / rewrite /
  generate`. The graph does not know or care whether those are answered by a 70B model or a local
  lexical scorer, which is what makes a no-key run a real system rather than a stub.
- **Offline generation is extractive and cited.** Answers are quoted from the retrieved context, so
  they cannot hallucinate - a deliberate choice for an evaluation target, not a placeholder.
- **Tools are offline.** The calculator walks a restricted AST instead of calling `eval`; the formula
  lookup is a curated, sourced table. Adding a web-search tool is a one-line entry in `TOOLS`, left
  out to keep the evaluation reproducible.
- **Unicode-safe persistence.** FAISS's C++ file API cannot open this project's non-ASCII path, so
  `store.py` serialises to bytes and lets Python do the file I/O.

## Limitations

Small self-authored corpus and test set; a lexical relevance grader; an extractive (not generative)
default answerer; single-run measurements. Each is called out, with consequences, in
`docs/4.3_evaluation.md`.
