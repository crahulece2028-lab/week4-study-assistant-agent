"""Task 4.3: the LangSmith-native evaluation.

    python eval/langsmith_eval.py --config offline     # free / deterministic
    python eval/langsmith_eval.py --config llm          # Groq-backed (spends tokens)

This uploads `eval/testset.json` as a LangSmith **dataset** and runs the agent once per example as
a LangSmith **experiment**, scoring it with the same metric definitions `run_eval.py` uses
(routing, retrieval hit, answer keyword, out-of-scope refusal, grounded confidence) plus an
optional LLM-as-judge groundedness evaluator.

The two evaluation paths are deliberately redundant: `run_eval.py` writes the detailed per-question
JSON the report reads, and this script produces the hosted experiment you can open in the LangSmith
UI. Both derive the label from the same test set and reuse the same helpers, so they cannot disagree
about what a metric means.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # so we can import run_eval's helpers
sys.path.insert(0, str(HERE.parent / "src"))

import config  # noqa: E402

TRACED = config.enable_langsmith()

import run_eval  # noqa: E402  (also imports graph, sets up the SRC path)
import graph  # noqa: E402

DATASET_NAME = "week4-study-assistant-eval"


def build_dataset(client) -> str:
    """Create the dataset from testset.json if it is missing; return its id."""
    try:
        dataset = client.create_dataset(
            DATASET_NAME,
            description="22-question labelled set for the ECE study assistant (routing, retrieval, "
                        "tool use, refusal). Mirrors eval/testset.json.")
    except Exception as exc:  # noqa: BLE001  (409 already-exists)
        if "already exists" not in str(exc).lower() and "409" not in str(exc):
            raise
        dataset = client.read_dataset(dataset_name=DATASET_NAME)

    existing = list(client.list_examples(dataset_id=dataset.id, limit=1))
    if not existing:
        testset = run_eval.load_testset()
        client.create_examples(
            dataset_id=dataset.id,
            examples=[{
                "inputs": {"id": q["id"], "question": q["question"]},
                "outputs": {"route": q["route"], "source": q.get("source"),
                            "keywords": q.get("keywords", []), "in_scope": q.get("in_scope", True)},
            } for q in testset["questions"]],
        )
    return dataset.id


def make_target(tag: str):
    def target(inputs: dict) -> dict:
        final = graph.answer(
            inputs["question"], thread_id=f"ls-{inputs['id']}",
            tags=[config.LANGSMITH_PROJECT, tag],
            metadata={"config": tag, "question_id": inputs["id"], "source": "langsmith_experiment"},
            run_name=f"ls-{inputs['id']}",
        )
        return {
            "answer": final.get("answer", "") or "",
            "route": final.get("route"),
            "sources": final.get("sources", []) or [],
            "confidence": final.get("confidence"),
            "tool_ok": final.get("tool_ok", False),
            "n_spans": len(final.get("trace", [])),
        }
    return target


# ------------------------------------------------------------------------------- evaluators
def _out(run) -> dict:
    return run.outputs or {}


def routing(run, example) -> dict:
    ref = example.outputs or {}
    return {"key": "routing", "score": float(_out(run).get("route") == ref.get("route"))}


def retrieval_hit(run, example) -> dict:
    ref = example.outputs or {}
    if ref.get("in_scope") is False or ref.get("route") != "retrieve":
        return {"key": "retrieval_hit", "score": None}
    return {"key": "retrieval_hit",
            "score": float(ref.get("source") in (_out(run).get("sources") or []))}


def answer_keyword(run, example) -> dict:
    ref = example.outputs or {}
    if ref.get("in_scope") is False or ref.get("route") != "retrieve":
        return {"key": "answer_keyword", "score": None}
    low = run_eval.normalise(_out(run).get("answer", "")).lower()
    hit = any(run_eval.normalise(k).lower() in low for k in ref.get("keywords", []))
    return {"key": "answer_keyword", "score": float(hit)}


def refusal(run, example) -> dict:
    ref = example.outputs or {}
    if ref.get("in_scope") is not False:
        return {"key": "refusal", "score": None}
    low = run_eval.normalise(_out(run).get("answer", "")).lower()
    return {"key": "refusal", "score": float(any(m in low for m in run_eval.REFUSAL_MARKERS))}


def grounded_confidence(run, example) -> dict:
    ref = example.outputs or {}
    if ref.get("in_scope") is False or ref.get("route") != "retrieve":
        return {"key": "grounded_confidence", "score": None}
    return {"key": "grounded_confidence", "score": float(_out(run).get("confidence") == "grounded")}


def make_judge():
    """LLM-as-judge groundedness: is the answer supported by the retrieved context?

    Costs model tokens, so it is opt-in via `--judge`.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    model = run_eval.graph.get_graph()  # noqa: F841  (ensure graph built)

    from llm import get_llm  # noqa: PLC0415
    llm = get_llm()
    if getattr(llm, "name", "").startswith("offline"):
        raise SystemExit("--judge needs a real model; the offline config has no judge.")

    def judge(run, example) -> dict:
        out = _out(run)
        if not out.get("sources"):
            return {"key": "groundedness_judge", "score": None}
        prompt = (f"Question: {example.inputs.get('question')}\n\n"
                  f"Answer: {out.get('answer')}\n\n"
                  "Is every factual claim in the answer supported by the question's topic in the "
                  "ECE lecture notes? Reply with a single word: YES or NO.")
        resp = llm.model.invoke([SystemMessage(content="You are a strict groundedness judge."),
                                 HumanMessage(content=prompt)])
        text = getattr(resp, "content", str(resp)).strip().lower()
        return {"key": "groundedness_judge", "score": float(text.startswith("yes"))}
    return judge


def main() -> None:
    parser = argparse.ArgumentParser(description="LangSmith dataset evaluation (task 4.3).")
    parser.add_argument("--config", choices=["offline", "llm"], default=None,
                        help="which backend to evaluate; default = whatever the environment selects.")
    parser.add_argument("--judge", action="store_true",
                        help="add the LLM-as-judge groundedness evaluator (spends tokens).")
    parser.add_argument("--prefix", default=None, help="LangSmith experiment name prefix.")
    args = parser.parse_args()

    tag = args.config or ("offline" if config.llm_mode() == "offline" else "llm")
    if tag == "offline":
        # Force the offline backend regardless of ambient keys.
        config.GROQ_API_KEY = ""
        import os
        os.environ["GROQ_API_KEY"] = ""
        import llm as llm_mod
        graph._GRAPH = graph.build_graph(model=llm_mod.OfflineLLM())
    else:
        model = config.GROQ_MODEL if config.llm_mode() == "groq" else config.GOOGLE_MODEL
        print(f"config: {tag} (model {model})")

    from langsmith import Client, evaluate

    if not TRACED:
        raise SystemExit("LANGSMITH_API_KEY / LANGSMITH_TRACING not set; nothing to upload.")

    client = Client()
    dataset_id = build_dataset(client)

    evaluators = [routing, retrieval_hit, answer_keyword, refusal, grounded_confidence]
    if args.judge:
        evaluators.append(make_judge())

    results = evaluate(
        make_target(tag),
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=args.prefix or f"{tag}-eval",
        client=client,
        max_concurrency=1,
        metadata={"config": tag, "project": config.LANGSMITH_PROJECT},
    )

    print(f"\nLangSmith experiment uploaded to project '{config.LANGSMITH_PROJECT}'")
    print(f"dataset: {dataset_id}")
    try:
        df = results.to_pandas()
        summary = df.select_dtypes("number").mean(numeric_only=True).round(3)
        print("mean scores across evaluators:")
        print(summary.to_string())
    except Exception as exc:  # noqa: BLE001
        print(f"(could not summarise results client-side: {exc})")


if __name__ == "__main__":
    main()
