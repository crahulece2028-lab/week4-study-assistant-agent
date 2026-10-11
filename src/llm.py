"""The LLM layer, behind one interface with two implementations.

`AgentLLM` exposes the four *decisions* the graph needs (route, grade, rewrite, generate) rather
than a generic `invoke`. That choice is what makes the offline path real rather than a stub: the
graph calls `llm.grade(...)`, and whether that is answered by a 70B model over a prompt or by a
local lexical scorer is invisible to the graph.

  - `ChatAgentLLM`   uses Groq or Gemini when an API key is present.
  - `OfflineLLM`     deterministic, grounded, no network. Used for CI and for this submission's
                     own evaluation run, and it is honest about being extractive rather than
                     generative.

`get_llm()` picks between them from `config.llm_mode()`.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

import config

# ---------------------------------------------------------------------------- data shapes
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "to", "in", "on",
    "for", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those", "it", "its",
    "as", "at", "by", "with", "from", "into", "about", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "does", "do", "did", "can", "could", "should", "would", "will",
    "shall", "may", "might", "must", "have", "has", "had", "you", "your", "i", "we", "they", "he",
    "she", "them", "his", "her", "our", "us", "me", "my", "give", "tell", "explain", "define",
    "describe", "state", "list", "name", "mean", "means", "meaning", "please", "kindly",
}

WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-\+]*")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
EXPR_CHARS = r"[\d\.\s\+\-\*/\^\(\)%pie\u03c0]"
EXPR_RE = re.compile(rf"[\d\.\(]{EXPR_CHARS}*[\d\)]")
CALC_WORDS = {"calculate", "compute", "evaluate", "solve", "sum", "product", "divide", "add",
              "subtract", "multiply", "times", "plus", "minus", "squared", "sqrt", "root"}
FORMULA_WORDS = {"formula", "equation", "expression"}


def content_words(text: str) -> list[str]:
    toks = WORD_RE.findall(text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    parts = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    return [p for p in parts if len(p) > 30]


@dataclass
class Route:
    route: str                     # "retrieve" | "tool" | "direct"
    reason: str
    tool_name: str | None = None
    tool_input: str | None = None


@dataclass
class Grade:
    relevant: bool
    score: float
    reason: str


@dataclass
class LLMResult:
    text: str
    meta: dict = field(default_factory=dict)


class AgentLLM(Protocol):
    name: str

    def route(self, question: str) -> Route: ...
    def grade(self, question: str, docs: list[tuple[str, str]]) -> Grade: ...
    def rewrite(self, question: str, docs: list[tuple[str, str]]) -> str: ...
    def generate(self, question: str, docs: list[tuple[str, str]], route: Route,
                 tool_result: str | None = None, low_confidence: bool = False) -> str: ...


# ------------------------------------------------------------------- offline implementation
class OfflineLLM:
    """Deterministic, grounded, network-free.

    Generation is *extractive*: it selects the sentences from the retrieved context that best
    match the question and returns them with a citation. That is a real RAG answer for a
    notes-QA task (the facts are in the context; nothing is invented) and it cannot hallucinate,
    which makes it a stable target for the evaluation and the failure analysis.
    """

    name = "offline-extractive"

    # --- routing -------------------------------------------------------------
    def route(self, question: str) -> Route:
        q = question.strip()
        low = q.lower()
        tokens = set(content_words(q))

        if low in {"hi", "hello", "hey", "thanks", "thank you", "help"} or \
                low.startswith(("who are you", "what can you do", "what are you")):
            return Route("direct", "greeting or meta-question, answered without retrieval")

        expr = self._extract_expression(q)
        if expr is not None:
            return Route("tool", "an arithmetic expression was detected, deferring to the "
                                 "calculator tool rather than the notes",
                         tool_name="calculator", tool_input=expr)

        if tokens & FORMULA_WORDS or {"formula", "equation"} & set(WORD_RE.findall(low)):
            hit = None
            for key in ("nyquist", "shannon", "settling", "overshoot", "pid", "vswr",
                        "reflection", "cutoff", "modulation", "bandwidth", "fft", "zener",
                        "transconductance", "damping", "quarter-wave", "carson"):
                if key in low:
                    hit = key
                    break
            if hit:
                return Route("tool", f"question asks for a formula or constant ({hit}); the "
                                     "formula tool returns the canonical expression",
                             tool_name="formula_lookup", tool_input=hit)

        return Route("retrieve", "in-scope question; answer from the notes via retrieval")

    @staticmethod
    def _extract_expression(question: str) -> str | None:
        """Pull a pure arithmetic expression out of a question, if there is exactly one.

        `pi` and `e` are allowed as symbols and resolved by the calculator tool, so
        "calculate 2*pi*50*0.001" survives intact instead of being cut at the first letter.
        """
        cleaned = question
        for w in CALC_WORDS:
            cleaned = re.sub(rf"\b{w}\b", " ", cleaned, flags=re.I)
        for cand in EXPR_RE.findall(cleaned):
            cand = cand.strip()
            if re.fullmatch(rf"{EXPR_CHARS}+", cand) and re.search(r"\d", cand) and \
                    re.search(r"[\+\-\*/\^]", cand):
                return cand
        return None

    # --- grading -------------------------------------------------------------
    def grade(self, question: str, docs: list[tuple[str, str]]) -> Grade:
        if not docs:
            return Grade(False, 0.0, "retrieved nothing to grade")
        q = set(content_words(question))
        if not q:
            return Grade(True, 1.0, "no content words to grade on; passing through")
        haystack = " ".join(w for text, _ in docs for w in content_words(text))
        present = {t for t in q if t in haystack}
        coverage = len(present) / len(q)
        missing = sorted(q - present)
        relevant = coverage >= 0.4
        reason = (f"question-term coverage {coverage:.0%}"
                  + (f"; missing {', '.join(missing[:5])}" if missing else ""))
        return Grade(relevant, coverage, reason)

    # --- query rewriting -----------------------------------------------------
    def rewrite(self, question: str, docs: list[tuple[str, str]]) -> str:
        """Keyword-ise the question, and bias toward a topic seen in the failed retrieval.

        The point of the retry is to change the embedding, so the rewrite strips the question
        scaffolding ("what is", "explain") which otherwise dominates the vector.
        """
        words = content_words(question)
        seen: list[str] = []
        for text, _ in docs:
            for w in content_words(text):
                if w not in words and w not in seen:
                    seen.append(w)
        rewritten = " ".join(words[:8] + seen[:4])
        return rewritten or question

    # --- generation ----------------------------------------------------------
    def generate(self, question: str, docs: list[tuple[str, str]], route: Route,
                 tool_result: str | None = None, low_confidence: bool = False) -> str:
        if route.route == "direct":
            return ("I am the Study Assistant for a set of eight ECE lecture-note PDFs (signals, "
                    "digital electronics, control systems, communications, semiconductors, "
                    "transmission lines, microprocessors and DSP). Ask me a question about those "
                    "topics and I will answer from the notes with citations.")
        if route.route == "tool" and tool_result:
            return tool_result
        if not docs or low_confidence:
            closest = ""
            if docs:
                best = self._rank_sentences(question, docs[:1], top=1)
                if best:
                    closest = f" The closest passage I found was: \u201c{best[0][1]}\u201d"
            return ("I could not find this in the provided lecture notes, so I will not guess at "
                    "an answer." + closest)

        ranked = self._rank_sentences(question, docs, top=3)
        if not ranked:
            return "I could not find this in the provided lecture notes."
        body = " ".join(s for _, s, _ in ranked)
        sources = sorted({src for _, _, src in ranked})
        return f"{body} [source: {', '.join(sources)}]"

    @staticmethod
    def _rank_sentences(question: str, docs: list[tuple[str, str]],
                        top: int = 3) -> list[tuple[float, str, str]]:
        """TF-IDF-flavoured sentence selection over the retrieved context."""
        q = content_words(question)
        qset = set(q)
        if not qset:
            return []

        sentences: list[tuple[str, str]] = []
        for text, src in docs:
            for s in split_sentences(text):
                sentences.append((s, src))
        if not sentences:
            return []

        n = len(sentences)
        df: dict[str, int] = {}
        tokenised: list[set[str]] = []
        for s, _ in sentences:
            toks = set(content_words(s))
            tokenised.append(toks)
            for t in toks:
                df[t] = df.get(t, 0) + 1

        scored: list[tuple[float, str, str]] = []
        for (s, src), toks in zip(sentences, tokenised):
            score = 0.0
            for t in qset & toks:
                score += math.log(1 + n / (1 + df.get(t, 0)))
            if score > 0:
                scored.append((score / math.sqrt(len(toks) + 1), s, src))
        scored.sort(key=lambda x: -x[0])

        picked: list[tuple[float, str, str]] = []
        seen: set[str] = set()
        for score, s, src in scored:
            key = s[:60]
            if key in seen:
                continue
            seen.add(key)
            picked.append((score, s, src))
            if len(picked) >= top:
                break
        return picked


# --------------------------------------------------------------- prompt implementation
ROUTE_SYS = """You are the router in a study-assistant agent whose answers MUST come from a fixed
set of eight ECE lecture-note PDFs (signals & systems, digital electronics, control systems,
communications, semiconductors, transmission lines, microprocessors, DSP). The agent cites the
notes; it must never answer from its own knowledge.

Choose exactly one route:
- DIRECT: ONLY a greeting, thanks, or a meta-question about the assistant itself
  (e.g. "hi", "thanks", "who are you", "what can you do"). Nothing else goes here.
- TOOL: the question is a bare arithmetic expression to evaluate, OR the question literally
  contains the word "formula" / "equation" / "expression" AND names one of these constants the
  formula tool knows: nyquist, shannon, settling, overshoot, pid, vswr, reflection, cutoff,
  modulation, bandwidth, fft, zener, transconductance, damping, quarter-wave, carson.
  Do NOT route to TOOL merely because the topic has a formula: only an explicit formula request
  or a bare calculation qualifies. "What is X?" / "How is X computed?" stay RETRIEVE.
- RETRIEVE: every other substantive question. Choose RETRIEVE for any question about the note
  subjects *even if you already know the answer* — the agent must answer from the notes with a
  citation, not from memory. Also choose RETRIEVE for substantive questions that fall outside
  these subjects (the relevance grader will reject those and the agent will decline).

Only arithmetic expressions and explicit formula requests (with a key from the list above) go to
TOOL. Never route a substantive question to DIRECT.

Reply in exactly this format, nothing else:
ROUTE: <RETRIEVE|TOOL|DIRECT>
TOOL: <calculator|formula_lookup|none>
INPUT: <expression or lookup key, or none>
REASON: <one short sentence>"""

GRADE_SYS = """You are a relevance grader in a RAG system. Given a question and retrieved
passages, decide whether the passages actually contain the information needed to answer.
Reply with JSON only: {"relevant": true|false, "reason": "<short reason>"}"""

REWRITE_SYS = """Rewrite the user's question into a short keyword query that will retrieve better
from a vector store of ECE lecture notes. Drop question scaffolding. Reply with the query only."""

GENERATE_SYS = """You are a study assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say so plainly and do not invent facts. Always name
the source document you used. Be concise (2-4 sentences)."""


def _format_context(docs: list[tuple[str, str]], limit: int = 6) -> str:
    return "\n\n".join(f"[{src}] {text}" for text, src in docs[:limit])


class ChatAgentLLM:
    """Prompt-driven implementation backed by a real chat model."""

    def __init__(self, model):
        self.model = model
        self.name = f"{model.__class__.__name__}"

    def _ask(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        resp = self.model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return getattr(resp, "content", str(resp)).strip()

    def route(self, question: str) -> Route:
        out = self._ask(ROUTE_SYS, question)
        route = "retrieve"
        tool, tool_input, reason = None, None, ""
        for line in out.splitlines():
            low = line.lower()
            if low.startswith("route:"):
                v = line.split(":", 1)[1].strip().upper()
                route = {"RETRIEVE": "retrieve", "TOOL": "tool", "DIRECT": "direct"}.get(v, "retrieve")
            elif low.startswith("tool:"):
                v = line.split(":", 1)[1].strip()
                tool = None if v.lower() in {"none", ""} else v
            elif low.startswith("input:"):
                v = line.split(":", 1)[1].strip()
                tool_input = None if v.lower() in {"none", ""} else v
            elif low.startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
        return Route(route, reason or "model routing decision", tool, tool_input)

    def grade(self, question: str, docs: list[tuple[str, str]]) -> Grade:
        if not docs:
            return Grade(False, 0.0, "retrieved nothing to grade")
        out = self._ask(GRADE_SYS, f"Question: {question}\n\nPassages:\n{_format_context(docs)}")
        try:
            blob = out[out.find("{"): out.rfind("}") + 1]
            data = json.loads(blob)
            relevant = bool(data.get("relevant", False))
            return Grade(relevant, 1.0 if relevant else 0.0, str(data.get("reason", ""))[:160])
        except Exception:
            relevant = "true" in out.lower()
            return Grade(relevant, 1.0 if relevant else 0.0, out[:160])

    def rewrite(self, question: str, docs: list[tuple[str, str]]) -> str:
        out = self._ask(REWRITE_SYS, question)
        return out.splitlines()[0].strip()[:200] or question

    def generate(self, question: str, docs: list[tuple[str, str]], route: Route,
                 tool_result: str | None = None, low_confidence: bool = False) -> str:
        if route.route == "direct":
            return self._ask(GENERATE_SYS, question)
        if route.route == "tool" and tool_result:
            return tool_result
        if not docs:
            return "I could not find this in the provided lecture notes."
        user = (f"Question: {question}\n\nContext:\n{_format_context(docs)}"
                + ("\n\nNote: retrieval confidence is low; say so if the context is insufficient."
                   if low_confidence else ""))
        return self._ask(GENERATE_SYS, user)


def get_llm() -> AgentLLM:
    mode = config.llm_mode()
    if mode == "groq":
        from langchain_groq import ChatGroq
        return ChatAgentLLM(ChatGroq(model=config.GROQ_MODEL, temperature=0,
                                     api_key=config.GROQ_API_KEY))
    if mode == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatAgentLLM(ChatGoogleGenerativeAI(model=config.GOOGLE_MODEL, temperature=0,
                                                   google_api_key=config.GOOGLE_API_KEY))
    return OfflineLLM()
