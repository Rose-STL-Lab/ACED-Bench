"""
agent_modular.py

The Modular agent for causal discovery.

The agent is split into four specialized LLM calls per outer turn:

    INIT     (once, on turn 1)      → hypothesis, verification criteria, plan
    CODE     (inner loop, ≤5 rounds) → Python code that analyzes the data
    ANALYSIS (once per turn with data) → interpret results, state confidence, decide
    DESIGN   (once per turn, if continuing) → next experiment as a query string

Interface to the orchestrator is unchanged:
    agent.initialize(world_info, question, max_queries)
    action = agent.get_next_action()
    agent.receive_result(result)

Turn sequencing inside get_next_action():
    Turn 1 (no data):       INIT → DESIGN → return query
    Turn N (has new data):  CODE loop → ANALYSIS → (answer) or DESIGN → return
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
from scipy.stats import chi2_contingency

from schemas import Question, QueryResult, WorldInfo

logger = logging.getLogger(__name__)

MAX_CODE_ROUNDS = 5          # max code actions per CODE inner loop
MAX_CODE_OUTPUT_CHARS = 3000  # truncate long outputs
CODE_TIMEOUT_SECONDS = 45    # spawn startup + fresh pandas import eats ~2 s; leaves ≥40 s for user code
# Cross-round state cap: drop DataFrames / large arrays and any var >1 MiB pickled.
# Small scalars / dicts / summary results still flow across rounds.
_MAX_VAR_PICKLE_BYTES = 1_000_000
_MAX_NDARRAY_ELEMS = 10_000

HIGH_CONFIDENCE_THRESHOLD = 85  # confidence threshold the model targets before answering
MAX_SAMPLES_PER_QUERY = 10_000  # enforced by world_model validation

_INJECTED_NAMES = frozenset({"pd", "np", "stats", "chi2_contingency", "query_files"})

# -----------------------------------------------------------------------------
# Regex helpers
# -----------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Strip markdown code fences (anchored to start-of-line so ``` inside a string literal
# is left alone). Matches opening fences like ```python / ```py / ```; the trailing
# .replace("```", "") in _execute_code strips closing fences.
_FENCE_RE = re.compile(r"^```[a-z]*\n?", re.IGNORECASE | re.MULTILINE)

# CODE-phase extraction: accept <code>...</code>, or a ```python``` fenced block,
# or treat the whole response as code if nothing else matches.
_CODE_TAG_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_ANALYSIS_READY_RE = re.compile(r"<analysis_ready\s*/?>", re.IGNORECASE)

# ANALYSIS-phase extraction
_CONFIDENCE_RE = re.compile(r"<confidence>\s*([0-9]{1,3})\s*%?\s*</confidence>", re.IGNORECASE)
_DECISION_RE = re.compile(r"<decision>\s*(continue|answer|give_up)\s*</decision>", re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_MEMORY_RE = re.compile(r"<memory_update>\s*(.*?)\s*</memory_update>", re.DOTALL | re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"<evidence_summary>\s*(.*?)\s*</evidence_summary>", re.DOTALL | re.IGNORECASE)
_CONF_REASON_RE = re.compile(r"<confidence_reasoning>\s*(.*?)\s*</confidence_reasoning>", re.DOTALL | re.IGNORECASE)
_HYPOTHESIS_REVISION_RE = re.compile(r"<hypothesis_revision>\s*(.*?)\s*</hypothesis_revision>", re.DOTALL | re.IGNORECASE)

# DESIGN-phase extraction
_QUERY_TAG_RE = re.compile(r"<query>\s*(.*?)\s*</query>", re.DOTALL | re.IGNORECASE)
_RATIONALE_RE = re.compile(r"<rationale>\s*(.*?)\s*</rationale>", re.DOTALL | re.IGNORECASE)

# INIT-phase extraction
_INIT_GUESS_RE = re.compile(r"<initial_guess>\s*(.*?)\s*</initial_guess>", re.DOTALL | re.IGNORECASE)
_STRATEGY_RE = re.compile(r"<strategy>\s*(.*?)\s*</strategy>", re.DOTALL | re.IGNORECASE)
_HYPOTHESIS_RE = re.compile(r"<hypothesis>\s*(.*?)\s*</hypothesis>", re.DOTALL | re.IGNORECASE)
_CRITERIA_RE = re.compile(r"<verification_criteria>\s*(.*?)\s*</verification_criteria>", re.DOTALL | re.IGNORECASE)
_PLAN_RE = re.compile(r"<experiment_plan>\s*(.*?)\s*</experiment_plan>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _format_interventions(conditions: Optional[List[Dict[str, Any]]]) -> str:
    """Format a list of intervention dicts as 'do(A=a) | do(B=b)'. Empty → ''."""
    if not conditions:
        return ""
    return " | ".join(
        "do(" + ", ".join(f"{k}={v}" for k, v in c.items()) + ")"
        for c in conditions
    )


# -----------------------------------------------------------------------------
# Structured logging helpers
# -----------------------------------------------------------------------------

_BAR = "═" * 78
_SUB = "─" * 78


def _log_banner(title: str, lines: Optional[List[str]] = None) -> None:
    logger.info(_BAR)
    logger.info(f"  {title}")
    logger.info(_BAR)
    if lines:
        for line in lines:
            logger.info(f"  {line}")
        logger.info(_SUB)


def _log_phase(phase: str, turn: int, extra: str = "") -> None:
    suffix = f"  [{extra}]" if extra else ""
    logger.info(f"{_SUB}")
    logger.info(f"  ▶ TURN {turn} — PHASE: {phase}{suffix}")
    logger.info(f"{_SUB}")


def _has_python_error(output: str) -> bool:
    return "[PYTHON ERROR]" in output or output.startswith("[TIMEOUT")


def _indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


# -----------------------------------------------------------------------------
# Execution namespace
# -----------------------------------------------------------------------------

class _SafeQueryDict(dict):
    """
    Read-only dict injected as `query_files` into the exec namespace.

    Raises a clear TypeError if exec'd code tries to assign to it
    (e.g. `query_files[1] = 'foo.csv'` would silently overwrite the real path).
    Internal population from _build_exec_namespace uses dict.__setitem__ directly
    to bypass the guard.
    """

    def __setitem__(self, key, value):
        if key in self:
            raise TypeError(
                f"query_files is read-only — do not assign to it. "
                f"Use query_{key}_csv (already pre-loaded) to read the file."
            )
        raise TypeError(
            f"query_files is read-only and query {key} has not been received yet. "
            f"Available query numbers: {sorted(self.keys())}. "
            f"Request more data first via the DESIGN phase."
        )

    def __missing__(self, key):
        available = sorted(self.keys())
        raise KeyError(
            f"{key!r} not in query_files. "
            f"Available query numbers: {available}. "
            f"Only access files listed in AVAILABLE CSV FILES."
        )


def _code_worker(code, query_files_map, extra_vars, result_q):
    """Top-level (spawn-pickleable) worker. Runs the agent's code in a fresh Python process."""
    import os
    # Single-threaded BLAS — fast enough for per-round CSV work and avoids any
    # residual thread-pool weirdness when the child spins up.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import contextlib
    import io
    import pickle
    import traceback

    import numpy as _np
    import pandas as _pd
    import scipy.stats as _stats
    from scipy.stats import chi2_contingency as _chi2

    class _ChildSafeQueryDict(dict):
        def __setitem__(self, key, value):
            if key in self:
                raise TypeError(
                    f"query_files is read-only — do not assign to it. "
                    f"Use query_{key}_csv (already pre-loaded) to read the file."
                )
            raise TypeError(
                f"query_files is read-only and query {key} has not been received yet. "
                f"Available query numbers: {sorted(self.keys())}. "
                f"Request more data first via the DESIGN phase."
            )

        def __missing__(self, key):
            raise KeyError(
                f"{key!r} not in query_files. "
                f"Available query numbers: {sorted(self.keys())}. "
                f"Only access files listed in AVAILABLE CSV FILES."
            )

    qf = _ChildSafeQueryDict()
    ns = {
        "pd": _pd,
        "np": _np,
        "stats": _stats,
        "chi2_contingency": _chi2,
        "query_files": qf,
    }
    for qn, path in query_files_map.items():
        ns[f"query_{qn}_csv"] = path
        dict.__setitem__(qf, qn, path)
    ns.update(extra_vars)

    buf = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)  # noqa: S102
    except Exception:
        error = traceback.format_exc()

    output = buf.getvalue()
    if error:
        output += f"\n[PYTHON ERROR]\n{error}"

    new_vars: Dict[str, Any] = {}
    dropped: List[str] = []
    for k, v in ns.items():
        if k in _INJECTED_NAMES or k.startswith("__"):
            continue
        if k.startswith("query_") and k.endswith("_csv"):
            continue
        if isinstance(v, (_pd.DataFrame, _pd.Series)):
            dropped.append(k)
            continue
        if isinstance(v, _np.ndarray) and v.size > _MAX_NDARRAY_ELEMS:
            dropped.append(k)
            continue
        try:
            data = pickle.dumps(v)
        except Exception:
            continue
        if len(data) > _MAX_VAR_PICKLE_BYTES:
            dropped.append(k)
            continue
        new_vars[k] = v

    if dropped:
        output += (
            f"\n[NOTE: variables not carried to next round (too large — re-read / "
            f"recompute if needed): {', '.join(sorted(dropped))}]"
        )

    result_q.put((output, new_vars))


def _wait_for_result(proc, result_q, timeout: float):
    """Return the worker's (output, new_vars), or None if it did not deliver one in time.

    The queue is read while the worker runs: a result larger than the pipe buffer (about 64 KB) keeps the worker
    blocked until it is read, so waiting for the worker to exit first would report a finished run as a timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return result_q.get(timeout=0.2)
        except queue.Empty:
            if not proc.is_alive():
                try:
                    return result_q.get(timeout=1.0)
                except queue.Empty:
                    return None
    return None


def _execute_code(
    code: str,
    namespace: Dict[str, Any],
    timeout: int = CODE_TIMEOUT_SECONDS,
    max_chars: int = MAX_CODE_OUTPUT_CHARS,
) -> str:
    """Execute code in a spawned subprocess with a hard-killable timeout.
    New variables created by the code are propagated back to the namespace.

    Uses "spawn" (not "fork") so the child does not inherit the parent's CUDA context,
    tokenizers thread pool, or BLAS mutex state — all of which previously caused
    `pd.read_csv` to deadlock under the fork-based implementation.
    """
    import pickle

    code = _FENCE_RE.sub("", code).replace("```", "").strip()

    # Extract a picklable payload from the parent namespace. Module objects (pd / np /
    # stats / chi2_contingency) and the _SafeQueryDict are reconstructed in the child
    # instead of being pickled across. Large objects (DataFrames, big ndarrays, anything
    # >_MAX_VAR_PICKLE_BYTES) are dropped — they'd dominate spawn IPC time and the agent
    # is expected to re-read CSVs each round anyway.
    query_files_map: Dict[int, str] = {}
    extra_vars: Dict[str, Any] = {}
    for k, v in namespace.items():
        if k in _INJECTED_NAMES or k.startswith("__"):
            continue
        if k.startswith("query_") and k.endswith("_csv") and isinstance(v, str):
            try:
                qn = int(k[len("query_"):-len("_csv")])
                query_files_map[qn] = v
                continue
            except ValueError:
                pass
        if isinstance(v, (pd.DataFrame, pd.Series)):
            continue
        if isinstance(v, np.ndarray) and v.size > _MAX_NDARRAY_ELEMS:
            continue
        try:
            data = pickle.dumps(v)
        except Exception:
            continue
        if len(data) > _MAX_VAR_PICKLE_BYTES:
            continue
        extra_vars[k] = v

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(
        target=_code_worker,
        args=(code, query_files_map, extra_vars, result_q),
        daemon=True,
    )
    proc.start()
    result = _wait_for_result(proc, result_q, timeout)
    timed_out = result is None and proc.is_alive()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join()
    if timed_out:
        return f"[TIMEOUT: execution exceeded {timeout}s — possible infinite loop]"
    if result is None:
        return "[Process ended without output — possible crash]"
    output, new_vars = result

    namespace.update(new_vars)

    if not output.strip():
        output = "[No output — did you forget to print()?]"
    if len(output) > max_chars:
        output = output[:max_chars] + f"\n... [truncated at {max_chars} chars]"
    return output


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------

@dataclass
class ModularAgent:
    """
    Modular causal discovery agent with separated reasoning and coding phases.
    """
    llm: Any

    # Set by initialize()
    world_info: Optional[WorldInfo] = field(default=None, init=False)
    question: Optional[Question] = field(default=None, init=False)
    max_queries: int = field(default=10, init=False)

    # History / state
    _query_history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _queries_made: int = field(default=0, init=False)
    _system_messages: List[str] = field(default_factory=list, init=False)
    _has_new_data: bool = field(default=False, init=False)

    # Modular memory
    _hypothesis_doc: str = field(default="", init=False)   # from INIT; ANALYSIS may append revisions
    _experiment_memory: str = field(default="", init=False)  # written by ANALYSIS
    _last_confidence: Optional[int] = field(default=None, init=False)

    # Exposed for orchestrator compatibility
    _scientist_memory: str = field(default="", init=False)

    # Logging counters
    _turn_number: int = field(default=0, init=False)
    _total_code_rounds: int = field(default=0, init=False)
    _total_code_errors: int = field(default=0, init=False)
    _total_llm_calls: int = field(default=0, init=False)
    _llm_calls_this_turn: int = field(default=0, init=False)

    # -------------------------------------------------------------------------
    # Orchestrator interface
    # -------------------------------------------------------------------------

    def initialize(self, world_info: WorldInfo, question: Question,
                   max_queries: int) -> None:
        self.world_info = world_info
        self.question = question
        self.max_queries = max_queries
        self._query_history = []
        self._queries_made = 0
        self._system_messages = []
        self._has_new_data = False
        self._hypothesis_doc = ""
        self._experiment_memory = ""
        self._last_confidence = None
        self._scientist_memory = ""
        self._turn_number = 0
        self._total_code_rounds = 0
        self._total_code_errors = 0
        self._total_llm_calls = 0
        self._llm_calls_this_turn = 0
        _log_banner(
            "EXPERIMENT START",
            [
                f"Question:     {question.question_text}",
                f"Question type: {question.question_type}",
                f"Query budget: {max_queries} total queries",
                f"Variables:    {len(world_info.variables)}",
                f"Agent:        ModularAgent (INIT / CODE / ANALYSIS / DESIGN)",
                f"Max code rounds per turn: {MAX_CODE_ROUNDS}",
            ],
        )

    def receive_result(self, result: QueryResult) -> None:
        """Record a new query result and flag that fresh data is ready to analyze."""
        if result.success:
            self._queries_made += 1

        query_num = len(self._query_history) + 1
        preview = None
        if result.success and result.data_file:
            preview = self._data_preview(result.data_file)

        conditions = result.query.interventions if result.query else []
        self._query_history.append({
            "query_num": query_num,
            "query": result.query.raw_query,
            "success": result.success,
            "data_file": result.data_file,
            "n_rows": result.n_rows,
            "error_message": result.error_message,
            "interventions": conditions,
            "query_type": result.query.query_type.value if result.query else "unknown",
            "preview": preview,
            "data_summary": preview,  # orchestrator reads this field
        })
        # OR-accumulate: if multiple results arrive before the next get_next_action
        # (e.g. a successful one followed by a failure), keep the "has data" signal.
        if result.success and result.data_file is not None:
            self._has_new_data = True

        interv = _format_interventions(conditions) or "observational"
        logger.info(
            f"  RECEIVED result #{query_num}: success={result.success}, "
            f"n={result.n_rows}, type={interv}, "
            f"file={os.path.basename(result.data_file) if result.data_file else 'none'}"
        )

    def receive_system_message(self, message: str) -> None:
        self._system_messages.append(message)
        logger.info(f"System message: {message[:100]}...")

    def _system_alerts_block(self) -> str:
        if not self._system_messages:
            return ""
        return "SYSTEM ALERTS:\n" + "\n".join(
            f"  - {m}" for m in self._system_messages
        ) + "\n\n"

    # -------------------------------------------------------------------------
    # Main dispatch
    # -------------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Single call site for the scientist LLM — centralises counting + <think> strip."""
        raw = self.llm.generate_messages(messages)
        self._total_llm_calls += 1
        self._llm_calls_this_turn += 1
        return _strip_think(raw)

    def get_next_action(self) -> Dict[str, Any]:
        if self.world_info is None or self.question is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        self._turn_number += 1
        self._llm_calls_this_turn = 0
        _log_banner(
            f"TURN {self._turn_number} START",
            [
                f"Queries used: {self._queries_made}/{self.max_queries}",
                f"Has new data to analyze: {self._has_new_data}",
                f"Last confidence: "
                f"{self._last_confidence if self._last_confidence is not None else '—'}%",
                f"Total code rounds so far: {self._total_code_rounds} "
                f"(errors: {self._total_code_errors})",
            ],
        )

        transcript: List[Dict[str, Any]] = []
        code_rounds: List[Dict[str, str]] = []

        # Turn 1: one-time initialization
        if not self._hypothesis_doc:
            _log_phase("INIT", self._turn_number)
            init_resp = self._run_init()
            transcript.append({"phase": "init", "response": init_resp})
            logger.info(f"INIT produced hypothesis doc ({len(self._hypothesis_doc)} chars)")

        # If data is available, run CODE loop then ANALYSIS
        analysis_result: Optional[Dict[str, Any]] = None
        if self._has_new_data:
            _log_phase("CODE LOOP", self._turn_number,
                       f"max {MAX_CODE_ROUNDS} rounds")
            code_rounds, code_transcript = self._run_code_loop()
            transcript.extend(code_transcript)
            self._has_new_data = False  # consumed

            remaining = self.max_queries - self._queries_made
            _log_phase("ANALYSIS", self._turn_number,
                       f"budget_remaining={remaining}")
            analysis_result = self._run_analysis(code_rounds, budget_remaining=remaining)
            transcript.append({"phase": "analysis", "response": analysis_result["raw"]})

            if analysis_result["decision"] == "answer":
                return self._build_action(
                    "answer", analysis_result["answer"], transcript, code_rounds,
                    reasoning=analysis_result["evidence"],
                )
            if analysis_result["decision"] == "give_up":
                return self._build_action(
                    "give_up", analysis_result.get("answer") or "Insufficient evidence",
                    transcript, code_rounds,
                    reasoning=analysis_result["evidence"],
                )
            # else: decision == "continue" → fall through to DESIGN

        # If budget is exhausted and we didn't already answer, force an answer now
        if self._queries_made >= self.max_queries:
            _log_phase("ANALYSIS (forced answer — budget exhausted)", self._turn_number)
            forced = self._run_analysis(code_rounds, budget_remaining=0, force_answer=True)
            transcript.append({"phase": "analysis_forced", "response": forced["raw"]})
            return self._build_action(
                "answer", forced["answer"] or "Unable to determine",
                transcript, code_rounds,
                reasoning=forced["evidence"],
            )

        # DESIGN the next experiment
        _log_phase("DESIGN", self._turn_number,
                   f"budget_remaining={self.max_queries - self._queries_made}")
        design = self._run_design()
        transcript.append({"phase": "design", "response": design["raw"]})
        if not design["query"]:
            # Design failed to produce a query — give up rather than loop
            return self._build_action(
                "give_up", "Failed to design next experiment",
                transcript, code_rounds,
                reasoning=design["rationale"],
            )
        return self._build_action(
            "query", design["query"], transcript, code_rounds,
            reasoning=design["rationale"],
        )

    # -------------------------------------------------------------------------
    # Build the action dict returned to the orchestrator
    # -------------------------------------------------------------------------

    def _build_action(
        self,
        action_type: str,
        content: str,
        transcript: List[Dict[str, Any]],
        code_rounds: List[Dict[str, str]],
        reasoning: str = "",
    ) -> Dict[str, Any]:
        self._scientist_memory = self._compose_public_memory()
        last_raw = transcript[-1]["response"] if transcript else ""
        round_errors = sum(1 for r in code_rounds if _has_python_error(r.get("output", "")))
        phases = [e.get("phase", "unknown") for e in transcript]
        phase_summary = {
            "turn_number": self._turn_number,
            "phases_run": phases,
            "n_llm_calls": self._llm_calls_this_turn,
            "code_rounds_this_turn": len(code_rounds),
            "code_errors_this_turn": round_errors,
            "cumulative_code_rounds": self._total_code_rounds,
            "cumulative_code_errors": self._total_code_errors,
            "cumulative_llm_calls": self._total_llm_calls,
            "confidence_after_turn": self._last_confidence,
            "queries_used_so_far": self._queries_made,
            "queries_remaining": self.max_queries - self._queries_made,
            "action_type": action_type,
        }
        _log_banner(
            f"TURN {self._turn_number} END",
            [
                f"Action:          {action_type}",
                f"Content:         {str(content)[:120]}",
                f"Code rounds:     {len(code_rounds)} this turn "
                f"(errors: {round_errors}) | cumulative: "
                f"{self._total_code_rounds} (errors: {self._total_code_errors})",
                f"Confidence:      "
                f"{self._last_confidence if self._last_confidence is not None else '—'}%",
                f"Queries used:    {self._queries_made}/{self.max_queries}",
            ],
        )
        return {
            "type": action_type,
            "content": content,
            "raw_response": last_raw,
            "reasoning": reasoning,
            "scientist_memory": self._scientist_memory,
            "code_rounds": code_rounds,
            "llm_transcript": transcript,
            "confidence": self._last_confidence,
            "phase_summary": phase_summary,
        }

    def _compose_public_memory(self) -> str:
        """The single string exposed to the orchestrator / logs."""
        parts = []
        if self._hypothesis_doc:
            parts.append("=== HYPOTHESIS & PLAN ===\n" + self._hypothesis_doc)
        if self._experiment_memory:
            parts.append("=== EXPERIMENT MEMORY ===\n" + self._experiment_memory)
        if self._last_confidence is not None:
            parts.append(f"=== LAST CONFIDENCE: {self._last_confidence}% ===")
        return "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # CALL 1 — INIT
    # -------------------------------------------------------------------------

    _INIT_SYSTEM = """You are a domain-knowledgeable scientist about to start an investigation. Read the question carefully and decide \
for yourself what it is asking and what approach will answer it. Different questions call for different approaches; picking the wrong \
one wastes budget and can mislead.

Before any data is collected, you must:
  1. State a prior guess about the answer based on common-sense / domain knowledge.
     Do NOT anchor on this — the underlying ground truth can violate real-world
     intuition. Treat the prior as a starting point, not a verdict.
  2. Decide your STRATEGY for this question in your own words: what is being asked,
     what kind of data and analysis would actually answer it, and what pitfalls
     you should watch for.
  3. Form a concrete, testable hypothesis aligned with your strategy.
  4. Write evidential criteria for what would support or argue against it.
     Be explicit about what counts as strong vs. weak evidence, and remember that
     absence of evidence is not evidence of absence — make sure your criteria
     account for whether the data you'll collect can actually distinguish the
     possibilities.
  5. Sketch a plan that is just large enough to meet your verification criteria.
     Do not invent busywork to consume the budget; do not converge prematurely
     when evidence is borderline.

Use ONLY the following XML tags in your response, in order:

<initial_guess>Prior guess + 1-2 sentences of reasoning.</initial_guess>
<strategy>
  What is the question really asking: ...
  How you plan to answer it (what kind of data, what analysis, why): ...
  Key pitfalls and what could go wrong: ...
</strategy>
<hypothesis>One concrete, testable statement. Frame to match the strategy above.</hypothesis>
<verification_criteria>
  Strong evidence FOR: ...
  Strong evidence AGAINST: ...
  Ambiguous: ...
  Sanity checks you'll perform on the data: ...
</verification_criteria>
<experiment_plan>
  Step 1: ...
  Step 2: ...
  Step 3: ...
  Step 4 (confirmatory): ...
</experiment_plan>

Keep each block tight — a few sentences, not paragraphs."""

    def _run_init(self) -> str:
        user = self._init_user_prompt()
        messages = [
            {"role": "system", "content": self._INIT_SYSTEM},
            {"role": "user", "content": user},
        ]
        response = self._call_llm(messages)
        logger.debug(f"INIT raw response:\n{response}")

        guess = _first_group(_INIT_GUESS_RE, response)
        strategy = _first_group(_STRATEGY_RE, response)
        hypothesis = _first_group(_HYPOTHESIS_RE, response)
        criteria = _first_group(_CRITERIA_RE, response)
        plan = _first_group(_PLAN_RE, response)

        if guess:
            logger.info(f"  Initial guess:\n{_indent(guess, 4)}")
        if strategy:
            logger.info(f"  Strategy:\n{_indent(strategy, 4)}")
        if hypothesis:
            logger.info(f"  Hypothesis:\n{_indent(hypothesis, 4)}")
        if criteria:
            logger.info(f"  Verification criteria:\n{_indent(criteria, 4)}")
        if plan:
            logger.info(f"  Experiment plan:\n{_indent(plan, 4)}")

        # Fall back to raw response if the model didn't tag — still store for context
        if not (strategy or hypothesis or criteria or plan):
            logger.warning("INIT: no structured tags found; storing full response")
            self._hypothesis_doc = response
        else:
            self._hypothesis_doc = (
                f"Initial guess:\n{guess or '(none)'}\n\n"
                f"Strategy:\n{strategy or '(none)'}\n\n"
                f"Hypothesis:\n{hypothesis or '(none)'}\n\n"
                f"Verification criteria:\n{criteria or '(none)'}\n\n"
                f"Experiment plan:\n{plan or '(none)'}"
            )
        return response

    def _init_user_prompt(self) -> str:
        return (
            f"QUESTION: {self.question.question_text}\n\n"
            f"{self._system_alerts_block()}"
            f"DOMAIN CONTEXT: {self.world_info.story}\n\n"
            f"VARIABLES:\n{self.world_info.get_variable_catalog()}\n\n"
            f"{self._intervention_limits_block()}\n\n"
            f"TOOLS YOU WILL HAVE:\n"
            f"  - Observational samples: show how variables jointly behave under the\n"
            f"    natural data-generating process; reveal correlations.\n"
            f"  - Interventional samples do(X=v): set X to v while sampling, severing\n"
            f"    X's incoming edges; reveal the downstream effect of fixing X.\n\n"
            f"BUDGET: {self.max_queries} total queries. Plan only as many "
            f"experiments as you need to meet your verification criteria — extra "
            f"queries are available for confirmatory follow-ups on borderline results, "
            f"not for busywork.\n\n"
            f"Now produce your <initial_guess>, <strategy>, <hypothesis>, "
            f"<verification_criteria>, and <experiment_plan>."
        )

    # -------------------------------------------------------------------------
    # CALL 2 — CODE LOOP
    # -------------------------------------------------------------------------

    _CODE_SYSTEM = """You are a data analyst. Your ONLY job is to write Python code that tests a specific hypothesis against available CSV data.

You have up to 5 code rounds in this turn. Between rounds you will see stdout from your previous code.

Each response must be EXACTLY ONE of:

  (a) A single Python code block inside <code>...</code> tags.
      Available in the namespace:
        - pd (pandas), np (numpy), stats (scipy.stats), chi2_contingency
        - query_N_csv  — string path to CSV for query N
        - query_files  — dict {N: path} (read-only)
      DataFrames, Series, and large arrays do NOT carry across rounds — re-read the
      CSV each round, or print() any value you need later. Small scalars/dicts persist.
      Always print() what you want to see. No strategy, no plans, just code.

  (b) <analysis_ready/>  — emit this when you have gathered enough evidence.
      Do NOT include code in the same response as <analysis_ready/>.

Rules:
  - Do NOT simulate interventions in code (e.g. don't overwrite columns).
  - Interventional CSVs already contain do(X=v) samples — just read them.
  - If a previous round errored, fix the bug in the next round.
  - Keep each code block small and focused (one comparison / one question)."""

    def _run_code_loop(self) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """Run the inner code loop. Returns (code_rounds, transcript_entries)."""
        messages = [
            {"role": "system", "content": self._CODE_SYSTEM},
            {"role": "user", "content": self._code_user_prompt()},
        ]
        namespace = self._build_exec_namespace()
        code_rounds: List[Dict[str, str]] = []
        transcript: List[Dict[str, Any]] = []

        for round_idx in range(MAX_CODE_ROUNDS):
            round_num = round_idx + 1
            logger.info(f"  ── CODE round {round_num}/{MAX_CODE_ROUNDS} ──")
            response = self._call_llm(messages)
            logger.debug(f"  LLM code response:\n{response}")
            messages.append({"role": "assistant", "content": response})
            transcript.append({"phase": f"code_round_{round_num}", "response": response})

            if _ANALYSIS_READY_RE.search(response):
                logger.info(
                    f"  CODE loop complete: <analysis_ready/> after "
                    f"{len(code_rounds)} executed round(s)"
                )
                break

            code = _extract_code(response)
            if not code:
                logger.warning(
                    f"  Round {round_num}: no extractable code, no <analysis_ready/> "
                    f"— ending loop early"
                )
                break

            output = _execute_code(code, namespace)
            errored = _has_python_error(output)
            self._total_code_rounds += 1
            if errored:
                self._total_code_errors += 1
            code_rounds.append({"code": code, "output": output, "errored": errored})
            status = "ERRORED" if errored else "OK"
            logger.info(
                f"  Round {round_num} {status} | "
                f"code_lines={code.count(chr(10)) + 1} | "
                f"output_chars={len(output)}"
            )
            logger.info(f"  CODE:\n{_indent(code, 4)}")
            logger.info(f"  OUTPUT:\n{_indent(output, 4)}")

            messages.append({"role": "user", "content": f"[stdout]\n{output}"})

        errors_this_turn = sum(1 for r in code_rounds if r.get("errored"))
        logger.info(
            f"  CODE loop summary: {len(code_rounds)} executed round(s), "
            f"{errors_this_turn} error(s)"
        )
        return code_rounds, transcript

    def _code_user_prompt(self) -> str:
        files_block = self._files_block()

        # Cumulative-error nudge: if prior turns hit timeouts, push toward a
        # lighter-weight approach.
        recovery = ""
        if self._total_code_errors > 0:
            recovery = (
                f">> RECOVERY HINT: {self._total_code_errors} prior code execution(s) "
                f"failed (often timeout). Read ONE file with usecols=[...]; consider "
                f"nrows= for an initial peek; avoid loading the largest CSV first.\n\n"
            )

        # Suggest the most recently added (typically smallest, interventional)
        # file as a starting point.
        latest = self._latest_data_query_num()
        latest_hint = (
            f"TIP: query_{latest}_csv is the most recent dataset — usually a small "
            f"interventional file. Prefer reading it first.\n\n"
            if latest is not None else ""
        )

        return (
            f"QUESTION: {self.question.question_text}\n\n"
            f"HYPOTHESIS & CRITERIA (from your earlier planning):\n"
            f"{self._hypothesis_doc}\n\n"
            f"AVAILABLE CSV FILES (already written to disk, readable by your code):\n"
            f"{files_block}\n\n"
            f"{latest_hint}"
            f"{recovery}"
            f"Write code to analyze the data and test the hypothesis. When you have\n"
            f"enough evidence to judge confidence, emit <analysis_ready/>."
        )

    def _latest_data_query_num(self) -> Optional[int]:
        for h in reversed(self._query_history):
            if h.get("success") and h.get("data_file"):
                return h["query_num"]
        return None

    # -------------------------------------------------------------------------
    # CALL 3 — ANALYSIS
    # -------------------------------------------------------------------------

    _ANALYSIS_SYSTEM = """You are a scientist interpreting evidence from your experiments.

Your job:
  1. Summarize what the evidence so far says — for AND against the hypothesis.
  2. Give an explicit numeric confidence (0-100%) with reasoning.
  3. Decide: continue collecting data, submit an answer, or give up.
     - Answer when your confidence is high and the evidence meets your verification
       criteria. Use as few or as many queries as you genuinely need — do NOT pad
       the count with redundant queries, and do NOT commit prematurely on borderline
       evidence.
  4. Update the experiment memory so future turns know what was tested and found.

Output format (all tags required, in this order):

<evidence_summary>
  For: ...
  Against: ...
  Data quality / limitations: ...
</evidence_summary>
<confidence>NN</confidence>
<confidence_reasoning>Why this level — what would move it up or down.
Explicitly note whether the data you analyzed is the right kind to answer
the question, and whether you have enough of it to draw the conclusion.</confidence_reasoning>
<decision>continue | answer | give_up</decision>
<answer>Your final answer (only if decision=answer). Variable names should match the VARIABLES catalog exactly. Answer the question \
fully and in the form the question implies.</answer>
<memory_update>
## Experiments & Findings
[list each query with one-line finding, using query_N_csv file names]

## Evidence so far
For: ...
Against: ...
Data quality notes: ...

## Open questions / next experiment idea
...
</memory_update>"""

    def _run_analysis(
        self,
        code_rounds: List[Dict[str, str]],
        budget_remaining: int,
        force_answer: bool = False,
    ) -> Dict[str, Any]:
        user = self._analysis_user_prompt(code_rounds, budget_remaining, force_answer)
        messages = [
            {"role": "system", "content": self._ANALYSIS_SYSTEM},
            {"role": "user", "content": user},
        ]
        response = self._call_llm(messages)
        logger.debug(f"ANALYSIS raw response:\n{response}")

        evidence = _first_group(_EVIDENCE_RE, response)
        confidence_m = _CONFIDENCE_RE.search(response)
        decision_m = _DECISION_RE.search(response)
        answer = _first_group(_ANSWER_RE, response)
        memory = _first_group(_MEMORY_RE, response)
        conf_reason = _first_group(_CONF_REASON_RE, response)
        revision = _first_group(_HYPOTHESIS_REVISION_RE, response)

        if memory:
            self._experiment_memory = memory
        if revision:
            self._hypothesis_doc = (
                f"{self._hypothesis_doc}\n\n"
                f"=== REVISED HYPOTHESIS (turn {self._turn_number}) ===\n{revision}"
            )
            logger.info(
                f"  Hypothesis revised at turn {self._turn_number}:\n"
                f"{_indent(revision, 4)}"
            )
        if confidence_m:
            try:
                self._last_confidence = max(0, min(100, int(confidence_m.group(1))))
            except ValueError:
                pass

        decision = decision_m.group(1).lower() if decision_m else ""
        if force_answer:
            decision = "answer"
            if not answer:
                answer = "Unable to determine"

        # Fallbacks
        if decision == "answer" and not answer:
            # Try to salvage from evidence_summary
            answer = (evidence.split("\n", 1)[0] if evidence else "").strip() or "Unable to determine"
            logger.warning("ANALYSIS: decision=answer but no <answer> tag — salvaged from evidence")
        if not decision:
            # No decision produced — default to continue unless budget spent
            decision = "answer" if budget_remaining <= 0 else "continue"
            logger.warning(f"ANALYSIS: no <decision> tag — defaulting to '{decision}'")

        logger.info(
            f"  ANALYSIS: confidence={self._last_confidence}%, decision={decision}"
            + (f", answer={answer!r}" if decision == "answer" else "")
        )
        if evidence:
            logger.info(f"  Evidence:\n{_indent(evidence, 4)}")
        if conf_reason:
            logger.info(f"  Confidence reasoning:\n{_indent(conf_reason, 4)}")

        return {
            "raw": response,
            "evidence": evidence or (conf_reason or ""),
            "confidence": self._last_confidence,
            "decision": decision,
            "answer": answer,
        }

    def _analysis_user_prompt(
        self,
        code_rounds: List[Dict[str, str]],
        budget_remaining: int,
        force_answer: bool,
    ) -> str:
        if code_rounds:
            rounds_str = "\n\n".join(
                f"--- Round {i + 1} ---\nCODE:\n{r['code']}\n\nOUTPUT:\n{r['output']}"
                for i, r in enumerate(code_rounds)
            )
        else:
            rounds_str = "(No code was executed this turn — reason from memory only.)"

        prior_memory = self._experiment_memory or "(empty — this is your first analysis)"

        budget_note = (
            ">> BUDGET EXHAUSTED — you MUST set <decision>answer</decision>.\n"
            if force_answer or budget_remaining <= 0
            else (
                f"Budget remaining: {budget_remaining} queries "
                f"(used {self._queries_made}/{self.max_queries}).\n"
                f">> Aim for confidence ≥ {HIGH_CONFIDENCE_THRESHOLD}% before "
                f"answering. If confidence is below that and budget remains, "
                f"prefer one more confirmatory query over committing.\n"
            )
        )

        sys_alerts = ""
        if self._system_messages:
            sys_alerts = ">> SYSTEM ALERTS:\n" + "\n".join(
                f"  - {m}" for m in self._system_messages
            ) + "\n"

        return (
            f"QUESTION: {self.question.question_text}\n\n"
            f"YOUR HYPOTHESIS & CRITERIA:\n{self._hypothesis_doc}\n\n"
            f"PRIOR EXPERIMENT MEMORY:\n{prior_memory}\n\n"
            f"QUERIES RUN SO FAR:\n{self._queries_summary()}\n\n"
            f"CODE EXECUTED THIS TURN:\n{rounds_str}\n\n"
            f"{budget_note}{sys_alerts}\n"
            f"Produce all required tags: <evidence_summary>, <confidence>, "
            f"<confidence_reasoning>, <decision>, <answer> (only if answering), "
            f"<memory_update>. If new evidence contradicts your original "
            f"hypothesis, ALSO include <hypothesis_revision>...</hypothesis_revision> "
            f"with the updated direction so the next DESIGN step pivots."
        )

    # -------------------------------------------------------------------------
    # CALL 4 — DESIGN
    # -------------------------------------------------------------------------

    _DESIGN_SYSTEM = """You are an experiment designer. Choose the data type (observational vs. interventional) that actually answers \
the strategy you laid out, then design ONE next experiment that most efficiently moves confidence toward a conclusion.

Guidelines:
  - ONE intervention per query. NEVER bundle do(X=A) and do(X=B) in a single
    query — the parser will reject it. Issue them as SEPARATE queries.
  - Always include ALL variables you'll need to compare in the sample list, so
    one query gathers everything you need at once.
  - Pick a sample size that matches the analysis you plan to run — large enough
    to support the comparisons you need, not so large you waste budget.
  - HARD LIMIT: request at most 10000 samples. Queries for 20000, 30000, etc.
    are invalid and will be rejected.
  - Also obey any smaller remaining sample budget or final-turn warning in
    SYSTEM ALERTS.
  - Do NOT re-run an experiment you already have data for.
  - Respect INTERVENTION LIMITS: some variables are non-manipulable.
  - If a previous query FAILED (parser error), simplify: drop the multi-do
    bundling and ask for the simplest single-intervention form.

Output format (both tags required):

<rationale>One short paragraph: what you're testing and why it moves confidence.</rationale>
<query>Natural-language request the world model will parse. Examples (use these
shapes — single intervention per query):
  "Give me 500 observational samples of A, B, C, D"
  "Give me 500 samples of A, B, Y where we intervene to set A=yes"
</query>"""

    def _run_design(self) -> Dict[str, Any]:
        user = self._design_user_prompt()
        messages = [
            {"role": "system", "content": self._DESIGN_SYSTEM},
            {"role": "user", "content": user},
        ]
        response = self._call_llm(messages)
        logger.debug(f"DESIGN raw response:\n{response}")

        rationale = _first_group(_RATIONALE_RE, response)
        query = _first_group(_QUERY_TAG_RE, response)
        if rationale:
            logger.info(f"  Rationale: {rationale}")
        if query:
            logger.info(f"  Query: {query}")
        else:
            logger.warning("  DESIGN produced no <query> tag")
        return {"raw": response, "rationale": rationale, "query": query}

    def _intervention_limits_block(self) -> str:
        non_interv = self.world_info.non_intervenable_variables
        if non_interv:
            lines = ["INTERVENTION LIMITS — cannot intervene on:"]
            for var, reason in non_interv.items():
                lines.append(f"  - {var}: {reason}")
            return "\n".join(lines)
        return "INTERVENTION LIMITS: all variables are intervenable"

    def _design_user_prompt(self) -> str:
        remaining = self.max_queries - self._queries_made
        nonint_str = self._intervention_limits_block()

        memory = self._experiment_memory or "(empty — no experiments yet)"

        # Recovery hint: if recent queries failed (parser errors), nudge toward
        # the simplest single-intervention form. Look at the last 2 queries.
        recent_failures = [
            h for h in self._query_history[-2:] if not h.get("success")
        ]
        recovery = ""
        if recent_failures:
            recovery = (
                f">> RECOVERY HINT: {len(recent_failures)} of your last queries "
                f"FAILED to parse. Simplify the next query: ONE intervention only "
                f"(no \"then\" / multi-do bundling), short variable list, plain phrasing, "
                f"and N <= {MAX_SAMPLES_PER_QUERY}.\n"
                f"   Use exactly the form: \"Give me N samples of A, B, C where we "
                f"intervene to set X=v\"\n\n"
            )
            failure_details = "\n".join(
                f"   - Failed query {h['query_num']}: {h.get('query')} "
                f"ERROR: {h.get('error_message') or 'unknown error'}"
                for h in recent_failures
            )
            recovery += f">> FAILED QUERY DETAILS:\n{failure_details}\n\n"

        return (
            f"QUESTION: {self.question.question_text}\n\n"
            f"{self._system_alerts_block()}"
            f"YOUR HYPOTHESIS & CRITERIA:\n{self._hypothesis_doc}\n\n"
            f"EXPERIMENT MEMORY:\n{memory}\n\n"
            f"LAST CONFIDENCE: "
            f"{self._last_confidence if self._last_confidence is not None else '(none yet)'}%\n\n"
            f"QUERIES RUN SO FAR:\n{self._queries_summary()}\n\n"
            f"VARIABLES:\n{self.world_info.get_variable_catalog()}\n\n"
            f"{nonint_str}\n\n"
            f"Budget remaining: {remaining} / {self.max_queries}\n\n"
            f"MAX SAMPLE SIZE PER QUERY: {MAX_SAMPLES_PER_QUERY}. Do not exceed it, "
            f"and obey any smaller remaining sample budget in SYSTEM ALERTS.\n\n"
            f"{recovery}"
            f"Produce <rationale> and <query>."
        )

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _build_exec_namespace(self) -> Dict[str, Any]:
        qd = _SafeQueryDict()
        namespace: Dict[str, Any] = {
            "pd": pd,
            "np": np,
            "stats": scipy_stats,
            "chi2_contingency": chi2_contingency,
            "query_files": qd,
        }
        for h in self._query_history:
            if h.get("data_file"):
                qn = h["query_num"]
                path = h["data_file"]
                namespace[f"query_{qn}_csv"] = path
                dict.__setitem__(qd, qn, path)
        return namespace

    def _files_block(self) -> str:
        if not self._query_history:
            return "(none)"
        lines = []
        for h in self._query_history:
            if not (h["success"] and h.get("data_file")):
                err = h.get("error_message") or "unknown error"
                lines.append(f"  # query_{h['query_num']}_csv FAILED: {err}")
                continue
            interv = _format_interventions(h.get("interventions")) or "observational"
            basename = os.path.basename(h["data_file"])
            lines.append(
                f"  query_{h['query_num']}_csv  # {h['query_type']} {interv}, "
                f"N={h['n_rows']}, file={basename}"
            )
        return "\n".join(lines)

    def _queries_summary(self) -> str:
        if not self._query_history:
            return "(none yet)"
        lines = []
        for h in self._query_history:
            status = "OK" if h["success"] else "FAIL"
            interv = _format_interventions(h.get("interventions")) or "observational"
            n = h["n_rows"] if h["success"] else "?"
            if h["success"]:
                lines.append(
                    f"  Query {h['query_num']}: [{status}] {h['query_type']} {interv}, "
                    f"N={n} -> query_{h['query_num']}_csv"
                )
            else:
                err = h.get("error_message") or "unknown error"
                lines.append(
                    f"  Query {h['query_num']}: [{status}] {h['query_type']} {interv}, "
                    f"N={n}, requested={h.get('query')!r}, error={err}"
                )
        return "\n".join(lines)

    def _data_preview(self, data_file: str) -> str:
        try:
            df = pd.read_csv(data_file)
        except Exception as e:
            return f"(could not read file: {e})"
        lines = [
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns: {', '.join(df.columns.tolist())}",
            "Marginals (first 5 columns):",
        ]
        for col in list(df.columns)[:5]:
            vc = df[col].value_counts(normalize=True).sort_index()
            lines.append("  " + col + ": " + ", ".join(
                f"{k}={v:.1%}" for k, v in vc.items()
            ))
        if len(df.columns) > 5:
            lines.append(f"  ... ({len(df.columns) - 5} more — load CSV to see all)")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Module-level parsing helpers
# -----------------------------------------------------------------------------

def _first_group(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


_CODE_BARE_START_RE = re.compile(
    r"^\s*(?:import\s|from\s|print\s*\(|df\s*=|pd\.read_csv|#\s)",
)


def _extract_code(response: str) -> str:
    """Extract Python code from a CODE-phase response.

    Preference order:
      1. <code>...</code> tag (canonical).
      2. ```python ... ``` fenced block.
      3. Bare fallback ONLY when the response LOOKS like pure code — i.e. starts
         with a Python statement. This prevents natural-language analysis (which
         may incidentally contain `pd.read_csv` or `print(` in prose) from being
         exec'd. Returns "" otherwise, letting the loop end cleanly.
    """
    m = _CODE_TAG_RE.search(response)
    if m:
        return m.group(1).strip()
    m = _CODE_FENCE_RE.search(response)
    if m:
        return m.group(1).strip()
    stripped = response.strip()
    if not stripped or _ANALYSIS_READY_RE.search(stripped):
        return ""
    # Bare fallback: require the very first non-blank line to be Python-like.
    if _CODE_BARE_START_RE.match(stripped):
        logger.warning("_extract_code: using bare-response fallback (no <code> tag or fence)")
        return stripped
    return ""
