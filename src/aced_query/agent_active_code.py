"""
agent_active_code.py

The Active+Code agent: the Active agent plus Python analysis of its datasets, with one <action type="X"> format.

Each turn runs an inner loop where the model picks one action per response:
  - type="code"     → execute Python, return stdout, continue loop
  - type="query"    → return to orchestrator to fetch a new dataset (costs budget)
  - type="answer"   → return final answer, end loop
  - type="give_up"  → return give-up, end loop

Outer interface:
    scientist.initialize(world_info, question, max_queries)
    action = scientist.get_next_action()   # {"type": ..., "content": ..., ...}
    scientist.receive_result(result)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import scipy.stats as scipy_stats

from schemas import Question, QueryResult, WorldInfo

logger = logging.getLogger(__name__)

MAX_CODE_ROUNDS = 8          # max code actions per outer turn
MAX_CODE_OUTPUT_CHARS = 3000  # truncate long outputs
CODE_TIMEOUT_SECONDS = 45    # kill runaway code (includes process start-up)
# Cross-round state cap: drop DataFrames / large arrays and any var >1 MiB pickled.
# Small scalars / dicts / summary results still flow across rounds.
_MAX_VAR_PICKLE_BYTES = 1_000_000
_MAX_NDARRAY_ELEMS = 10_000

# Match any <action type="X">...</action> tag (strict: requires closing tag)
_ACTION_RE = re.compile(
    r'<action\s+type="(code|query|answer|give_up)">\s*(.*?)\s*</action>',
    re.DOTALL | re.IGNORECASE,
)
# Lenient fallback for code actions where the LLM omits </action>.
# Captures everything after <action type="code"> up to </action>, a subsequent
# XML-like tag, or end-of-string — whichever comes first.
_CODE_OPEN_RE = re.compile(
    r'<action\s+type="code">\s*(.*?)(?=\s*</action>|\s*<(?:scientist_memory|reasoning|action)\b|$)',
    re.DOTALL | re.IGNORECASE,
)
# Same lenient fallback for terminal action types (query/answer/give_up).
_TERMINAL_OPEN_RE = re.compile(
    r'<action\s+type="(query|answer|give_up)">\s*(.*?)(?=\s*</action>|\s*<(?:scientist_memory|reasoning|action)\b|$)',
    re.DOTALL | re.IGNORECASE,
)
# Heuristic: detect a bare type="query" attribute anywhere (e.g. malformed tag).
_QUERY_ATTR_RE = re.compile(r'type\s*=\s*["\']?query["\']?', re.IGNORECASE)
# Strip markdown code fences if the LLM wraps code in them
_FENCE_RE = re.compile(r"^```[a-z]*\n?", re.MULTILINE)
# Names injected by _build_exec_namespace — skip when propagating state back to parent
_INJECTED_NAMES = frozenset({"pd", "np", "stats", "chi2_contingency", "query_files"})


def _has_python_error(output: str) -> bool:
    """Detect a failed / timed-out code execution from the captured stdout."""
    return "[PYTHON ERROR]" in output or output.startswith("[TIMEOUT")


def _extract_tag(response: str, tag: str) -> str:
    """Extract the first <tag>…</tag> block, empty string if missing."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


# Forced terminal prompt injected when the code-round limit is reached.
# Appended to the last env-output message so the model has full context first,
# then a hard, unambiguous demand for a terminal action.
_FORCE_TERMINAL_PROMPT = """\
━━━ CODE LIMIT REACHED — TERMINAL ACTION REQUIRED ━━━
No more code rounds. Put the <action> FIRST, then reasoning:

<action type="answer">YES / NO / your conclusion</action>
<reasoning>Your final conclusion based on the analysis done so far.</reasoning>
<scientist_memory>Key findings.</scientist_memory>

Or if budget remains: <action type="query">data request</action>
Or: <action type="give_up">reason</action>

Do NOT output <action type="code"> — it will NOT be executed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


_FORMAT_REPAIR_PROMPT = """\
Your previous response could not be parsed because it was empty or did not
contain a valid <action> block.

Return exactly ONE valid action block now. Put the action first. Do not include
any prose before it.

Valid formats:
<action type="query">Give me N observational samples of variables A, B</action>
<action type="query">Give me N samples of variables A, B where we intervene to set X=value</action>
<action type="code">python code here</action>
<action type="answer">final answer here</action>
<action type="give_up">brief reason</action>

Then optionally include:
<reasoning>brief reason</reasoning>
<scientist_memory>Tested: ... Known: ... Uncertain: ... Next: ...</scientist_memory>
"""


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
        else:
            raise TypeError(
                f"query_files is read-only and query {key} has not been received yet. "
                f"Available query numbers: {sorted(self.keys())}. "
                f"Request more data first with an <action type=\"query\">."
            )

    def __missing__(self, key):
        available = sorted(self.keys())
        raise KeyError(
            f"{key!r} not in query_files. "
            f"Available query numbers: {available}. "
            f"Only access files listed in AVAILABLE DATA FILES."
        )


# -----------------------------------------------------------------------------
# Code execution helper
# -----------------------------------------------------------------------------

def _code_worker(code, query_files_map, extra_vars, result_q):
    """Top-level (spawn-pickleable) worker. Runs the agent's code in a fresh Python process."""
    import os
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
                f"Request more data first with an <action type=\"query\">."
            )

        def __missing__(self, key):
            raise KeyError(
                f"{key!r} not in query_files. "
                f"Available query numbers: {sorted(self.keys())}. "
                f"Only access files listed in AVAILABLE DATA FILES."
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
    dropped = []
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
    """
    Execute Python code in a spawned child process with a hard-killable timeout.
    New variables created by the code are propagated back to the namespace.

    Uses "spawn" (not "fork") so the child does not inherit the parent's CUDA context,
    tokenizers thread pool, or BLAS mutex state — all of which previously caused
    `pd.read_csv` to deadlock under the fork-based implementation.
    """
    import pickle

    code = _FENCE_RE.sub("", code).replace("```", "").strip()

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
    out, new_vars = result

    namespace.update(new_vars)

    if not out.strip():
        out = "[No output — did you forget to print()?]"

    if len(out) > max_chars:
        out = out[:max_chars] + f"\n... [truncated at {max_chars} chars]"

    return out


# -----------------------------------------------------------------------------
# CodeAct Scientist Agent
# -----------------------------------------------------------------------------

@dataclass
class ActiveCodeAgent:
    """
    LLM-powered causal discovery agent using CodeAct-style Python execution.

    Inner loop per outer turn
    ─────────────────────────
    messages = [system, user]
    loop:
        response = llm(messages)
        if response contains  <action type="code">...</action>:
            execute code → append (assistant, response) + (user, [env stdout]...)
        else:
            final state → parse response for query / answer / give_up
            break

    The LLM must implement generate_messages(messages) -> str.
    BedrockLLM, OpenAILLM and GeminiLLM satisfy this.
    """
    llm: Any

    # Set by initialize()
    world_info: Optional[WorldInfo] = field(default=None, init=False)
    question: Optional[Question] = field(default=None, init=False)
    max_queries: int = field(default=10, init=False)

    _query_history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _queries_made: int = field(default=0, init=False)
    _system_messages: List[str] = field(default_factory=list, init=False)
    _scientist_memory: str = field(default="", init=False)

    # Analysis counters — populated into phase_summary each turn and rolled up
    # into experiment_summary by the orchestrator.
    _turn_number: int = field(default=0, init=False)
    _total_code_rounds: int = field(default=0, init=False)
    _total_code_errors: int = field(default=0, init=False)
    _total_llm_calls: int = field(default=0, init=False)

    def initialize(self, world_info: WorldInfo, question: Question,
                   max_queries: int) -> None:
        self.world_info = world_info
        self.question = question
        self.max_queries = max_queries
        self._query_history = []
        self._queries_made = 0
        self._system_messages = []
        self._scientist_memory = ""
        self._turn_number = 0
        self._total_code_rounds = 0
        self._total_code_errors = 0
        self._total_llm_calls = 0
        logger.info(f"ActiveCodeAgent initialized. Question: {question.question_text}")

    # -------------------------------------------------------------------------
    # Main CodeAct loop
    # -------------------------------------------------------------------------

    def get_next_action(self) -> Dict[str, Any]:
        """
        Run the CodeAct inner loop for one outer turn.

        Returns dict with:
            type, content, raw_response, reasoning, scientist_memory,
            code_rounds, llm_transcript, phase_summary
        """
        if self.world_info is None or self.question is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        self._turn_number += 1
        user_prompt = self._get_user_prompt()
        has_data = any(h.get("data_file") for h in self._query_history)
        logger.info(
            f"── TURN {self._turn_number} START ── queries_made={self._queries_made}/"
            f"{self.max_queries}, has_data={has_data}, "
            f"cum_code_rounds={self._total_code_rounds} (errors={self._total_code_errors})"
        )
        logger.debug(f"USER PROMPT:\n{user_prompt}")

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user",   "content": user_prompt},
        ]

        # Namespace persists across all code rounds this turn
        namespace = self._build_exec_namespace()

        final_response = ""
        code_rounds: List[Dict[str, Any]] = []  # [{round_num, code, output, errored, ...}]
        phases_run: List[str] = []
        llm_calls_this_turn = 0
        format_repair_used = False

        def _call_llm(label: str) -> str:
            nonlocal llm_calls_this_turn
            out = self._strip_think_block(self.llm.generate_messages(messages))
            llm_calls_this_turn += 1
            self._total_llm_calls += 1
            phases_run.append(label)
            return out

        for round_idx in range(MAX_CODE_ROUNDS):
            response_clean = _call_llm(f"round_{round_idx + 1}")
            logger.info(f"LLM response (round {round_idx + 1}):\n{response_clean}")

            messages.append({"role": "assistant", "content": response_clean})

            m = _ACTION_RE.search(response_clean)
            if not m:
                # Strict match failed. Try lenient extraction for unclosed code actions:
                # LLMs (especially Qwen) frequently emit <action type="code"> without
                # the closing </action>, which causes the regex to return None and the
                # code to never execute. Extract the code anyway.
                m_code = _CODE_OPEN_RE.search(response_clean)
                if m_code:
                    action_type = "code"
                    action_content = m_code.group(1).strip()
                    logger.warning(
                        f"Round {round_idx + 1}: <action type='code'> missing </action> — "
                        "using lenient extraction fallback"
                    )
                else:
                    # No recognisable action at all. Some OpenAI-compatible
                    # reasoning models occasionally return empty content or
                    # plain prose ("We need to run code") despite the required
                    # XML contract. Give them one immediate repair chance inside
                    # the loop so a repaired code action can still execute.
                    if not format_repair_used:
                        logger.warning(
                            "Round %d: no valid action block; requesting one format repair",
                            round_idx + 1,
                        )
                        messages.append({
                            "role": "user",
                            "content": (
                                _FORMAT_REPAIR_PROMPT
                                + "\n\nPrevious unparseable response:\n"
                                + (response_clean if response_clean.strip() else "[empty response]")
                            ),
                        })
                        response_clean = _call_llm(f"round_{round_idx + 1}_format_repair")
                        logger.info(
                            "LLM format-repair response (round %d):\n%s",
                            round_idx + 1,
                            response_clean,
                        )
                        messages.append({"role": "assistant", "content": response_clean})
                        format_repair_used = True

                        m = _ACTION_RE.search(response_clean)
                        if m:
                            action_type = m.group(1).lower()
                            action_content = m.group(2).strip()
                        else:
                            m_code = _CODE_OPEN_RE.search(response_clean)
                            if m_code:
                                action_type = "code"
                                action_content = m_code.group(1).strip()
                                logger.warning(
                                    "Round %d repair: <action type='code'> missing </action> — "
                                    "using lenient extraction fallback",
                                    round_idx + 1,
                                )
                            else:
                                # Still unparseable — treat as final and let
                                # the normal parser produce the give_up.
                                final_response = response_clean
                                break
                    else:
                        # No recognisable action and repair already used — treat as final
                        final_response = response_clean
                        break
            else:
                action_type = m.group(1).lower()
                action_content = m.group(2).strip()

            if action_type == "code":
                # ── Code execution ────────────────────────────────────────
                if not has_data:
                    output = (
                        "[ERROR] No datasets available — code not executed. "
                        "Use <action type=\"query\"> to request data first."
                    )
                else:
                    logger.info(f"Code round {round_idx + 1}: executing:\n{action_content}")
                    output = _execute_code(action_content, namespace)
                    logger.info(f"Output:\n{output}")

                errored = _has_python_error(output)
                self._total_code_rounds += 1
                if errored:
                    self._total_code_errors += 1

                code_rounds.append({
                    "round_num": round_idx + 1,
                    "code": action_content,
                    "output": output,
                    "errored": errored,
                    # Per-round reasoning and memory — useful for post-hoc analysis
                    # of how the agent's thinking evolved across code rounds.
                    "reasoning": _extract_tag(response_clean, "reasoning"),
                    "scientist_memory": _extract_tag(response_clean, "scientist_memory"),
                    "raw_response": response_clean,
                })

                obs = f"[env]\n{output}".strip()
                messages.append({"role": "user", "content": obs})
                final_response = response_clean
                continue

            else:
                # ── query / answer / give_up → return to orchestrator ─────
                final_response = response_clean
                break

        # If the loop exhausted all code rounds, the last message is the env output.
        # Append the strong forcing prompt to that message, then call the LLM once
        # more so it can produce a terminal action with full context available.
        forced_terminal = False
        last_resort_used = False
        if messages[-1]["role"] == "user":
            messages[-1]["content"] += "\n\n" + _FORCE_TERMINAL_PROMPT
            extra = _call_llm("force_terminal")
            messages.append({"role": "assistant", "content": extra})
            final_response = extra
            forced_terminal = True
            logger.info(f"Post-loop forced-terminal LLM call:\n{extra}")

            # If the model still responds with a code action, make one last-resort call.
            m_final = _ACTION_RE.search(final_response)
            is_code_final = (m_final and m_final.group(1).lower() == "code") or (
                not m_final and bool(_CODE_OPEN_RE.search(final_response))
            )
            if is_code_final:
                logger.warning("Post-loop response still a code action — forcing last-resort terminal call")
                messages.append({
                    "role": "user",
                    "content": (
                        "FINAL ATTEMPT — do not write any code.\n"
                        f"Answer the question: \"{self.question.question_text}\"\n"
                        "Respond with ONLY (action FIRST):\n"
                        "<action type=\"answer\">your answer</action>\n"
                        "<reasoning>conclusion</reasoning>"
                    ),
                })
                last_resort = _call_llm("last_resort")
                messages.append({"role": "assistant", "content": last_resort})
                final_response = last_resort
                last_resort_used = True
                logger.info(f"Last-resort terminal call:\n{last_resort}")

        # Parse structured output from final response
        reasoning = self._extract_reasoning(final_response)
        if reasoning:
            logger.info(f"Reasoning: {reasoning[:300]}...")

        self._update_memory_from_response(final_response)
        action = self._parse_action(final_response)

        code_errors_this_turn = sum(1 for r in code_rounds if r.get("errored"))
        phase_summary = {
            "turn_number": self._turn_number,
            "phases_run": phases_run,
            "n_llm_calls": llm_calls_this_turn,
            "code_rounds_this_turn": len(code_rounds),
            "code_errors_this_turn": code_errors_this_turn,
            "cumulative_code_rounds": self._total_code_rounds,
            "cumulative_code_errors": self._total_code_errors,
            "cumulative_llm_calls": self._total_llm_calls,
            "hit_code_round_limit": len(code_rounds) == MAX_CODE_ROUNDS,
            "forced_terminal": forced_terminal,
            "last_resort_used": last_resort_used,
            "format_repair_used": format_repair_used,
            # This agent doesn't emit explicit numeric confidence — left None
            # so experiment_summary.confidence_trajectory stays empty.
            "confidence_after_turn": None,
            "queries_used_so_far": self._queries_made,
            "queries_remaining": self.max_queries - self._queries_made,
            "action_type": action["type"],
        }

        action["raw_response"] = final_response
        action["reasoning"] = reasoning
        action["scientist_memory"] = self._scientist_memory
        action["code_rounds"] = code_rounds
        action["llm_transcript"] = messages
        action["phase_summary"] = phase_summary

        logger.info(
            f"── TURN {self._turn_number} END ── action={action['type']} "
            f"content={str(action['content'])[:80]}... | "
            f"code_rounds={len(code_rounds)} (errors={code_errors_this_turn}) | "
            f"llm_calls={llm_calls_this_turn}"
            + (" | FORCED" if forced_terminal else "")
            + (" | LAST-RESORT" if last_resort_used else "")
        )
        return action

    # -------------------------------------------------------------------------
    # Result ingestion
    # -------------------------------------------------------------------------

    def receive_result(self, result: QueryResult) -> None:
        """Store query result with a brief data preview."""
        # Only count successful queries toward budget (aligned with orchestrator)
        if result.success:
            self._queries_made += 1

        query_num = len(self._query_history) + 1
        preview = None
        if result.success and result.data_file:
            preview = self._data_preview(result.data_file)

        self._query_history.append({
            "query_num": query_num,
            "query": result.query.raw_query,
            "success": result.success,
            "data_file": result.data_file,
            "n_rows": result.n_rows,
            "interventions": result.query.interventions,
            "query_type": result.query.query_type.value if result.query else "unknown",
            "preview": preview,
        })
        logger.info(f"Received result #{query_num}: success={result.success}, n={result.n_rows}")

    def receive_system_message(self, message: str) -> None:
        self._system_messages.append(message)
        logger.info(f"System message: {message[:100]}...")

    # -------------------------------------------------------------------------
    # Execution namespace
    # -------------------------------------------------------------------------

    def _build_exec_namespace(self) -> Dict[str, Any]:
        """
        Build the Python namespace injected into every exec() call this turn.

        Pre-loaded:
            pd               — pandas
            np               — numpy
            stats            — scipy.stats
            chi2_contingency — scipy.stats.chi2_contingency
            query_files      — dict {query_num (int): csv_path (str)}
            query_N_csv      — convenience alias: the path for query N
        """
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
                dict.__setitem__(qd, qn, path)  # bypass read-only guard
        return namespace

    # -------------------------------------------------------------------------
    # Brief data preview for the prompt (anchors the agent, not analysis)
    # -------------------------------------------------------------------------

    def _data_preview(self, data_file: str) -> str:
        """Return a short preview of column names + value distributions."""
        try:
            df = pd.read_csv(data_file)
        except Exception as e:
            return f"(could not read file: {e})"

        lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
                 f"Columns: {', '.join(df.columns.tolist())}",
                 "Marginals (first 5 columns):"]
        for col in list(df.columns)[:5]:
            vc = df[col].value_counts(normalize=True).sort_index()
            lines.append("  " + col + ": " + ", ".join(
                f"{k}={v:.1%}" for k, v in vc.items()
            ))
        if len(df.columns) > 5:
            lines.append(f"  ... ({len(df.columns) - 5} more — load CSV to see all)")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Prompts
    # -------------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        return """You are a scientist investigating an unknown system. Read the question carefully and decide for yourself what it is \
asking and what approach will answer it.

You can collect data AND analyze it using Python.

━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each response must contain EXACTLY ONE action:

1. <action type="code">
   Run Python to analyze data you already have (does NOT cost queries)

2. <action type="query">
   Request new data (costs one query)
   - Observational: "Give me N observational samples of variables A, B, C"
   - Interventional: "Give me N samples of variables A, B, C where we intervene to set X=value"

3. <action type="answer">
   Submit your final answer in the form the question implies. Variable names must
   match the VARIABLES catalog exactly.

4. <action type="give_up">
   If the question cannot be answered

━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO THINK
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Think like a scientist running experiments. Different questions call for different
approaches — choose the data and analysis that actually answer what is being asked,
rather than defaulting to one recipe.

You have two kinds of data:
- Observational: drawn from the system's natural data-generating process; reveals
  how variables jointly behave (associations, correlations).
- Interventional do(X=v): X is fixed to v while sampling, severing X's incoming
  edges; reveals the downstream effect of fixing X.

A few principles to keep in mind:
- Correlation in observational data is not the same as a causal effect.
- Pick the data type that matches the question; large samples don't fix using the
  wrong kind of data.
- Prefer the smallest experiment that can answer the question; use code to compare
  distributions or compute whatever the question actually needs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO USE CODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use Python ONLY to analyze data you already have.

LOADING DATA — read this carefully:
  Each successful query pre-loads two things into your Python namespace:
    - query_N_csv  — a STRING variable holding the absolute path to query N's CSV
                     (N is the per-experiment query number shown in AVAILABLE DATA FILES,
                      e.g. query_1_csv, query_2_csv — NOT the long basename like
                      "query_0019_observational_….csv" shown after `file=` in the comment)
    - query_files  — a read-only DICT mapping {N: path}, e.g. query_files[1]

  Correct ways to load query N:
    df = pd.read_csv(query_1_csv)         # bare variable, no quotes
    df = pd.read_csv(query_files[1])      # dict lookup

  WRONG — these will all raise FileNotFoundError:
    df = pd.read_csv("query_1_csv")                                  # quoting the variable name
    df = pd.read_csv("query_0019_observational_20260501_233850.csv") # using the basename literal
    df = pd.read_csv("/absolute/path/from/world_output.csv")         # don't paste paths from <data_file>

Typical workflow:
- Load data: df = pd.read_csv(query_1_csv)
- Look at distributions:
    df["X"].value_counts(normalize=True)
- Compare distributions across conditions:
    df[df["X"]=="a"]["Y"].value_counts(normalize=True)

DataFrames, Series, and large arrays do NOT carry across code rounds — re-read the CSV
each round, or print() any value you need later. Small scalars/dicts persist.

Use whatever analysis genuinely fits the question — simple comparisons, summary
statistics, or formal tests are all fair game. Pick the simplest tool that gives
you a clear answer.

Do NOT:
- Simulate interventions in code
- Overwrite columns to fake experiments

━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEFORE ANSWERING
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ensure:
- You actually collected the kind of data that can answer this question.
- Your conclusion is grounded in the data you analyzed, not assumption.

━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (REQUIRED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━

<Action must come FIRST>

<action type="code|query|answer|give_up">...</action>

<reasoning>[Brief analysis and decision]</reasoning>

<scientist_memory>
Tested:
Known:
Uncertain:
Next:
</scientist_memory>
"""

    def _get_user_prompt(self) -> str:
        # --- Goal ---
        goal = (
            "════════════════════════════════════════════════════════════════════════════════\n"
            f"YOUR QUESTION: {self.question.question_text}\n"
            "════════════════════════════════════════════════════════════════════════════════"
        )

        # --- Budget & constraints ---
        remaining = self.max_queries - self._queries_made
        if remaining <= 0:
            budget = (f">> BUDGET: {self._queries_made}/{self.max_queries} queries used — "
                      "NO QUERIES LEFT — You MUST answer now!")
        elif remaining <= 2:
            budget = (f">> BUDGET: {self._queries_made}/{self.max_queries} queries used — "
                      f"Only {remaining} remaining — Answer soon!")
        else:
            budget = f"BUDGET: {self._queries_made}/{self.max_queries} queries used — {remaining} remaining"

        non_interv = self.world_info.non_intervenable_variables
        if non_interv:
            interv_lines = ["INTERVENTION LIMITS — Cannot intervene on (non-manipulable variables):"]
            for var, reason in non_interv.items():
                interv_lines.append(f"  - {var}: {reason}")
            interv_str = "\n".join(interv_lines)
        else:
            interv_str = "INTERVENTION LIMITS: All variables are intervenable"

        constraints = budget + "\n" + interv_str
        if self._system_messages:
            constraints += "\n>> SYSTEM ALERTS:\n" + "\n".join(f"  - {msg}" for msg in self._system_messages)

        # --- Available data files ---
        if self._query_history:
            file_lines = ["AVAILABLE DATA FILES — use these variable names in your code:"]
            for h in self._query_history:
                if h["success"] and h.get("data_file"):
                    interv_tag = ""
                    conditions = h.get("interventions") or []
                    if conditions:
                        parts = ["do(" + ", ".join(f"{k}={v}" for k, v in c.items()) + ")" for c in conditions]
                        interv_tag = " " + " | ".join(parts)
                    basename = os.path.basename(h["data_file"])
                    file_lines.append(
                        f"  query_{h['query_num']}_csv"
                        f"  # {h['query_type']}{interv_tag}, N={h['n_rows']}, file={basename}"
                    )
                else:
                    file_lines.append(
                        f"  # query_{h['query_num']}_csv  FAILED — no file"
                    )
            data_files = "\n".join(file_lines)
        else:
            data_files = (
                "AVAILABLE DATA FILES: None yet.\n"
                "  → You MUST request data first using <action type=\"query\">.\n"
                "  → Do NOT use <action type=\"code\">."
            )

        # --- Latest query preview ---
        if self._query_history:
            latest = self._query_history[-1]
            qn = latest["query_num"]
            if latest["success"] and latest.get("preview"):
                interv_tag = ""
                conditions = latest.get("interventions") or []
                if conditions:
                    parts = ["do(" + ", ".join(f"{k}={v}" for k, v in c.items()) + ")" for c in conditions]
                    interv_tag = " " + " | ".join(parts)
                latest_section = (
                    f"─── LATEST DATA (Query {qn}) — quick preview ───\n"
                    f"Request: {latest['query']}\n"
                    f"Type: {latest['query_type']}{interv_tag}, N={latest['n_rows']}\n"
                    f"{latest['preview']}\n"
                    f"→ Load with: pd.read_csv(query_{qn}_csv)"
                )
            else:
                latest_section = (
                    f"─── LATEST QUERY (Query {qn}) — FAILED ───\n"
                    f"Request: {latest['query']}"
                )
        else:
            latest_section = (
                "─── NO QUERIES YET ───\n"
            )

        # --- Memory ---
        if self._scientist_memory:
            memory = f"YOUR CURRENT CAUSAL UNDERSTANDING (from previous turn):\n{self._scientist_memory}"
        else:
            memory = (
                "YOUR CURRENT CAUSAL UNDERSTANDING:\n"
                "(Empty — this is your first turn. After analyzing data, update your memory to track "
                "your evolving causal map: which variables are associated, which causal directions "
                "are confirmed, what remains uncertain.)"
            )

        # --- Past queries summary (all except latest) ---
        history = ""
        if len(self._query_history) > 1:
            lines = ["PAST QUERIES SUMMARY:"]
            for h in self._query_history[:-1]:
                interv_tag = ""
                conditions = h.get("interventions") or []
                if conditions:
                    parts = ["do(" + ", ".join(f"{k}={v}" for k, v in c.items()) + ")" for c in conditions]
                    interv_tag = " " + " | ".join(parts)
                status = f"N={h['n_rows']}" if h["success"] else "FAILED"
                lines.append(
                    f"  Query {h['query_num']}: {status}, "
                    f"{h['query_type']}{interv_tag} → query_{h['query_num']}_csv"
                )
            lines.append(f"  Query {len(self._query_history)}: (latest, see above)")
            history = "\n".join(lines)

        # --- Variables ---
        if self._queries_made == 0:
            variables = (
                f"AVAILABLE VARIABLES (use exact names and state values for interventions):\n"
                f"{self.world_info.get_variable_catalog()}\n\n"
                f"CONTEXT: {self.world_info.story}"
            )
        else:
            variables = (
                f"AVAILABLE VARIABLES (use exact names and state values for interventions):\n"
                f"{self.world_info.get_variable_catalog()}"
            )

        # --- Assemble ---
        parts = [goal, "", variables, "", constraints, "", data_files, "", latest_section, "", memory]
        if history:
            parts += ["", history]
        parts += [
            "",
            "=" * 80,
            "Now: Analyze the latest data → Update your memory → Decide your next action.",
            "Output: <action>, <reasoning>, and <scientist_memory> blocks.",
        ]
        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Response parsing
    # -------------------------------------------------------------------------

    def _strip_think_block(self, response: str) -> str:
        return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    def _update_memory_from_response(self, response: str) -> None:
        response = self._strip_think_block(response)
        match = re.search(r'<scientist_memory>(.*?)</scientist_memory>', response, re.DOTALL)
        if match:
            self._scientist_memory = match.group(1).strip()
            logger.info(f"Memory updated: {self._scientist_memory[:200]}...")
        else:
            # Code-action responses don't carry <scientist_memory> — the LLM has
            # full conversation history within a turn, so no cross-turn persistence
            # is needed mid-loop. Only warn for terminal (query/answer/give_up) actions.
            m = _ACTION_RE.search(response)
            if not (m and m.group(1).lower() == "code"):
                logger.warning("No <scientist_memory> block found in terminal response")

    def _extract_reasoning(self, response: str) -> str:
        response = self._strip_think_block(response)
        match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _parse_action(self, response: str) -> Dict[str, Any]:
        response = self._strip_think_block(response)
        # Normalise curly quotes that some LLMs emit instead of straight quotes
        response = response.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

        # ── Level 1: strict match (closing </action> present) ─────────────────
        match = _ACTION_RE.search(response)
        if match:
            action_type = match.group(1).lower()
            content = match.group(2).strip()
            if action_type == "code":
                # Code action as final: try to salvage reasoning as answer.
                reasoning = self._extract_reasoning(response)
                if reasoning and len(reasoning.split()) > 15:
                    logger.warning("_parse_action: code-as-final — using <reasoning> as answer")
                    return {"type": "answer", "content": f"[From reasoning] {reasoning}"}
                logger.warning("_parse_action: code-as-final, no usable reasoning — giving up")
                return {"type": "give_up", "content": "Agent produced code action instead of terminal action"}
            return {"type": action_type, "content": content}

        # ── Level 2: lenient match for terminal types (unclosed </action>) ────
        m2 = _TERMINAL_OPEN_RE.search(response)
        if m2:
            action_type = m2.group(1).lower()
            content = m2.group(2).strip()
            if content:
                logger.warning(f"_parse_action: unclosed <action type='{action_type}'> — lenient extraction")
                return {"type": action_type, "content": content}

        # ── Level 3: bare type="query" attribute anywhere in the response ─────
        # The world model LLM can parse messy natural language queries,
        # so extract whatever follows the tag opening and pass it along.
        if _QUERY_ATTR_RE.search(response):
            # Strip everything up to and including the malformed opening tag,
            # then cut at any subsequent XML-like tag.
            after = re.sub(r'.*?type\s*=\s*["\']?query["\']?\s*>?', '', response,
                           count=1, flags=re.DOTALL | re.IGNORECASE).strip()
            content = re.split(r'\s*<(?:scientist_memory|reasoning|action|\/?action)\b', after)[0].strip()
            if content:
                logger.warning("_parse_action: heuristic query extraction from bare type=query attribute")
                return {"type": "query", "content": content}

        # ── Level 4: code fallback — reasoning as answer ──────────────────────
        if _CODE_OPEN_RE.search(response):
            reasoning = self._extract_reasoning(response)
            if reasoning and len(reasoning.split()) > 15:
                logger.warning("_parse_action: unclosed code action, using <reasoning> as answer")
                return {"type": "answer", "content": f"[From reasoning] {reasoning}"}
            return {"type": "give_up", "content": "Agent produced code action instead of terminal action"}

        # ── Level 5: reasoning-only response — use it as answer ──────────────
        reasoning = self._extract_reasoning(response)
        if reasoning and len(reasoning.split()) > 15:
            logger.warning("_parse_action: no action tag — using <reasoning> as answer fallback")
            return {"type": "answer", "content": f"[From reasoning] {reasoning}"}

        # ── Level 6: truly unrecoverable ──────────────────────────────────────
        logger.warning(f"_parse_action: could not extract any action: {response[:200]}...")
        return {"type": "give_up", "content": f"No valid <action> tag found: {response[:100]}"}
