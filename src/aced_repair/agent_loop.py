#!/usr/bin/env python3
"""One episode: prompts, action parsing, the turn loop, and answer resolution."""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prompt import AGENT_PROMPT, MEMORY_FULL, MEMORY_STATELESS
from simulator import Simulator

logger = logging.getLogger("agent_loop")


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


_THINK_RE = re.compile(r'<think\b[^>]*>.*?</think\s*>', re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r'<think\b[^>]*>.*$', re.DOTALL | re.IGNORECASE)
# Claude models sometimes write their tool-call namespace into tags (<action>, <invoke>, <memory>),
# occasionally twice; read them as the plain tags. No other model's reply contains "antml:".
_NS_RE = re.compile(r"<(/?)\s*(?:antml:)+", re.IGNORECASE)
_TYPES = ("measure", "intervene", "code", "answer", "give_up")
_START_RE = re.compile(r"<(action|invoke|parameter|code)(?::(\w+))?\b([^<>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""(\w+)\s*=\s*["']?([\w\-]+)["']?""")
_CHAT_TOKEN = r"<\|[a-z_]+\|>"
_END_RE = {
    "action": re.compile(r"</action|<action\b|<invoke\b|<memory\b|</memory|" + _CHAT_TOKEN, re.IGNORECASE),
    "invoke": re.compile(r"</invoke|<invoke\b|<action\b|<memory\b|" + _CHAT_TOKEN, re.IGNORECASE),
    "parameter": re.compile(r"</parameter|</invoke|</action|<parameter\b|<invoke\b|<action\b|<memory\b|</memory|"
                            + _CHAT_TOKEN, re.IGNORECASE),
    "code": re.compile(r"</code|<code\b|<action\b|<invoke\b|<memory\b|</memory|" + _CHAT_TOKEN, re.IGNORECASE),
}
_PARAM_RE = re.compile(
    r'<parameter\s+name\s*=\s*["\']?([\w\-]+)["\']?[^<>]*>(.*?)'
    r'(?:</parameter>|(?=</invoke>|</action>|<parameter\b|<memory>)|$)',
    re.DOTALL | re.IGNORECASE)
_STRAY_CLOSE_RE = re.compile(r"\s*</(?:parameter|invoke|code|answer|action)\s*>?\s*$", re.IGNORECASE)
# our own forced-answer directive, when a model quotes it back
_DIRECTIVE_QUOTE_RE = re.compile(r'<action type="answer">\s*NOW', re.IGNORECASE)


def _strip_think(raw: str) -> str:
    raw = _NS_RE.sub(r"<\1", raw or "")
    t = _THINK_OPEN_RE.sub(" ", _THINK_RE.sub(" ", raw))
    if re.search(r"</think\s*>", t, re.IGNORECASE):
        t = re.split(r"</think\s*>", t, flags=re.IGNORECASE)[-1]
    return t


def _is_json(v: str) -> bool:
    try:
        json.loads(v)
        return True
    except Exception:
        return False


def _loads_lenient(s: str):
    """JSON decode that tolerates what models actually write: code fences, stray closing tags (</parameter>,
    </answer>), leaked chat tokens, trailing text or an extra closing brace, trailing commas.
    Returns (object, repaired: bool); raises ValueError if nothing readable is found."""
    s = _strip_fences(s or "")
    try:
        return json.loads(s), False
    except Exception:
        pass
    t = re.sub(r"</?(?:parameter|invoke|answer|action|code)\b[^>]*>", "", s, flags=re.IGNORECASE)
    t = _strip_fences(re.sub(_CHAT_TOKEN, "", t))
    dec = json.JSONDecoder()
    # an object first (a broken answer must not degrade to an inner list), then a top-level list
    for st in [i for i in (t.find("{"), t.find("[")) if i >= 0]:
        for cand in (t[st:], re.sub(r",\s*([}\]])", r"\1", t[st:])):
            try:
                obj, _ = dec.raw_decode(cand)
            except Exception:
                continue
            if isinstance(obj, dict) or (isinstance(obj, list) and st == t.find("[")
                                         and (t.find("{") < 0 or t.find("[") < t.find("{"))):
                return obj, True
    raise ValueError("no readable JSON object")


_LIST_FIELDS = {"measure": ("requests",), "intervene": ("actions", "measure")}


def _invoke_payload(atype: str, body: str) -> str:
    params = [(n, v.strip()) for n, v in _PARAM_RE.findall(body)]
    if not params:
        return _STRAY_CLOSE_RE.sub("", body.strip())
    if atype == "code" or (len(params) == 1 and params[0][1][:1] in "{["):
        payload = params[0][1]
        if atype != "code" and payload[:1] == "[" and _is_json(payload):   # a bare list -> the action's list field
            payload = json.dumps({"requests" if atype == "measure" else "actions": json.loads(payload)})
        return payload
    obj = {}
    for n, v in params:
        if not v or (atype in _LIST_FIELDS and n not in _LIST_FIELDS[atype]):
            continue                       # e.g. a placeholder <parameter name="dummy"></parameter>
        obj[n] = json.loads(v) if _is_json(v) else v
    return json.dumps(obj) if obj else ""


def _candidates(raw: str) -> List[tuple]:
    """Every action block in a reply, in either dialect: <action type="X">, tool-call <invoke name="X"> with
    <parameter> children, a bare <parameter name="X"> / <parameter name="action" type="X">, or a bare <code>
    block. Returns [(type, payload, dialect)] in order, skipping empty blocks."""
    t = _DIRECTIVE_QUOTE_RE.sub(" NOW", _strip_think(raw))
    out, pos = [], 0
    for m in _START_RE.finditer(t):
        if m.start() < pos:
            continue                       # nested inside the previous block (e.g. <parameter> inside <invoke>)
        tag = m.group(1).lower()
        attrs = {k.lower(): v.lower() for k, v in _ATTR_RE.findall(m.group(3))}
        if m.group(2) and m.group(2).lower() in _TYPES:      # Haiku's mangled form <invoke:code>
            atype = m.group(2).lower()
        elif tag == "action":
            atype = attrs.get("type")
        elif tag == "invoke":
            atype = attrs.get("name") if attrs.get("name") in _TYPES else attrs.get("type")
        elif tag == "parameter":
            atype = attrs.get("name") if attrs.get("name") in _TYPES else attrs.get("type")
            if atype is None:                                  # Haiku's <parameter="code">
                mm = re.match(r"""\s*=\s*["']?(\w+)""", m.group(3))
                atype = mm.group(1).lower() if mm else None
        else:
            atype = "code"
        if atype not in _TYPES:
            continue
        e = _END_RE[tag].search(t, m.end())
        body = t[m.end(): e.start() if e else len(t)]
        # a closing tag is consumed; an opening tag (the next block) is left for the next match
        pos = (e.end() if e.group(0).startswith("</") else e.start()) if e else len(t)
        payload = _invoke_payload(atype, body) if tag == "invoke" else _STRAY_CLOSE_RE.sub("", body.strip())
        payload = _strip_fences(payload)
        if not payload.strip() and atype != "give_up":
            continue                       # an empty block (e.g. Sonnet's empty <action type="code"></action>)
        out.append((atype, payload, "action" if tag == "action" else "invoke"))
    return out


def _valid(atype: str, payload: str) -> bool:
    if atype == "give_up":
        return True
    if atype == "code":
        try:
            compile(payload, "<agent>", "exec")
            return True
        except Exception:
            return False
    try:
        obj, _ = _loads_lenient(payload)
    except ValueError:
        return False
    if atype in _LIST_FIELDS and isinstance(obj, dict):
        # a measure needs a "requests" list and an intervene an "actions" list; any other listed field
        # present must be a list too (a garbled '<parameter name="measure":[...]' gave measure="")
        need, opt = _LIST_FIELDS[atype][0], _LIST_FIELDS[atype][1:]
        return isinstance(obj.get(need), list) and all(isinstance(obj[k], list) for k in opt if k in obj)
    return True


def _parse_reply(raw: str, forced: bool = False):
    """Action parser. Returns (type, payload, dialect, note, error).
    One action per turn. With several blocks, the first block is executed and the agent is told the rest were
    ignored; it is skipped only for a later VALID block of the same type (code that compiles, JSON that parses).
    On a forced-answer turn the last valid answer is preferred.
    Prose that merely mentions a tag ("I will output an <action type=\"code\"> block") yields an invalid block and
    is skipped in favour of the real one."""
    cands = _candidates(raw)
    if not cands:
        return None, None, None, None, "no action block found in your reply"
    if forced:
        answers = [c for c in cands if c[0] == "answer" and _valid(c[0], c[1])]
        if answers:
            c = answers[-1]
            return c[0], c[1], c[2], ("answer taken from a reply with several actions" if len(cands) > 1 else None), None
    # The first block is executed. It is skipped only for a later block of the SAME type when it is invalid
    # (prose that merely mentions the tag, followed by the real block); a reply never jumps to a different kind
    # of action (e.g. an invalid code block followed by an invented answer runs the code, which then reports
    # its error).
    c = cands[0]
    skipped = 0
    if not _valid(c[0], c[1]):
        later = [x for x in cands[1:] if x[0] == c[0] and _valid(x[0], x[1])]
        if later:
            skipped = cands.index(later[0])
            c = later[0]
    rest = [x for x in cands if x is not c and " ".join(x[1].split()) != " ".join(c[1].split())]
    note = None
    if rest:
        kinds = ", ".join(dict.fromkeys(x[0] for x in rest))
        note = (f"your reply contained {len(rest) + 1} action blocks; only one ({c[0]}) was executed"
                + (" (an earlier, unreadable block was skipped)" if skipped else "")
                + f"; the other(s) ({kinds}) were ignored. Write exactly ONE action per turn")
    return c[0], c[1], c[2], note, None


def _tag(text: str, tag: str) -> str:
    text = _strip_think(text)
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if not m:   # an unclosed block at the end of the reply
        m = re.search(rf"<{tag}>\s*(.*?)\s*$", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _norm_req(req: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (req or "").lower())).strip()[:60]


def build_agent_prompt(history: str = "stateless") -> str:
    if history == "full":
        return AGENT_PROMPT.replace(MEMORY_STATELESS, MEMORY_FULL)
    return AGENT_PROMPT


class MockAgent:
    """Scripted agent that measures, runs code, tests the oracle fix and submits the oracle answer (no API)."""

    def __init__(self):
        self.calls = 0
        self.sim = None

    def prime(self, sim: Simulator):
        self.calls = 0
        self.sim = sim

    def _marker_to_display(self, var: str, x: float) -> float:
        u = self.sim.scm.variables.get(var, {}).get("units") if self.sim.display else None
        return round(u["lo"] + float(x) / 100.0 * (u["hi"] - u["lo"]), 4) if u else x

    def _alias(self, actuator_id: str) -> str:
        return self.sim.scm.actuators[actuator_id].get("aliases", [actuator_id])[0]

    def generate(self, sysp: str, userp: str, max_new_tokens: Optional[int] = None) -> str:
        self.calls += 1
        outcome_alias = self.sim.scm.variables[self.sim.scm.outcome].get("aliases", [self.sim.scm.outcome])[0]
        proxy_alias = self.sim.scm.variables[self.sim.battery["true_mechanism_proxy"]].get("aliases", [self.sim.battery["true_mechanism_proxy"]])[0]
        if self.calls == 1:
            a = {"requests": [outcome_alias, proxy_alias]}
            return f'<reasoning>baseline</reasoning>\n<action type="measure">{json.dumps(a)}</action>\n<memory>Next: analyze</memory>'
        if self.calls == 2:
            code = ("df = experiment_1_df\n"
                    "print('rows', len(df), 'cols', list(df.columns))\n"
                    "print(df.corr(numeric_only=True).round(3))\n")
            return f'<reasoning>analyze raw data</reasoning>\n<action type="code">\n{code}\n</action>\n<memory>Next: test</memory>'
        gold_iv = self.sim.gold["intervention"]
        rec = [{"request": self._alias(aid), "value": self.sim.act_to_display(aid, val)}
               for aid, val in gold_iv.items() if not isinstance(val, dict)]
        policy_text = None
        gp = self.sim.gold.get("policy")
        if gp:
            treat_id = next((aid for aid, v in gold_iv.items() if isinstance(v, dict)), None)
            if treat_id is not None:
                marker = gp["marker"]
                policy_text = {"treatment": self._alias(treat_id),
                               "stratifier": self.sim.scm.variables[marker].get("aliases", [marker])[0],
                               "threshold": self._marker_to_display(marker, gp["threshold"]),
                               "dose_if_ge": self.sim.act_to_display(treat_id, gp["dose_if_ge"]),
                               "dose_if_lt": self.sim.act_to_display(treat_id, gp["dose_if_lt"])}
        bat = self.sim.battery
        if self.calls == 3 and rec:
            a = {"actions": [rec[0]], "measure": [outcome_alias, proxy_alias]}
            return f'<reasoning>test targeted lever</reasoning>\n<action type="intervene">{json.dumps(a)}</action>\n<memory>Next: answer</memory>'
        signs = {self._alias(aid): s for aid, s in bat["actuator_sign_predictions"].items()}
        decoy_aliases = [self.sim.scm.variables[d].get("aliases", [d])[0] for d in bat["confounded_decoys"]]
        ans = {"recommended_intervention_text": rec,
               "structured": {"true_mechanism_proxy": proxy_alias,
                              "confounded_decoys": decoy_aliases,
                              "actuator_sign_predictions": signs},
               "explanation": self.sim.gt["latent_plain_name"]}
        if policy_text:
            ans["recommended_policy_text"] = policy_text
        return f'<reasoning>done</reasoning>\n<action type="answer">{json.dumps(ans)}</action>\n<memory>done</memory>'


VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"


def build_llm(backend, model, temperature, max_new_tokens):
    if backend == "mock":
        return MockAgent()
    if backend == "bedrock":
        from llm_bedrock import BedrockLLM
        return BedrockLLM(model_id=model, temperature=temperature, max_new_tokens=max_new_tokens)
    if backend in ("openai", "vllm"):
        from llm_openai import OpenAILLM
        if backend == "vllm":
            base_url = os.environ.get("VLLM_BASE_URL", VLLM_DEFAULT_BASE_URL)
            api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        else:
            base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAILLM(model_name=model, base_url=base_url, api_key=api_key,
                         max_new_tokens=max_new_tokens,
                         temperature=None if temperature is None else temperature)
    raise ValueError(backend)


# The request parser is the same model for every agent, so all agents get the same data for the same request.
RESOLVER_MODEL = "us.anthropic.claude-opus-4-8"


def build_resolver_llm(agent_backend: str, no_resolver_llm: bool = False, resolver_model: str = RESOLVER_MODEL):
    """The LLM that maps free-text requests to the world's signals and controls (Bedrock). None means the
    deterministic alias matcher alone (the scripted agent, or --no-resolver-llm)."""
    if no_resolver_llm or agent_backend == "mock":
        return None
    from llm_bedrock import BedrockLLM
    return BedrockLLM(model_id=resolver_model, max_new_tokens=200)


def _resolve_answer_intervention(sim: Simulator, rec_text: List[Dict[str, Any]]) -> Dict[str, Any]:
    iv = {}
    echoes = []
    for item in rec_text or []:
        r = sim.resolver.resolve_answer_term(item.get("request", ""), "intervene", item.get("value"))
        echoes.append({"request": item.get("request", ""), **r.to_dict()})
        if r.ok and r.target_id:
            iv[r.target_id] = r.value
    return iv, echoes


def _resolve_answer_policy(sim: Simulator, pol_text: Dict[str, Any]) -> Dict[str, Any]:
    if not pol_text:
        return None
    if not isinstance(pol_text, dict):
        return {"_unresolved": True, "malformed": repr(pol_text)[:200]}
    tr = sim.resolver.resolve_answer_term(pol_text.get("treatment", ""), "intervene", pol_text.get("dose_if_ge", 100))
    st = sim.resolver.resolve_answer_term(pol_text.get("stratifier", ""), "measure")
    if not (tr.ok and tr.target_id and st.ok and st.target_id):
        return {"_unresolved": True, "treatment_echo": tr.to_dict(), "stratifier_echo": st.to_dict()}
    try:
        return {"treatment": tr.target_id, "stratifier": st.target_id,
                "threshold": float(pol_text.get("threshold", 50.0)),
                "dose_if_ge": float(pol_text.get("dose_if_ge", 0.0)),
                "dose_if_lt": float(pol_text.get("dose_if_lt", 0.0))}
    except (TypeError, ValueError):     # e.g. a threshold given as a list -> policy unresolved, not a crash
        return {"_unresolved": True, "malformed": repr(pol_text)[:200]}


_TAG_LINE_RE = re.compile(r"\s*</?[A-Za-z_][\w:.-]*\s*/?>\s*")


def _not_executed(reason: str, previous: str, format_hint: bool = True) -> str:
    """What the agent sees after a reply that could not be executed."""
    return ("YOUR LAST REPLY WAS NOT EXECUTED: " + reason + ".\n"
            + ("Write exactly ONE action block per turn, closed with </action>, followed by your <memory> block, e.g.\n"
               "<action type=\"code\">print(experiment_1_df.describe())</action>\n" if format_hint else "")
            + "No experiment was used.\n\n"
            "THE RESULT BEFORE THAT (repeated):\n" + previous)


def _data_line(experiments: Dict[int, Dict[str, Any]], sim: Simulator) -> str:
    if not experiments:
        return "data for code: none yet (run a measure or intervene first)"
    parts = []
    for eid in sorted(experiments):
        e = experiments[eid]
        if e["mode"] == "intervene" and not e["intervention"]:
            parts.append(f"experiment_{eid}_df (baseline batch: representative units, no control changed)")
            continue
        if e["mode"] == "intervene" and e["intervention"]:
            what = "intervention: " + ", ".join(f"{sim.csv_columns.get('do_' + a, sim.act_label.get(a, a))}={v}"
                                                for a, v in e["intervention"].items())
        else:
            what = "observational"
        parts.append(f"experiment_{eid}_df ({what})")
    return "data for code: " + "; ".join(parts)


def run_world(sim: Simulator, llm, max_turns: int, budget: int, verbose: bool,
              history: str = "stateless") -> Dict[str, Any]:
    """Run one episode. history="stateless" (the benchmark protocol) shows the agent the situation report, the
    result of its last action and its own memory each turn; history="full" (ablation) sends the whole conversation."""
    assert history in ("stateless", "full"), history
    system_prompt = build_agent_prompt(history)
    conv: List[Dict[str, str]] = []  # history="full": earlier user/assistant messages
    pub = sim.public()
    if isinstance(llm, MockAgent):
        llm.prime(sim)
    latest = "(none yet)"            # what "RESULT OF YOUR LAST ACTION" shows
    last_result = "(none yet)"       # the last executed action's result (repeated after an unexecuted reply)
    memory = "(empty)"
    used = 0
    n_interventions = 0
    turns = []
    grade = None
    csv_map: Dict[str, str] = {}
    carried_vars: Dict[str, Any] = {}
    rejected_counts: Dict[str, int] = {}
    forced_answer = False

    for turn in range(max_turns):
        remaining = budget - used
        turns_left = max_turns - turn

        directives = []
        # only the turn limit forces the answer; with the budget spent and turns left, code (free) is still
        # allowed and measure/intervene are refused without charge
        forced_now = turns_left <= 2
        budget_spent = remaining <= 0
        if forced_now:
            directives.append(
                "STOP EXPLORING. You are out of turns. Submit your final answer NOW (an answer action), "
                "using the evidence you have. Do not measure or run code this turn.")
            forced_answer = True
        elif budget_spent:
            directives.append(
                "Your experiment budget is spent: measure and intervene can no longer run. You can still analyse "
                "your data with code (free), and you must answer before the turn limit.")
        if n_interventions == 0 and used >= 3:
            directives.append(
                "You have not run any INTERVENTION yet. Observation and code alone "
                "cannot establish causation or find the fix — apply a control/"
                "additive with an intervene action to test a cause and to "
                "find what actually improves the outcome.")
        stuck = [req for req, c in rejected_counts.items() if c >= 3]
        if stuck:
            directives.append(
                "Several of your requests have been rejected repeatedly because "
                "those things are NOT part of this system (no such measurement or "
                "actuator exists): " + "; ".join(sorted(stuck)[:5]) + ". Stop "
                "asking for them and reconsider your hypothesis using only what IS "
                "available.")
        directive_block = ("\nDIRECTIVES\n" + "\n".join(f"- {d}" for d in directives) + "\n") if directives else ""

        if history == "full" and turn > 0:
            # the situation report, earlier results and the agent's replies are already in the conversation
            user = f"""BUDGET: {used}/{budget} experiments used; {remaining} left. (code does not cost budget)
TURNS: {turn}/{max_turns} used; {turns_left} left. INTERVENTIONS run so far: {n_interventions}.
{_data_line(sim.experiments, sim)}
{directive_block}
RESULT OF YOUR LAST ACTION
{latest}
"""
        else:
            user = f"""SITUATION REPORT
{pub['scenario']}

OUTCOME OF INTEREST: {pub['outcome_name']} ({pub['outcome_direction']})

BUDGET: {used}/{budget} experiments used; {remaining} left. (code does not cost budget)
TURNS: {turn}/{max_turns} used; {turns_left} left. INTERVENTIONS run so far: {n_interventions}.
{_data_line(sim.experiments, sim)}
{directive_block}
RESULT OF YOUR LAST ACTION (shown this turn only: copy what you need into <memory>)
{latest}

YOUR MEMORY
{memory}
"""
            if history == "full":
                user = user.replace(" (shown this turn only: copy what you need into <memory>)", "")
        t0 = time.time()
        try:
            if history == "full" and hasattr(llm, "generate_messages"):
                msgs = [{"role": "system", "content": system_prompt}] + [dict(m) for m in conv] \
                    + [{"role": "user", "content": user}]
                if len(msgs) > 2:
                    msgs[-2]["cache"] = True   # Bedrock/Claude: cache the conversation prefix (ignored elsewhere)
                raw = llm.generate_messages(msgs)
            else:
                raw = llm.generate(system_prompt, user)
        except RuntimeError as e:
            # a Bedrock reply with no text block (e.g. gpt-oss returning only reasoning) is an empty turn
            if "no text in Bedrock reply" not in str(e):
                raise
            logger.warning("empty agent reply (%s); counted as a turn with no action", e)
            raw = ""
        dt = round(time.time() - t0, 2)
        if history == "full":
            conv += [{"role": "user", "content": user}, {"role": "assistant", "content": raw or "(empty reply)"}]
        finish_reason = getattr(llm, "last_finish_reason", None)
        memory = _tag(raw, "memory") or memory
        reasoning = _tag(raw, "reasoning")
        atype, payload, dialect, note, perr = _parse_reply(raw, forced=forced_now or budget_spent)
        rec: Dict[str, Any] = {"turn": turn, "latency_s": dt, "finish_reason": finish_reason,
                               "reasoning": reasoning, "memory": memory, "raw": raw,
                               "model_reasoning": getattr(llm, "last_reasoning", None),
                               "usage": getattr(llm, "last_usage", None), "prompt_chars": len(user)}
        if getattr(llm, "last_served_model", None):
            rec["served_model"] = llm.last_served_model
        if getattr(llm, "last_text_blocks", None):
            rec["text_blocks"] = llm.last_text_blocks      # audit: a reply with several text blocks (last one is used)
        if dialect == "invoke":
            rec["parsed_via"] = "invoke"
        if note:
            rec["parse_note"] = note
        if atype is None:
            rec["error"] = "no action parsed"
            rec["error_detail"] = perr
            latest = _not_executed(perr, last_result)
            rec["shown_next"] = latest
            turns.append(rec)
            continue
        data: Any = {}
        if atype in ("measure", "intervene", "answer"):
            try:
                data, repaired = _loads_lenient(payload)
                if repaired:
                    rec["json_repaired"] = True
                if isinstance(data, list):          # a bare list: the action's list field
                    if atype == "answer":
                        raise ValueError("an answer must be a JSON object")
                    data = {"requests" if atype == "measure" else "actions": data}
            except ValueError as e:
                # unreadable JSON is not executed (and no experiment is charged); the agent is told
                rec["error"] = f"json parse: {e}"
                rec["attempted_action_type"] = atype
                latest = _not_executed(f"the JSON in your {atype} action could not be read", last_result)
                rec["shown_next"] = latest
                turns.append(rec)
                continue
        if atype in _LIST_FIELDS and isinstance(data, dict) and not _valid(atype, json.dumps(data)):
            # a measure without a "requests" list, or an intervene without an "actions" list, is not run
            need = _LIST_FIELDS[atype][0]
            rec["error"] = f"{atype} without a {need} list"
            rec["attempted_action_type"] = atype
            latest = _not_executed(f'your {atype} action needs a "{need}" list'
                                   + (' (use "actions": [] for a baseline batch with no control changed)'
                                      if atype == "intervene" else ""), last_result)
            rec["shown_next"] = latest
            turns.append(rec)
            continue
        if atype in ("measure", "intervene") and budget_spent:
            rec["error"] = "budget spent"
            rec["attempted_action_type"] = atype
            latest = _not_executed(f"the experiment budget is spent ({used}/{budget}), so a {atype} can no longer "
                                   "run. Use code to analyse your data, or answer", last_result, format_hint=False)
            rec["shown_next"] = latest
            turns.append(rec)
            continue
        rec["action_type"] = atype

        if atype == "answer":
            # normalise the answer's fields so a malformed answer is graded (as what it validly says) instead
            # of crashing the batch; the original is kept in answer_raw.
            if not isinstance(data, dict):
                rec["grade_error"] = f"answer is a {type(data).__name__}, not an object"
                data = {}
            data = dict(data)
            for k, typ, empty in (("recommended_intervention_text", list, []), ("structured", dict, {}),
                                  ("recommended_policy_text", dict, None)):
                if k in data and data[k] is not None and not isinstance(data[k], typ):
                    rec["grade_error"] = f"{k} is a {type(data[k]).__name__}"
                    data[k] = empty
            data["recommended_intervention_text"] = [x for x in data.get("recommended_intervention_text") or []
                                                     if isinstance(x, dict)]
            if isinstance(data.get("structured"), dict):
                st_ = data["structured"] = dict(data["structured"])
                px = st_.get("true_mechanism_proxy")
                if px is not None and not isinstance(px, str):
                    st_["true_mechanism_proxy"] = next((x for x in px if isinstance(x, str)), "") \
                        if isinstance(px, list) else str(px)
                dc = st_.get("confounded_decoys")
                if dc is not None and not isinstance(dc, list):
                    st_["confounded_decoys"] = [dc] if isinstance(dc, str) else []
                if isinstance(st_.get("confounded_decoys"), list):
                    st_["confounded_decoys"] = [x for x in st_["confounded_decoys"] if isinstance(x, str)]
                sg = st_.get("actuator_sign_predictions")
                if sg is not None and not isinstance(sg, dict):
                    st_["actuator_sign_predictions"] = {}
            try:
                iv, echoes = _resolve_answer_intervention(sim, data.get("recommended_intervention_text", []))
                pol = _resolve_answer_policy(sim, data.get("recommended_policy_text"))
                answer = {"recommended_intervention": iv,
                          "structured": _translate_structured(sim, data.get("structured", {})),
                          "explanation": data.get("explanation", "")}
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning("malformed answer (%s); graded as empty", e)
                rec["grade_error"] = f"{type(e).__name__}: {e}"
                iv, echoes, pol = {}, [], None
                answer = {"recommended_intervention": {}, "structured": {}, "explanation": str(data.get("explanation", ""))}
            if pol and not pol.get("_unresolved"):
                answer["recommended_policy"] = pol
            try:
                grade = sim.grade(answer)
            except (TypeError, ValueError) as e:
                logger.warning("answer could not be graded as given (%s); dropping malformed values", e)
                rec["grade_error"] = f"{type(e).__name__}: {e}"
                answer.pop("recommended_policy", None)
                answer["recommended_intervention"] = {
                    k: v for k, v in (answer.get("recommended_intervention") or {}).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}
                try:
                    grade = sim.grade(answer)
                except (TypeError, ValueError):
                    answer["recommended_intervention"] = {}
                    grade = sim.grade(answer)
            rec["policy_resolution"] = pol
            rec["answer_raw"] = data
            rec["resolved_intervention"] = iv
            rec["answer_resolution_echo"] = echoes
            rec["grade"] = grade
            turns.append(rec)
            if verbose:
                logger.info("turn %d ANSWER accepted=%s A=%s B=%.2f gap=%.2f",
                            turn, grade["accepted"], grade["part_a_utility_ok"],
                            grade["battery_fraction"], grade["utility_gap"])
            break

        if atype == "give_up":
            rec["give_up_reason"] = payload
            turns.append(rec)
            break

        if atype == "code":
            from code_sandbox import run_code
            # drop lines that are only an XML-like tag (models append invented closers such as
            # </parameter_text>); such a line is never valid Python and made the whole cell a SyntaxError
            payload = "\n".join(l for l in payload.split("\n") if not _TAG_LINE_RE.fullmatch(l))
            out, new_vars = run_code(payload, csv_map, carried_vars)
            carried_vars.update(new_vars)
            rec["code"] = payload
            rec["code_output"] = out
            latest = last_result = "CODE OUTPUT:\n" + out + (f"\n[note: {note}]" if note else "")
            if verbose:
                logger.info("turn %d CODE (%d chars out, %d files)", turn, len(out), len(csv_map))
            turns.append(rec)
            continue

        if not isinstance(data, dict):
            data = {}
        if atype == "measure":
            result = sim.measure(data.get("requests", []) if isinstance(data.get("requests", []), list) else [])
            if result.get("raw_csv"):
                used += 1
                csv_map[f"experiment_{result['experiment_id']}_csv"] = result["raw_csv"]
            for e in result["resolutions"]:
                if not e["ok"] and e.get("kind") == "reject" and not e.get("candidates"):
                    key = _norm_req(e["request"])
                    rejected_counts[key] = rejected_counts.get(key, 0) + 1
            rec["result"] = result
            if verbose:
                ok = [e["target_id"] for e in result["resolutions"] if e["ok"]]
                logger.info("turn %d MEASURE -> %s (used %d/%d)", turn, ok, used, budget)
        else:  # intervene
            acts = data.get("actions", [])
            meas = data.get("measure", [])
            result = sim.intervene(acts if isinstance(acts, list) else [], meas if isinstance(meas, list) else [])
            if result.get("raw_csv"):
                used += 1
                csv_map[f"experiment_{result['experiment_id']}_csv"] = result["raw_csv"]
            if result.get("applied_intervention"):
                n_interventions += 1
            for e in result["action_resolutions"]:
                if not e["ok"] and e.get("kind") in ("reject", "signal") and not e.get("candidates"):
                    key = _norm_req(e["request"])
                    rejected_counts[key] = rejected_counts.get(key, 0) + 1
            rec["result"] = result
            if verbose:
                logger.info("turn %d INTERVENE %s (used %d/%d, interventions=%d)", turn,
                            result["applied_intervention"], used, budget, n_interventions)
        view = sim.agent_view(result)
        if not result.get("raw_csv"):
            # nothing could be run (every request rejected): no experiment is used, and the agent is told
            view["note"] = "Nothing was run (no request could be carried out), so no experiment was used."
            rec["not_charged"] = True
        latest = last_result = json.dumps(view, default=str) + (f"\n[note: {note}]" if note else "")
        rec["shown_to_agent"] = latest
        turns.append(rec)

    if grade is None:
        synth = {"recommended_intervention": {}, "structured": {}, "explanation": memory}
        grade = sim.grade(synth)
        grade["forced_no_answer"] = True
        turns.append({"turn": "post", "action_type": "forced_answer",
                      "note": "hit turn cap without answering; graded empty answer from memory",
                      "grade": grade})

    return {"world_id": sim.world["world_id"], "domain": sim.world["domain"], "history": history,
            "queries_used": used, "interventions_run": n_interventions,
            "answered": not grade.get("forced_no_answer", False),
            "hit_turn_cap": forced_answer and grade.get("forced_no_answer", False) if grade else False,
            "grade": grade, "gold": {"intervention": sim.gold["intervention"],
                                     "expected_utility": round(sim.gold["expected_utility"], 2)},
            "turns": turns}


def _translate_structured(sim: Simulator, st: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    pm = sim.resolver.resolve_answer_term(st.get("true_mechanism_proxy", ""), "measure") if st.get("true_mechanism_proxy") else None
    out["true_mechanism_proxy"] = pm.target_id if (pm and pm.ok) else st.get("true_mechanism_proxy")
    decoys = []
    for d in st.get("confounded_decoys", []) or []:
        dm = sim.resolver.resolve_answer_term(d, "measure")
        decoys.append(dm.target_id if dm.ok else d)
    out["confounded_decoys"] = decoys
    signs = {}
    for req, sign in (st.get("actuator_sign_predictions", {}) or {}).items():
        ar = sim.resolver.resolve_answer_term(req, "intervene")
        key = ar.target_id if (ar.ok and ar.target_id) else req
        signs[key] = sign
    out["actuator_sign_predictions"] = signs
    return out


def load_world_file(path: str) -> tuple:
    from scm import WorldSCM
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        rec = json.load(f)
    world = {"world_id": rec["world_id"], "domain": rec["domain"],
             "scenario": rec["scenario"], "scm": WorldSCM.from_dict(rec["scm"]),
             "ground_truth": rec["ground_truth"]}
    pre = {"gold": rec["oracle"]["gold"], "battery": rec["oracle"]["counterfactual_battery"]}
    return world, pre
