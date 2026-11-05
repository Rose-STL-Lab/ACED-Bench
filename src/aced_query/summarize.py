#!/usr/bin/env python3
"""Accuracy per agent on the Qual and Core question sets for finished ACED-Query runs.

  python summarize.py out/query/opus55 [out/query/sonnet55 ...]       # reads <run>/eval/eval_*.json

Qual = the yes/no structural questions; Core = the causal-list structural questions plus all causal-decision questions.
Agents: guess (guess-only, no data), active, active_code (Active+Code), modular.

It reads the layout run_model.sh writes: <run>/eval/eval_<name>.json, where <name> contains _guess_, _active_,
_active_code_ or _modular_ (the agent) and "decision" for causal-decision questions.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YESNO = {"causal_effect", "direct_marginal", "chain_marginal", "fork_marginal", "v_structure_marginal",
         "other_marginal", "chain_conditional", "fork_conditional", "v_structure_conditional", "other_conditional"}
LIST = {"all_causes_of", "all_effects_of"}
AGENTS = ["guess", "active", "active_code", "modular"]
# Structural questions count only on the 30 guess_struct worlds (core_struct's 10 worlds are among them).
SUBSET = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(HERE, "..", "..", "data", "query", "guess_struct", "world_*.json"))}


def rows(path, decision):
    """(group, world, correct) for every scored question in one eval file."""
    out = []
    for e in json.load(open(path)).get("evaluated", []):
        qt = e.get("question_type")
        group = "core" if decision or qt in LIST else "qual" if qt in YESNO else None
        world = e.get("world_name") or os.path.basename(str(e.get("world_file", "")))[:-5]
        if group and (decision or world in SUBSET):
            out.append((group, world, int(bool((e.get("eval") or {}).get("correct")))))
    return out


def agent_of(name):
    if "_guess_" in name:
        return "guess"
    for a in ("active_code", "modular", "active"):
        if f"_{a}_" in name:
            return a
    return None


def cells(files):
    """{agent: {group: [correct, total, worlds]}} over a list of eval files."""
    acc = {}
    for f in files:
        name = os.path.basename(f)
        a = agent_of(name)
        if a is None:
            continue
        for g, w, c in rows(f, decision="decision" in name):
            cell = acc.setdefault(a, {}).setdefault(g, [0, 0, set()])
            cell[0] += c
            cell[1] += 1
            cell[2].add(w)
    return {a: {g: [c, n, len(ws)] for g, (c, n, ws) in gs.items()} for a, gs in acc.items()}


def show(table):
    head = f"{'run':14s}" + "".join(f"{a + ' Qual':>18s}{a + ' Core':>18s}" for a in AGENTS)
    print(head)
    print("-" * len(head))
    for run, c in table.items():
        line = f"{run:14s}"
        for a in AGENTS:
            for g in ("qual", "core"):
                x = c.get(a, {}).get(g)
                line += f"{'-':>18s}" if not x else f"{x[0] / x[1]:>11.2f} ({x[1]:3d})"
        print(line)


if __name__ == "__main__":
    if not sys.argv[1:]:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        found = [f for f in glob.glob(os.path.join(d, "eval", "eval_*.json")) if agent_of(os.path.basename(f))]
        if not found:
            print(f"warning: no eval/eval_*.json files named by agent in {d} (see the layout above)", file=sys.stderr)
    show({os.path.basename(os.path.normpath(d)): cells(glob.glob(os.path.join(d, "eval", "eval_*.json")))
          for d in sys.argv[1:]})
    print("\nNumbers: accuracy (number of questions).")
