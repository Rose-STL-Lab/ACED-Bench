#!/usr/bin/env python3
"""Run an agent on every world in a folder; write per-world results and summary.json."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from simulator import Simulator
from agent_loop import RESOLVER_MODEL, build_llm, build_resolver_llm, run_world, load_world_file

logger = logging.getLogger("run_benchmark")


def summarize_result(res: Dict[str, Any], world: Dict[str, Any]) -> Dict[str, Any]:
    g = res.get("grade") or {}
    n_length = sum(1 for t in res.get("turns", []) if t.get("finish_reason") == "length")
    return {
        "world_id": world["world_id"], "domain": world.get("domain"),
        "structure": (world.get("ground_truth") or {}).get("_archetype"),
        "accepted": bool(g.get("accepted")),
        "fix_ok": bool(g.get("part_a_utility_ok")),
        "benefit_recovered": g.get("benefit_recovered"),
        "mechanism_score": round(float(g.get("battery_fraction") or 0), 2),
        "gap": round(float(g.get("utility_gap") or 0), 1),
        "interventions": res.get("interventions_run"),
        "queries": res.get("queries_used"),
        "hit_cap": bool(res.get("hit_turn_cap")),
        "n_length_truncated_turns": n_length,
        "wall_s": res.get("wall_seconds"),
    }


def main():
    ap = argparse.ArgumentParser(description="Run an agent on every ACED-Repair world in a folder.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--worlds-dir", required=True)
    ap.add_argument("--backend", choices=["bedrock", "openai", "vllm", "mock"], default="bedrock")
    ap.add_argument("--model", default=None, help="model id for the backend (not needed for --backend mock)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature; None = the model's preset")
    ap.add_argument("--max-new-tokens", type=int, default=32768,
                    help="output cap per agent reply; set it to the model's maximum output length")
    ap.add_argument("--budget", type=int, default=15, help="experiments per world")
    ap.add_argument("--max-turns", type=int, default=32, help="turns per world (code turns count here but not in --budget)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="number of worlds to run in parallel (threads); ignored for --backend mock")
    ap.add_argument("--resolver-model", default=RESOLVER_MODEL,
                    help="Bedrock model id of the request parser (the same for every agent model)")
    ap.add_argument("--no-resolver-llm", action="store_true",
                    help="parse requests with the deterministic alias matcher alone (no Bedrock calls for parsing)")
    ap.add_argument("--history", choices=["stateless", "full"], default="stateless",
                    help="'stateless' (the benchmark protocol) shows the last result and the agent's memory each turn; "
                         "'full' (ablation) sends the whole conversation")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--force", action="store_true",
                    help="re-run every world; by default worlds that already have a result_<world>.json in outdir are "
                         "skipped, so an interrupted run resumes (a world that was cut off restarts from its first turn)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    if args.backend != "mock" and not args.model:
        ap.error("--model is required for this backend")
    paths = sorted(glob.glob(os.path.join(args.worlds_dir, "world_*.json*")))
    if not paths:
        ap.error(f"no world_*.json(.gz) in {args.worlds_dir}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _wid_of(p):
        try:
            return load_world_file(p)[0]["world_id"]
        except Exception:
            return Path(p).name.split(".")[0].replace("world_", "")

    settings = {"model": args.model, "budget": args.budget, "max_turns": args.max_turns, "history": args.history}
    todo, done_paths = [], []
    for path in paths:
        wid = _wid_of(path)
        result_file = outdir / f"result_{wid}.json"
        if not args.force and result_file.exists():
            old = json.load(open(result_file)).get("settings")
            if old != settings:
                ap.error(f"{result_file} was made with different settings ({old}); use another --outdir, or --force")
            done_paths.append(path)
        else:
            todo.append(path)
    if done_paths:
        logger.info("resume: %d world(s) already done, %d to run", len(done_paths), len(todo))
        print(f"resume: skipping {len(done_paths)} completed world(s), running {len(todo)}")

    llm = resolver_llm = None
    if todo:
        llm = build_llm(args.backend, args.model, args.temperature, args.max_new_tokens)
        resolver_llm = build_resolver_llm(args.backend, args.no_resolver_llm, args.resolver_model)

    def _run_one(path: str) -> Dict[str, Any]:
        world, pre = load_world_file(path)
        wid = world["world_id"]
        result_file = outdir / f"result_{wid}.json"
        if path in done_paths:
            try:
                with open(result_file, encoding="utf-8") as f:
                    res = json.load(f)
                return summarize_result(res, world)
            except Exception as e:
                logger.warning("resume: could not read %s (%s); re-running", result_file, e)
        # the world id names the causal structure, so the experiment folder gets a hashed name
        data_dir = str(outdir / f"data_{hashlib.sha1(wid.encode()).hexdigest()[:10]}")
        sim = Simulator(world, resolver_llm=resolver_llm, data_dir=data_dir, precomputed=pre)
        logger.info("=== %s (gold util %.1f) ===", wid, sim.gold["expected_utility"])
        t0 = time.time()
        res = run_world(sim, llm, args.max_turns, args.budget, args.verbose, history=args.history)
        res["wall_seconds"] = round(time.time() - t0, 1)
        res["model"] = args.model
        res["settings"] = settings
        res["world_file"] = path
        res["data_dir"] = data_dir
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        return summarize_result(res, world)

    rows: List[Dict[str, Any]] = []
    failed: List[str] = []
    t_start = time.time()
    conc = 1 if args.backend == "mock" else max(1, args.concurrency)
    print(f"running {len(paths)} world(s) with concurrency={conc}", flush=True)

    def _report(n_done, row):
        print(f"  [{n_done}/{len(paths)}] {row['world_id']} accepted={row['accepted']} fix={row['fix_ok']} "
              f"mechanism={row['mechanism_score']:.2f}", flush=True)

    def _failed(path, e):
        # one world's error (e.g. an API timeout after retries) does not stop the run; rerunning picks it up
        logger.error("world %s failed: %s: %s", path, type(e).__name__, e)
        print(f"  FAILED {os.path.basename(path)}: {type(e).__name__}: {e}", flush=True)
        failed.append(path)

    if conc > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(_run_one, p): p for p in paths}
            for fut in as_completed(futs):
                try:
                    rows.append(fut.result())
                    _report(len(rows), rows[-1])
                except Exception as e:
                    _failed(futs[fut], e)
    else:
        for path in paths:
            try:
                rows.append(_run_one(path))
                _report(len(rows), rows[-1])
            except Exception as e:
                _failed(path, e)

    n = len(rows)
    acc = sum(r["accepted"] for r in rows)
    fix = sum(r["fix_ok"] for r in rows)
    caps = sum(r["hit_cap"] for r in rows)
    no_iv = sum(1 for r in rows if (r["interventions"] or 0) == 0)
    summary = {
        "model": args.model, "backend": args.backend, "resolver_model": None if resolver_llm is None else args.resolver_model,
        "history": args.history, "budget": args.budget, "max_turns": args.max_turns,
        "n_worlds": n, "accepted": acc, "acceptance_rate": round(acc / n, 3) if n else 0,
        "fix_ok": fix, "fix_rate": round(fix / n, 3) if n else 0,
        "mean_mechanism_score": round(sum(r["mechanism_score"] for r in rows) / n, 3) if n else 0,
        "hit_turn_cap": caps, "no_intervention_runs": no_iv, "failed_worlds": failed,
        "wall_seconds": round(time.time() - t_start, 1), "rows": sorted(rows, key=lambda r: r["world_id"]),
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ({args.model or args.backend}, {n} worlds) ===")
    print(f"fix {fix}/{n} ({100 * fix / max(n, 1):.1f}%) | accepted {acc}/{n} ({100 * acc / max(n, 1):.1f}%) | "
          f"mean mechanism score {summary['mean_mechanism_score']} | hit turn cap {caps} | "
          f"no intervention {no_iv} | {summary['wall_seconds']}s")
    print(f"wrote per-world results and summary.json to {outdir}/")
    if failed:
        print(f"{len(failed)} world(s) failed and are not in the summary; run the same command again to retry them.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
