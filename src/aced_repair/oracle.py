#!/usr/bin/env python3
"""The oracle: expected utility, the best fix ("gold"), the scoring key ("battery"), and grading.

The gold fix and the battery are stored with every world; optimal_gold and counterfactual_battery recompute them."""
from __future__ import annotations

import itertools
from typing import Any, Dict, Optional

import numpy as np

from scm import WorldSCM


def expected_utility(scm: WorldSCM, intervention: Dict[str, Any], *, n: int, seed: int) -> float:
    vals = scm.sample(n, intervention=intervention, seed=seed)
    return float(np.mean(scm.utility(vals)))


def _golden_section(scm, act_id, other, a, b, *, n, seed, iters=22):
    gr = (np.sqrt(5) - 1) / 2

    def f(x):
        iv = dict(other)
        iv[act_id] = x
        return expected_utility(scm, iv, n=n, seed=seed)

    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    x = (a + b) / 2
    return x, f(x)


def _policy_intervention(sp: Dict[str, Any], marker_thresh: float,
                         dose_ge: float, dose_lt: float) -> Dict[str, Any]:
    return {sp["treatment_actuator"]: {"policy": {
        "stratifier": sp["marker"], "threshold": marker_thresh,
        "dose_if_ge": dose_ge, "dose_if_lt": dose_lt}}}


def optimal_policy(scm: WorldSCM, sp: Dict[str, Any], *, n=15000, seed=12345,
                   coarse=7, base_intervention: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_iv = dict(base_intervention or {})
    base = expected_utility(scm, base_iv, n=n, seed=seed)
    mvals = scm.sample(20000, seed=seed + 3, select=True)[sp["marker"]]
    thr_grid = [float(np.percentile(mvals, q)) for q in (30, 40, 50, 60, 70)]
    lo, hi = scm.actuators[sp["treatment_actuator"]]["range"]
    dose_grid = [round(x, 2) for x in np.linspace(lo, hi, coarse)]
    best = {"policy": None, "expected_utility": base}
    for thr in thr_grid:
        for dge in dose_grid:
            for dlt in dose_grid:
                iv = dict(base_iv)
                iv.update(_policy_intervention(sp, thr, dge, dlt))
                u = expected_utility(scm, iv, n=n, seed=seed)
                if u > best["expected_utility"]:
                    best = {"policy": {"marker": sp["marker"], "threshold": round(thr, 3),
                                       "dose_if_ge": dge, "dose_if_lt": dlt},
                            "expected_utility": u}
    best["baseline_utility"] = base
    return best


def _cont_grid(act, k):
    lo, hi = act["range"]
    return [round(float(x), 2) for x in np.linspace(lo, hi, k)]


def optimal_intervention(scm: WorldSCM, *, n=15000, seed=12345, eps=1.0, margin=0.3,
                         screen_points=11, combo_points=5, max_joint=3, top_refine=5) -> Dict[str, Any]:
    base = expected_utility(scm, {}, n=n, seed=seed)
    acts = list(scm.actuators)
    best_single, best_val, worst_single = {}, {}, {}
    for a in acts:
        act = scm.actuators[a]
        grid = _cont_grid(act, screen_points)
        us = [(expected_utility(scm, {a: v}, n=n, seed=seed), v) for v in grid]
        u, v = max(us, key=lambda t: t[0])
        best_single[a], best_val[a] = u, v
        worst_single[a] = min(t[0] for t in us)
    marg = {a: best_single[a] - base for a in acts}
    cand = [a for a in acts if marg[a] > eps]

    def _ext(a):
        return scm.actuators[a]["range"][1]
    for b in [a for a in acts if a not in cand]:
        for a in list(cand):
            u = expected_utility(scm, {a: best_val[a], b: _ext(b)}, n=n, seed=seed)
            if (u - best_single[a]) > eps and (u - base) > eps and (u - best_single[b]) > eps:
                cand.append(b)
                break

    def refine(iv):
        iv = dict(iv)
        for a in list(iv):
            lo, hi = scm.actuators[a]["range"]
            x, _ = _golden_section(scm, a, {k: v for k, v in iv.items() if k != a}, lo, hi, n=n, seed=seed)
            iv[a] = round(float(x), 2)
        return iv, expected_utility(scm, iv, n=n, seed=seed)

    scored = []
    for r in range(1, min(max_joint, len(cand)) + 1):
        for combo in itertools.combinations(cand, r):
            grids = [_cont_grid(scm.actuators[a], combo_points) for a in combo]
            bu, bp = -1e18, None
            for point in itertools.product(*grids):
                u = expected_utility(scm, dict(zip(combo, point)), n=n, seed=seed)
                if u > bu:
                    bu, bp = u, point
            scored.append((bu, dict(zip(combo, bp))))
    scored.sort(key=lambda t: -t[0])
    best_iv, best_u = {}, base
    for _, iv in scored[:top_refine]:
        riv, ru = refine(iv)
        if ru > best_u:
            best_iv, best_u = riv, ru
    for _ in range(4):
        improved = False
        for a in cand:
            if a in best_iv:
                trial = {k: v for k, v in best_iv.items() if k != a}
                tu = expected_utility(scm, trial, n=n, seed=seed)
            else:
                trial, tu = refine(dict(best_iv, **{a: best_val[a]}))
            if tu > best_u + margin:
                best_iv, best_u, improved = trial, tu, True
        if not improved:
            break
    best_iv, best_u = refine(best_iv) if best_iv else (best_iv, best_u)
    tol = max(margin, 0.01 * max(0.0, best_u - base))
    pruned = True
    while pruned and len(best_iv) > 1:
        pruned = False
        for a in sorted(best_iv, key=lambda k: best_single[k] if k in best_single else 0.0):
            trial, tu = refine({k: v for k, v in best_iv.items() if k != a})
            if tu >= best_u - tol:
                best_iv, best_u, pruned = trial, tu, True
                break
    return {"intervention": best_iv, "expected_utility": best_u, "baseline_utility": base,
            "active_actuators": cand, "single_utility": {a: round(best_single[a] - base, 3) for a in acts},
            "single_range": {a: [round(worst_single[a] - base, 3), round(best_single[a] - base, 3)] for a in acts},
            "single_best_value": best_val}


def optimal_gold(world: Dict[str, Any], *, n=15000, seed=12345) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    gold = optimal_intervention(scm, n=n, seed=seed)
    sp = world["ground_truth"].get("subtype_policy")
    if not sp:
        return gold
    base_iv = {k: v for k, v in gold["intervention"].items()
               if k != sp["treatment_actuator"]}
    pol = optimal_policy(scm, sp, n=n, seed=seed, base_intervention=base_iv)
    if pol["policy"] is not None and pol["expected_utility"] > gold["expected_utility"]:
        merged = dict(base_iv)
        merged.update(_policy_intervention(
            sp, pol["policy"]["threshold"],
            pol["policy"]["dose_if_ge"], pol["policy"]["dose_if_lt"]))
        return {"intervention": merged,
                "policy": pol["policy"],
                "base_intervention": base_iv,
                "expected_utility": pol["expected_utility"],
                "baseline_utility": gold["baseline_utility"],
                "active_actuators": gold["active_actuators"],
                **{k: gold[k] for k in ("single_utility", "single_range", "single_best_value") if k in gold},
                "is_conditional_policy": True}
    return gold


def _canon_sign(s):
    if s is None:
        return None
    if isinstance(s, bool):
        return "+" if s else "0"
    if isinstance(s, (int, float)):
        return "+" if s > 0 else ("-" if s < 0 else "0")
    t = str(s).strip().lower()
    if t in ("+", "1", "+1", "pos", "positive", "up", "increase", "increases", "higher"):
        return "+"
    if t in ("-", "-1", "neg", "negative", "down", "decrease", "decreases", "lower"):
        return "-"
    if t in ("0", "0.0", "none", "null", "no effect", "no_effect", "neutral", "flat", "zero", "skip"):
        return "0"
    return t


def counterfactual_battery(world: Dict[str, Any], *, n=15000, seed=777) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    eps = 0.5
    base = expected_utility(scm, {}, n=n, seed=seed)
    # a control's sign: "+" if some setting improves the objective, "-" if every change worsens it, "0" otherwise
    act_signs = {}
    for aid, act in scm.actuators.items():
        lo_, hi_ = act["range"]
        us = [expected_utility(scm, {aid: lo_ + k * (hi_ - lo_) / 10.0}, n=n, seed=seed) - base for k in range(11)]
        act_signs[aid] = "+" if max(us) > eps else ("-" if min(us) < -eps else "0")
        if aid == ((gt.get("subtype_policy") or {}).get("treatment_actuator")):
            act_signs[aid] = "+"

    tgt = gt["targeted_actuator"]
    thi = scm.actuators[tgt]["range"][1]
    excluded = set(gt.get("_selection_nodes", []))
    excluded.add(scm.outcome)
    if scm.goal:
        excluded.add(scm.goal)
    measurables = [nm for nm, s in scm.variables.items()
                   if (s["kind"] in ("observable",) or s.get("measurable"))
                   and nm != scm.outcome and nm not in excluded]
    base_obs = scm.measure(scm.sample(n, seed=seed + 1), measurables, seed=seed + 2)
    do_obs = scm.measure(scm.sample(n, intervention={tgt: thi}, seed=seed + 1),
                         measurables, seed=seed + 2)
    valid_proxies = []
    for nm in measurables:
        shift = abs(float(do_obs[nm].mean() - base_obs[nm].mean()))
        sd = float(base_obs[nm].std()) + 1e-9
        if shift / sd > 0.5:
            valid_proxies.append(nm)
    return {
        "sign_scope": sorted(a for a, r in gt["roles"].items() if r == "primary"),
        "true_mechanism_proxy": gt["true_mechanism_proxy"],
        "valid_mechanism_proxies": sorted(set(valid_proxies) | {gt["true_mechanism_proxy"]}),
        "confounded_decoys": sorted(gt["confounded_decoys"]),
        "actuator_sign_predictions": act_signs,
        "targeted_actuator": gt["targeted_actuator"],
        "symptom_trap_actuator": gt["symptom_trap_actuator"],
    }


def _score_battery(battery, answer, recommended=None):
    st = answer.get("structured", {}) or {}
    items = []
    proxy_set = set(battery["valid_mechanism_proxies"])
    items.append(("true_mechanism_proxy", st.get("true_mechanism_proxy") in proxy_set))
    agent_decoys = set(st.get("confounded_decoys", []))
    gt_decoys = set(battery["confounded_decoys"])
    if agent_decoys & proxy_set:
        decoy_score = 0.0
    elif not gt_decoys and not agent_decoys:
        decoy_score = 1.0
    else:
        union = gt_decoys | agent_decoys
        decoy_score = len(gt_decoys & agent_decoys) / len(union) if union else 1.0
    items.append(("confounded_decoys", decoy_score))
    pred = st.get("actuator_sign_predictions", {}) or {}
    gold_signs = battery["actuator_sign_predictions"]
    required = battery["targeted_actuator"]
    used_valid = [a for a in (recommended or {})
                  if gold_signs.get(a, "0") not in ("0", "skip")]
    if used_valid and pred.get(required) is None:
        required = used_valid[0]
    active = {aid for aid, s in gold_signs.items() if s in ("+", "-")} & set(battery["sign_scope"])
    # only controls of this world are scored
    scored = {required} | active | {k for k in pred if k in gold_signs}
    for aid in sorted(scored):
        gold = gold_signs.get(aid, "0")
        items.append((f"sign:{aid}", _canon_sign(pred.get(aid)) == _canon_sign(gold)))
    score_sum = sum(float(ok) for _, ok in items)
    return score_sum / len(items), items


def _normalize_policy_answer(scm: WorldSCM, answer: Dict[str, Any],
                             valid: Dict[str, Any]) -> Dict[str, Any]:
    pol = answer.get("recommended_policy")
    if not pol:
        return valid
    treat = pol.get("treatment")
    strat = pol.get("stratifier")
    if treat not in scm.actuators or strat not in scm.variables:
        return valid
    out = dict(valid)
    out[treat] = {"policy": {"stratifier": strat,
                             "threshold": float(pol.get("threshold", 50.0)),
                             "dose_if_ge": float(pol.get("dose_if_ge", 0.0)),
                             "dose_if_lt": float(pol.get("dose_if_lt", 0.0))}}
    return out


def mechanism_score(items, roles) -> float:
    """The mechanism score: the mechanism proxy, the confounded decoys, and the signs of the true levers (controls
    with role "primary"). Signs the agent gives for other controls are not scored, so listing many inert controls as
    "0" cannot raise the score."""
    signs = [float(ok) for name, ok in items[2:] if roles.get(name[len("sign:"):]) == "primary"]
    return (float(items[0][1]) + float(items[1][1]) + sum(signs)) / (2 + len(signs))


def grade(world: Dict[str, Any], answer: Dict[str, Any], gold: Dict[str, Any],
          battery: Dict[str, Any], *, n=30000, seed=999, tolerance=2.0) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    rec = answer.get("recommended_intervention", {}) or {}
    valid = {k: v for k, v in rec.items() if k in scm.actuators}
    valid = _normalize_policy_answer(scm, answer, valid)
    u = expected_utility(scm, valid, n=n, seed=seed)
    gold_u = gold["expected_utility"]
    base_u = gold.get("baseline_utility")
    if base_u is not None and (gold_u - base_u) > 1e-6:
        benefit = (u - base_u) / (gold_u - base_u)
        part_a = bool(benefit >= 0.90)
    else:
        benefit = None
        part_a = bool(u >= gold_u - tolerance)
    _, items = _score_battery(battery, answer, recommended=valid)
    frac = mechanism_score(items, world["ground_truth"]["roles"])
    part_b = bool(frac >= 0.8 - 1e-9)
    return {"accepted": bool(part_a and part_b), "part_a_utility_ok": part_a, "part_b_battery_ok": part_b,
            "recommended_intervention": valid, "recommended_utility": round(u, 3),
            "gold_intervention": gold["intervention"], "gold_utility": round(gold_u, 3),
            "baseline_utility": round(base_u, 3) if base_u is not None else None,
            "benefit_recovered": round(benefit, 3) if benefit is not None else None,
            "is_conditional_policy": bool(gold.get("is_conditional_policy")),
            "utility_gap": round(gold_u - u, 3), "battery_fraction": round(frac, 3),
            "battery_items": items}
