#!/usr/bin/env python3
"""Serves measurements and interventions on a world (physical units) and grades answers."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from scm import WorldSCM
from resolver import Resolver
from oracle import grade as _grade


def _camel(t: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z0-9]+", t) if w)


class Simulator:
    def __init__(self, world: Dict[str, Any], *, precomputed: Dict[str, Any], resolver_llm: Any = None,
                 n_sample: int = 400, data_dir: Optional[str] = None, display_units: bool = True):
        """precomputed: the oracle fix ("gold") and the scoring key ("battery") stored with the world."""
        self.world = world
        self.scm: WorldSCM = world["scm"]
        self.gt = world["ground_truth"]
        self.n_sample = n_sample
        self.display = bool(display_units) and any("units" in a for a in self.scm.actuators.values())
        self.resolver = Resolver(self.scm, llm=resolver_llm, display=self.display)
        self.gold = precomputed["gold"]
        self.battery = precomputed["battery"]
        self.act_label: Dict[str, str] = {aid: (act.get("aliases") or [aid])[0]
                                          for aid, act in self.scm.actuators.items()}
        # each control's CSV column is "Set" + its name in CamelCase (e.g. SetUpperCutoffVoltage), in the same style
        # as the signal columns; the prefix is the same for every control, so it says nothing about its role
        self.csv_columns: Dict[str, str] = {}
        taken = set(self.scm.variables)
        for aid, lab in self.act_label.items():
            n = "Set" + (_camel(lab) or aid)
            while n in taken:
                n += "_"
            taken.add(n)
            self.csv_columns[f"do_{aid}"] = n
            self.resolver.column_ids[n.lower()] = aid
        self._seed0 = 1000
        self._q = 0
        self.data_dir = Path(data_dir).resolve() if data_dir else None
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[int, Dict[str, Any]] = {}

    def act_to_internal(self, aid: str, v: Any) -> Any:
        u = self.scm.actuators.get(aid, {}).get("units") if self.display else None
        if u is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            return v
        return round(min(100.0, max(0.0, (float(v) - u["lo"]) / (u["hi"] - u["lo"]) * 100.0)), 4)

    def act_to_display(self, aid: str, x: Any) -> Any:
        u = self.scm.actuators.get(aid, {}).get("units") if self.display else None
        if u is None or isinstance(x, bool) or not isinstance(x, (int, float)):
            return x
        return round(u["lo"] + float(x) / 100.0 * (u["hi"] - u["lo"]), 4)

    def var_to_internal(self, var: str, v: Any) -> Any:
        u = self.scm.variables.get(var, {}).get("units") if self.display else None
        if u is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            return v
        return (float(v) - u["lo"]) / (u["hi"] - u["lo"]) * 100.0

    def _obs_display(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.display:
            return obs
        out = {}
        for k, v in obs.items():
            u = self.scm.variables.get(k, {}).get("units")
            out[k] = (u["lo"] + np.asarray(v, dtype=float) / 100.0 * (u["hi"] - u["lo"])) if u else v
        return out

    def answer_to_internal(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        if not self.display:
            return answer
        a = dict(answer)
        a["recommended_intervention"] = {k: self.act_to_internal(k, v)
                                         for k, v in (answer.get("recommended_intervention") or {}).items()}
        pol = answer.get("recommended_policy")
        if isinstance(pol, dict) and pol.get("treatment") in self.scm.actuators:
            p = dict(pol)
            for key in ("dose_if_ge", "dose_if_lt"):
                if key in p:
                    p[key] = self.act_to_internal(pol["treatment"], p[key])
            if "threshold" in p and pol.get("stratifier") in self.scm.variables:
                p["threshold"] = self.var_to_internal(pol["stratifier"], p["threshold"])
            a["recommended_policy"] = p
        return a

    def _write_rows(self, exp_id: int, obs: Dict[str, np.ndarray],
                    intervention: Dict[str, Any], mode: str) -> Optional[str]:
        if not self.data_dir:
            return None
        import pandas as pd
        df = pd.DataFrame({k: np.asarray(v) for k, v in obs.items()})
        # unit ids are unique to the experiment (every experiment draws a fresh batch of units): integers
        # experiment * 1000 + row, e.g. 3000-3399
        base = 10 ** max(3, len(str(len(df))))
        df.insert(0, "unit_id", [exp_id * base + i for i in range(len(df))])
        for aid, val in (intervention or {}).items():
            df[f"do_{aid}"] = val
        df = df.rename(columns=self.csv_columns)
        path = str(self.data_dir / f"experiment_{exp_id}.csv")
        df.to_csv(path, index=False)
        self.experiments[exp_id] = {"path": path, "columns": list(df.columns),
                                    "mode": mode, "intervention": dict(intervention or {})}
        return path

    def _control_level(self, aid: str, applied: Optional[Dict[str, Any]] = None) -> Any:
        if applied and aid in applied:
            return applied[aid]
        return self.act_to_display(aid, self.scm.actuators[aid].get("default"))

    def measure(self, requests: List[str], intervention: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._q += 1
        requests = [r for r in requests if isinstance(r, str)]
        resolutions = [self.resolver.resolve_measure(r) for r in requests]
        var_ids, echoes = [], []
        for req, res in zip(requests, resolutions):
            echoes.append({"request": req, **res.to_dict()})
            if res.kind == "control" and res.target_id in self.scm.actuators:
                echoes[-1]["control_level"] = self._control_level(res.target_id)
            if res.ok and res.target_id:
                var_ids.append(res.target_id)
        readings = {}
        csv_path = None
        if var_ids:
            iv = self._resolve_intervention_dict(intervention)
            vals = self.scm.sample(self.n_sample, intervention=iv, seed=self._seed0 + self._q, select=True)
            obs = self._obs_display(self.scm.measure(vals, var_ids, seed=self._seed0 + 5000 + self._q))
            readings = {k: {"mean": round(float(np.mean(v)), 3),
                            "sd": round(float(np.std(v)), 3),
                            "se": round(float(np.std(v) / np.sqrt(len(v))), 3)}
                        for k, v in obs.items()}
            names = list(obs)
            corr = {}
            pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
            # the summary lists correlations with the outcome only (all pairs when the outcome is not measured
            # and at most 8 signals are)
            if self.scm.outcome in names:
                k = names.index(self.scm.outcome)
                pairs = [(i, j) for i, j in pairs if k in (i, j)]
            elif len(names) > 8:
                pairs = []
            for i, j in pairs:
                a, b = obs[names[i]], obs[names[j]]
                c = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
                corr[f"{names[i]}~{names[j]}"] = round(c, 3)
            readings["_correlations"] = corr
            csv_path = self._write_rows(self._q, obs, iv, "measure")
        if not var_ids:
            self._q -= 1   # nothing ran, so no experiment number is used
        return {"mode": "measure", "experiment_id": self._q if var_ids else None, "n_units": self.n_sample,
                "resolutions": echoes, "readings": readings, "raw_csv": csv_path}

    def intervene(self, actions: List[Dict[str, Any]], measure_requests: List[str]) -> Dict[str, Any]:
        self._q += 1
        resolutions, iv = [], {}
        for a in actions:
            if not isinstance(a, dict):
                continue
            res = self.resolver.resolve_intervene(a.get("request", ""), a.get("value"))
            resolutions.append({"request": a.get("request", ""), **res.to_dict()})
            if not res.ok and not res.candidates:
                # setting a measured signal -> say so (signals are public), instead of "no match"
                sig = self.resolver._best_alias(a.get("request", ""), self.resolver.var_aliases)
                if sig and sig[0][1] >= 0.85:
                    resolutions[-1].update(kind="signal", target_id=sig[0][0])
            if res.ok and res.target_id:
                if res.target_id in iv:   # the same control twice in one experiment
                    resolutions[-1].update(ok=False, kind="duplicate", value=None)
                    continue
                iv[res.target_id] = res.value
        measure_requests = [r for r in measure_requests if isinstance(r, str)]
        m_res = [self.resolver.resolve_measure(r) for r in measure_requests]
        m_ids = [r.target_id for r in m_res if r.ok and r.target_id]
        m_echo = [{"request": rq, **rr.to_dict()} for rq, rr in zip(measure_requests, m_res)]
        for e in m_echo:
            if e.get("kind") == "control" and e.get("target_id") in self.scm.actuators:
                e["control_level"] = self._control_level(e["target_id"], iv)
        if self.scm.outcome not in m_ids:
            m_ids.append(self.scm.outcome)
        readings = {}
        csv_path = None
        iv_int = {k: self.act_to_internal(k, v) for k, v in iv.items()}
        # an intervention with no controls is a do-nothing baseline: a fresh representative batch at the
        # current settings (the prompt tells the agent to compare interventions with each other)
        baseline = not [a for a in actions if isinstance(a, dict)]
        if (iv or baseline) and m_ids:
            vals = self.scm.sample(self.n_sample, intervention=iv_int, seed=self._seed0 + self._q)
            obs = self._obs_display(self.scm.measure(vals, m_ids, seed=self._seed0 + 5000 + self._q))
            readings = {k: round(float(np.mean(v)), 3) for k, v in obs.items()}
            csv_path = self._write_rows(self._q, obs, iv, "intervene")
        ran = bool((iv or baseline) and m_ids)
        if not ran:
            self._q -= 1   # nothing ran, so no experiment number is used
        out = {"mode": "intervene", "experiment_id": self._q if ran else None, "n_units": self.n_sample,
                "applied_intervention": iv, "action_resolutions": resolutions,
                "measurement_resolutions": m_echo, "readings": readings, "raw_csv": csv_path}
        if self.display:
            out["applied_internal"] = iv_int
        return out

    def _resolve_intervention_dict(self, intervention: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not intervention:
            return {}
        return {k: v for k, v in intervention.items() if k in self.scm.actuators}

    _REJECT = "Not available on this system: no assay or control matches that request."

    def agent_view(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """What the agent is shown of an experiment result (the parser's internal fields are left out)."""
        lab = lambda aid: self.act_label.get(aid, aid)
        col = lambda aid: self.csv_columns.get(f"do_{aid}", lab(aid))

        def echo(e: Dict[str, Any]) -> Dict[str, Any]:
            out = {"request": e.get("request", "")}
            if e.get("kind") == "control" and e.get("target_id"):
                out.update(ok=False, interpretation=f"'{lab(e['target_id'])}' is a control, not a measured signal; "
                           f"it was at {e.get('control_level')} in this experiment. Change it with an intervene action.")
                return out
            if e.get("kind") == "signal":
                out.update(ok=False, interpretation=f"'{e['target_id']}' is a measured signal, not a control; "
                           "it cannot be set directly.")
                return out
            if e.get("kind") == "duplicate":
                out.update(ok=False, interpretation=f"'{lab(e['target_id'])}' was set more than once in this experiment; "
                           "only its first setting was applied.")
                return out
            if not e.get("ok"):
                if e.get("candidates"):
                    names = [lab(c) if c in self.scm.actuators else c for c in e["candidates"]]
                    out.update(ok=False, interpretation="ambiguous request; did you mean one of: " + ", ".join(names))
                else:
                    out.update(ok=False, interpretation=self._REJECT)
            elif e.get("kind") == "intervene":
                out.update(ok=True, interpretation=f"apply '{lab(e['target_id'])}' at {e.get('value')} "
                                                   f"(column {col(e['target_id'])})")
            else:
                out.update(ok=True, interpretation=f"measure '{e['target_id']}'")
            return out

        keys = ("mode", "experiment_id", "n_units", "readings") if result.get("raw_csv") else ("mode",)
        v = {k: result[k] for k in keys if k in result}
        for key in ("resolutions", "action_resolutions", "measurement_resolutions"):
            if key in result:
                v[key] = [echo(e) for e in result[key]]
        if "applied_intervention" in result:
            # keyed by the CSV column name
            v["applied_intervention"] = {col(a): val for a, val in result["applied_intervention"].items()}
        if result.get("raw_csv"):
            # name the DataFrame the code tool provides, list its columns, and say the units are a fresh batch
            eid = result["experiment_id"]
            v["data"] = (f"per-unit rows are in experiment_{eid}_df (pandas DataFrame in your code; "
                         f"file experiment_{eid}.csv). Fresh batch of {result.get('n_units')} units; "
                         f"unit ids are unique to this experiment.")
            cols = (self.experiments.get(eid) or {}).get("columns")
            if cols:
                v["columns"] = cols
        return v

    def grade(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        return _grade(self.world, self.answer_to_internal(answer), self.gold, self.battery)

    def public(self) -> Dict[str, Any]:
        return {
            "world_id": self.world["world_id"],
            "domain": self.world["domain"],
            "scenario": self.world["scenario"],
            "outcome_name": self.scm.outcome,
            "outcome_direction": "higher_is_better" if self.scm.higher_is_better else "lower_is_better",
        }
