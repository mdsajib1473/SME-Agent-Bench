"""Main results table, breakdowns and McNemar tests as DataFrames."""

import itertools
import math

import numpy as np
import pandas as pd

from analysis import stats
from analysis.load import (
    CATEGORY_ORDER,
    DIFFICULTY_ORDER,
    LANGUAGE_ORDER,
    arch_label,
    model_label,
    ordered,
)

MAX_K = 3
BREAKDOWNS = (
    ("category", CATEGORY_ORDER),
    ("difficulty", DIFFICULTY_ORDER),
    ("language", LANGUAGE_ORDER),
    ("is_trap", (False, True)),
)


def pass_k_value(data):
    return min(len(data.seeds), MAX_K)


def per_task_successes(rows):
    """task_id -> successes ordered by seed."""
    return {
        task_id: [bool(s) for s in group.sort_values("seed")["success"]]
        for task_id, group in rows.groupby("task_id", sort=True)
    }


def rate(series):
    """Share of runs where a count is above zero or a flag is true; NaN when every value is missing."""
    values = series.dropna()
    if values.empty:
        return math.nan
    return float((values.astype(float) > 0).mean())


def nan_mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else math.nan


def energy_per_success(energy, successes):
    values = pd.to_numeric(energy, errors="coerce")
    if successes == 0 or values.isna().all():
        return math.nan
    return float(values.sum(skipna=True)) / successes


def cell_stats(rows, k, n_resamples):
    tasks = per_task_successes(rows)
    task_rates = [sum(s) / len(s) for s in tasks.values()]
    low, high = stats.bootstrap_ci(task_rates, n_resamples=n_resamples)
    pass_k, pass_k_tasks = stats.pass_hat_k(tasks.values(), k)
    traps = rows[rows["is_trap"] == True]  # noqa: E712, is_trap may hold None
    successes = int(rows["success"].sum())
    wall = rows["wall_time_s"].dropna()
    energy_by_task = rows.groupby("task_id")["net_energy_wh"].mean().dropna()
    e_low, e_high = stats.bootstrap_ci(energy_by_task, n_resamples=n_resamples)
    return {
        "n_tasks": len(tasks),
        "n_runs": len(rows),
        "seeds": len(rows["seed"].unique()),
        "successes": successes,
        "success_rate": float(np.mean(task_rates)) if task_rates else math.nan,
        "success_ci_low": low,
        "success_ci_high": high,
        "k": k,
        "pass_k": pass_k,
        "pass_k_tasks": pass_k_tasks,
        "trap_runs": len(traps),
        "trap_violation_rate": float(traps["policy_violation"].mean()) if len(traps) else math.nan,
        "violations_total": int(rows["policy_violation"].sum()),
        "mean_llm_calls": nan_mean(rows["llm_calls"]),
        "mean_prompt_tokens": nan_mean(rows["prompt_tokens"]),
        "mean_completion_tokens": nan_mean(rows["completion_tokens"]),
        "wall_median_s": float(wall.median()) if len(wall) else math.nan,
        "wall_p95_s": float(np.percentile(wall, 95)) if len(wall) else math.nan,
        "mean_net_energy_wh": nan_mean(rows["net_energy_wh"]),
        "net_energy_ci_low": e_low,
        "net_energy_ci_high": e_high,
        "energy_per_success_wh": energy_per_success(rows["net_energy_wh"], successes),
        "mean_net_energy_counter_wh": nan_mean(rows["net_energy_counter_wh"]),
        "energy_per_success_counter_wh": energy_per_success(rows["net_energy_counter_wh"], successes),
        "dropped_rate": rate(rows["dropped_tool_calls"]),
        "malformed_rate": rate(rows["malformed_tool_calls"]),
        "text_tool_rate": rate(rows["text_tool_calls"]),
        "empty_reply_rate": rate(rows["empty_reply"]),
        "budget_exceeded_rate": rate(rows["budget_exceeded"]),
        "wrong_script_rate": rate(rows["wrong_script"]),
        "dropped_calls_total": int(rows["dropped_tool_calls"].fillna(0).sum()),
        "malformed_calls_total": int(rows["malformed_tool_calls"].fillna(0).sum()),
        "text_calls_total": int(rows["text_tool_calls"].fillna(0).sum()),
    }


def main_results(data, n_resamples=stats.N_RESAMPLES):
    k = pass_k_value(data)
    records = []
    for model in data.models:
        for arch in data.archs:
            rows = data.rows[(data.rows["model"] == model) & (data.rows["arch"] == arch)]
            if rows.empty:
                continue
            record = {"model": model, "model_label": model_label(model), "arch": arch,
                      "arch_label": arch_label(arch)}
            record.update(cell_stats(rows, k, n_resamples))
            record["mean_delegations"] = nan_mean(rows["delegations"]) if arch == "supervisor" else math.nan
            record["mean_replans"] = nan_mean(rows["replans"]) if arch == "plan_execute" else math.nan
            records.append(record)
    return pd.DataFrame(records)


def breakdowns(data):
    records = []
    for dimension, preferred in BREAKDOWNS:
        levels = ordered([v for v in data.rows[dimension] if v is not None and v == v], preferred)
        for model in data.models:
            for arch in data.archs:
                cell = data.rows[(data.rows["model"] == model) & (data.rows["arch"] == arch)]
                if cell.empty:
                    continue
                for level in levels:
                    rows = cell[cell[dimension] == level]
                    if rows.empty:
                        continue
                    records.append({
                        "dimension": dimension,
                        "level": level,
                        "model": model,
                        "arch": arch,
                        "n_tasks": rows["task_id"].nunique(),
                        "n_runs": len(rows),
                        "successes": int(rows["success"].sum()),
                        "success_rate": float(rows["success"].mean()),
                    })
    return pd.DataFrame(records)


def majority_by_task(rows):
    return {task_id: stats.majority_success(s) for task_id, s in per_task_successes(rows).items()}


def compare(family, group, rows_a, rows_b, label_a, label_b):
    votes_a, votes_b = majority_by_task(rows_a), majority_by_task(rows_b)
    shared = sorted(set(votes_a) & set(votes_b))
    a = [votes_a[t] for t in shared]
    b = [votes_b[t] for t in shared]
    both, a_only, b_only, neither = stats.discordant_counts(a, b)
    return {
        "family": family,
        "group": group,
        "a": label_a,
        "b": label_b,
        "n_tasks": len(shared),
        "both_success": both,
        "a_only": a_only,
        "b_only": b_only,
        "both_fail": neither,
        "success_a": sum(a) / len(a) if a else math.nan,
        "success_b": sum(b) / len(b) if b else math.nan,
        "odds_ratio": stats.discordant_odds_ratio(a_only, b_only),
        "p_exact": stats.mcnemar_exact(a_only, b_only),
    }


def mcnemar_tests(data):
    """Architecture pairs within each model, then model pairs within each architecture.

    Holm correction runs over the architecture pairs of one model, and over the
    model comparisons of all architectures together.
    """
    rows = data.rows
    records = []
    for model in data.models:
        family = []
        for arch_a, arch_b in itertools.combinations(data.archs, 2):
            a = rows[(rows["model"] == model) & (rows["arch"] == arch_a)]
            b = rows[(rows["model"] == model) & (rows["arch"] == arch_b)]
            if not a.empty and not b.empty:
                family.append(compare("architectures", model, a, b, arch_a, arch_b))
        for record, p in zip(family, stats.holm([r["p_exact"] for r in family])):
            record["p_holm"] = p
        records.extend(family)
    family = []
    for arch in data.archs:
        for model_a, model_b in itertools.combinations(data.models, 2):
            a = rows[(rows["model"] == model_a) & (rows["arch"] == arch)]
            b = rows[(rows["model"] == model_b) & (rows["arch"] == arch)]
            if not a.empty and not b.empty:
                family.append(compare("models", arch, a, b, model_a, model_b))
    for record, p in zip(family, stats.holm([r["p_exact"] for r in family])):
        record["p_holm"] = p
    records.extend(family)
    columns = ["family", "group", "a", "b", "n_tasks", "both_success", "a_only", "b_only", "both_fail",
               "success_a", "success_b", "odds_ratio", "p_exact", "p_holm"]
    return pd.DataFrame(records, columns=columns)
