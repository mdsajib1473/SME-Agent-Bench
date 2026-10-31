import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis import latex, report, stats
from analysis.load import (
    CATEGORY_ORDER,
    LANGUAGE_ORDER,
    RESULTS_DIR,
    ROOT,
    arch_label,
    category_label,
    language_label,
    load_run,
    model_label,
    model_size_b,
)

ASSETS_DIR = ROOT / "paper_assets"
SINGLE_IN = 3.5
DOUBLE_IN = 7.16
STYLE = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.default": "regular",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "hatch.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
}
ARCH_VISUAL = {
    "react": {"color": "#0072B2", "hatch": "///", "marker": "o", "dash": "-"},
    "plan_execute": {"color": "#E69F00", "hatch": "...", "marker": "s", "dash": "--"},
    "supervisor": {"color": "#F0E442", "hatch": "xxx", "marker": "^", "dash": ":"},
}
ERROR_KW = {"ecolor": "black", "elinewidth": 0.7, "capsize": 1.5, "capthick": 0.7}


def save(fig, out_dir, name):
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{suffix}")
    plt.close(fig)


def metric_ci(rows, column):
    by_task = rows.groupby("task_id")[column].mean().astype(float)
    low, high = stats.bootstrap_ci(by_task)
    return float(by_task.mean()), low, high


def cell_rows(rows, model, arch):
    return rows[(rows["model"] == model) & (rows["arch"] == arch)]


def grouped_bars(ax, groups, archs, cells, scale=1.0):
    width = 0.8 / len(archs)
    x = np.arange(len(groups))
    for i, arch in enumerate(archs):
        values = np.array([cells[(group, arch)] for group in groups], dtype=float) * scale
        visual = ARCH_VISUAL[arch]
        ax.bar(
            x - 0.4 + width * (i + 0.5), values[:, 0], width, color=visual["color"], hatch=visual["hatch"],
            edgecolor="black", linewidth=0.5,
            yerr=[np.maximum(values[:, 0] - values[:, 1], 0), np.maximum(values[:, 2] - values[:, 0], 0)],
            error_kw=ERROR_KW,
        )
    ax.set_xticks(x)


def arch_patches(archs):
    return [Patch(facecolor=ARCH_VISUAL[a]["color"], hatch=ARCH_VISUAL[a]["hatch"], edgecolor="black", linewidth=0.5,
                  label=arch_label(a)) for a in archs]


def top_legend(fig, handles, ncol, columnspacing=1.2):
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol, frameon=False, handlelength=1.8,
               columnspacing=columnspacing, handletextpad=0.5)


def percent_axis(ax, top=105):
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, 101, 20))


def fig_category(data, out_dir):
    rows = data.rows
    categories = [c for c in CATEGORY_ORDER if c in set(rows["category"])]
    with matplotlib.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_IN, 2.6), sharey=True, layout="constrained")
        for ax, model in zip(axes, data.models):
            cells = {(c, a): metric_ci(cell_rows(rows, model, a).query("category == @c"), "success")
                     for c in categories for a in data.archs}
            grouped_bars(ax, categories, data.archs, cells, 100)
            ax.set_xticklabels([category_label(c).replace(" ", "\n") for c in categories])
            ax.set_xlabel(f"Task category, {model_label(model)}")
            percent_axis(ax)
        axes[0].set_ylabel("Success rate (%)")
        top_legend(fig, arch_patches(data.archs), len(data.archs))
        save(fig, out_dir, "fig_success_by_category")


def fig_language(data, out_dir):
    rows = data.rows
    languages = [l for l in LANGUAGE_ORDER if l in set(rows["language"])]
    with matplotlib.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(SINGLE_IN, 2.5), sharey=True, layout="constrained")
        for ax, model in zip(axes, data.models):
            cells = {(l, a): metric_ci(cell_rows(rows, model, a).query("language == @l"), "success")
                     for l in languages for a in data.archs}
            grouped_bars(ax, languages, data.archs, cells, 100)
            ax.set_xticklabels([language_label(l) for l in languages])
            ax.set_xlabel(model_label(model))
            percent_axis(ax)
        axes[0].set_ylabel("Success rate (%)")
        top_legend(fig, arch_patches(data.archs), 2)
        save(fig, out_dir, "fig_success_by_language")


def fig_success_energy(main_df, data, out_dir):
    with matplotlib.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(SINGLE_IN, 2.9), layout="constrained")
        for _, row in main_df.iterrows():
            visual = ARCH_VISUAL[row["arch"]]
            filled = data.models.index(row["model"]) == 0
            x, y = row["mean_net_energy_wh"], row["success_rate"] * 100
            ax.errorbar(
                x, y,
                xerr=[[x - row["net_energy_ci_low"]], [row["net_energy_ci_high"] - x]],
                yerr=[[y - row["success_ci_low"] * 100], [row["success_ci_high"] * 100 - y]],
                fmt=visual["marker"], markersize=6, markerfacecolor=visual["color"] if filled else "white",
                markeredgecolor="black", markeredgewidth=0.8, **ERROR_KW,
            )
        ax.set_xlabel("Mean net GPU energy per task (Wh)")
        ax.set_ylabel("Success rate (%)")
        ax.set_xlim(0, 1.3)
        ax.set_ylim(0, 65)
        arch_handles = [Line2D([], [], linestyle="none", marker=ARCH_VISUAL[a]["marker"], markersize=6,
                               markerfacecolor=ARCH_VISUAL[a]["color"], markeredgecolor="black", markeredgewidth=0.8,
                               label=arch_label(a)) for a in data.archs]
        model_handles = [Patch(facecolor="0.3" if i == 0 else "white", edgecolor="black", linewidth=0.6,
                               label=f"{model_size_b(m):g}B ({'colored' if i == 0 else 'white'})")
                         for i, m in enumerate(data.models)]
        blank = Patch(facecolor="none", edgecolor="none", label=" ")
        handles = [arch_handles[0], model_handles[0], arch_handles[1], model_handles[1], arch_handles[2], blank]
        top_legend(fig, handles, 3, columnspacing=0.6)
        save(fig, out_dir, "fig_success_vs_energy")


def fig_tokens_latency(data, out_dir):
    rows = data.rows
    labels = [model_label(m) for m in data.models]
    with matplotlib.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_IN, 2.5), layout="constrained")
        prompt = {(m, a): metric_ci(cell_rows(rows, m, a), "prompt_tokens") for m in data.models for a in data.archs}
        completion = {(m, a): metric_ci(cell_rows(rows, m, a), "completion_tokens")
                      for m in data.models for a in data.archs}
        wall = {}
        for m in data.models:
            for a in data.archs:
                values = cell_rows(rows, m, a)["wall_time_s"]
                median = float(values.median())
                wall[(m, a)] = (median, median, float(np.percentile(values, 95)))
        grouped_bars(axes[0], data.models, data.archs, prompt, 0.001)
        grouped_bars(axes[1], data.models, data.archs, completion)
        grouped_bars(axes[2], data.models, data.archs, wall)
        axes[0].set_ylabel("Prompt tokens per run (thousands)")
        axes[1].set_ylabel("Completion tokens per run")
        axes[2].set_ylabel("Agent run time (s)")
        for ax in axes:
            ax.set_xticklabels(labels)
            ax.set_xlabel("Model")
            ax.set_ylim(bottom=0)
        top_legend(fig, arch_patches(data.archs), len(data.archs))
        save(fig, out_dir, "fig_tokens_latency")


def fig_violations(data, out_dir):
    rows = data.rows.assign(violation=data.rows["policy_violation"].astype(float))
    traps = rows[rows["is_trap"] == True]
    labels = [model_label(m) for m in data.models]
    with matplotlib.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(SINGLE_IN, 2.5), sharey=True, layout="constrained")
        for ax, subset, name in ((axes[0], rows, "All 80 tasks"), (axes[1], traps, "16 trap tasks")):
            cells = {(m, a): metric_ci(cell_rows(subset, m, a), "violation")
                     for m in data.models for a in data.archs}
            grouped_bars(ax, data.models, data.archs, cells, 100)
            ax.set_xticklabels(labels)
            ax.set_xlabel(name)
            ax.set_ylim(0, 85)
        axes[0].set_ylabel("Policy violation rate (% of runs)")
        top_legend(fig, arch_patches(data.archs), 2)
        save(fig, out_dir, "fig_policy_violations")


def pass_k_values(rows):
    tasks = report.per_task_successes(rows)
    return [stats.pass_hat_k(tasks.values(), k)[0] * 100 for k in (1, 2, 3)]


def fig_pass_k(data, out_dir):
    rows = data.rows
    with matplotlib.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(SINGLE_IN, 2.5), sharey=True, layout="constrained")
        for ax, model in zip(axes, data.models):
            for i, arch in enumerate(data.archs):
                visual = ARCH_VISUAL[arch]
                x = [k + (i - 1) * 0.1 for k in (1, 2, 3)]
                ax.plot(x, pass_k_values(cell_rows(rows, model, arch)), color="black", linewidth=0.9,
                        linestyle=visual["dash"], marker=visual["marker"], markersize=5,
                        markerfacecolor=visual["color"], markeredgecolor="black", markeredgewidth=0.8)
            ax.set_xticks((1, 2, 3))
            ax.set_xlim(0.7, 3.3)
            ax.set_ylim(0, 60)
            ax.set_xlabel(f"k, {model_label(model)}")
        axes[0].set_ylabel("$pass^{k}$ (%)")
        handles = [Line2D([], [], color="black", linewidth=0.9, linestyle=ARCH_VISUAL[a]["dash"],
                          marker=ARCH_VISUAL[a]["marker"], markersize=6, markerfacecolor=ARCH_VISUAL[a]["color"],
                          markeredgecolor="black", markeredgewidth=0.8, label=arch_label(a)) for a in data.archs]
        top_legend(fig, handles, 2)
        save(fig, out_dir, "fig_pass_k")


def copy_tables(analysis_dir, out_dir):
    for source, target in (("table_main_results.tex", "main_results.tex"), ("table_mcnemar.tex", "mcnemar.tex"),
                           ("main_results.csv", "main_results.csv"), ("mcnemar.csv", "mcnemar.csv"),
                           ("breakdowns.csv", "breakdowns.csv")):
        shutil.copyfile(analysis_dir / source, out_dir / target)


def pct(value):
    return f"{value * 100:.1f}%"


def key_numbers(data, main_df, tests_df, failures_df):
    tag = data.tag
    main_csv = f"results/{tag}/analysis/main_results.csv"
    mcnemar_csv = f"results/{tag}/analysis/mcnemar.csv"
    runs_file = f"results/{tag}/runs.jsonl"
    meta_file = f"results/{tag}/meta.json"
    failures_csv = f"results/{tag}/analysis/failures_auto.csv"
    rows = data.rows
    cells = [(r, f"{r['model_label']}, {r['arch_label']}") for _, r in main_df.iterrows()]
    lines = [
        "# Key numbers",
        "",
        f"Generated by analysis/paper_assets.py from results/{tag}. Percentages are per task means unless stated.",
        "Confidence intervals are 95% bootstrap intervals over tasks (10,000 resamples, seed 0).",
        "",
        "## Run",
        "",
        f"* Runs: {len(rows)}, from {rows['task_id'].nunique()} tasks, {len(data.models)} models,"
        f" {len(data.archs)} architectures and {len(data.seeds)} seeds. Source: {runs_file}.",
    ]
    for name in ("policy_sha256", "prompt_sha256", "harness_sha256", "tasks_sha256"):
        lines.append(f"* {name}: {data.meta[name]}. Source: {meta_file}.")
    lines += ["", "## Success per model and architecture", ""]
    for r, label in cells:
        lines.append(
            f"* {label}: {pct(r['success_rate'])} (95% CI {pct(r['success_ci_low'])} to {pct(r['success_ci_high'])}),"
            f" {r['successes']} of {r['n_runs']} runs. Source: {main_csv}."
        )
    lines += ["", "## Success pooled", ""]
    for name, column, values, label in (
        ("architecture", "arch", data.archs, arch_label),
        ("model", "model", data.models, model_label),
    ):
        for value in values:
            mean, low, high = metric_ci(rows[rows[column] == value], "success")
            lines.append(f"* {label(value)} ({name}, pooled): {pct(mean)} (95% CI {pct(low)} to {pct(high)})."
                         f" Source: {runs_file}.")
    lines += [
        "", "## pass^k", "",
        "pass^k averages comb(c, k) / comb(n, k) over tasks, where c is the number of successes among the n seeds of a"
        " task. pass^1 equals the success rate and pass^3 is the share of tasks solved in all 3 seeds.",
        "",
    ]
    for r, label in cells:
        values = pass_k_values(cell_rows(rows, r["model"], r["arch"]))
        lines.append(f"* {label}: pass^1 {values[0]:.1f}%, pass^2 {values[1]:.1f}%, pass^3 {values[2]:.1f}%."
                     f" Source: {runs_file}; pass^3 also in {main_csv}.")
    lines += ["", "## McNemar tests after Holm correction", ""]
    for _, t in tests_df.iterrows():
        if t["family"] == "architectures":
            name = f"{model_label(t['group'])}, {arch_label(t['a'])} vs {arch_label(t['b'])}"
        else:
            name = f"{arch_label(t['group'])}, {model_label(t['a'])} vs {model_label(t['b'])}"
        lines.append(
            f"* {name}: b = {t['a_only']}, c = {t['b_only']}, OR = {t['odds_ratio']:.2f}, p = {t['p_exact']:.4f},"
            f" Holm p = {t['p_holm']:.4f}. Source: {mcnemar_csv}."
        )
    lines += ["", "## Tokens per run", ""]
    for r, label in cells:
        lines.append(f"* {label}: {r['mean_prompt_tokens']:.0f} prompt, {r['mean_completion_tokens']:.0f} completion,"
                     f" {r['mean_llm_calls']:.2f} LLM calls. Source: {main_csv}.")
    lines += ["", "## Latency per run", ""]
    for r, label in cells:
        llm = cell_rows(rows, r["model"], r["arch"])["llm_latency_s"].mean()
        lines.append(f"* {label}: agent run time median {r['wall_median_s']:.1f} s, p95 {r['wall_p95_s']:.1f} s"
                     f" (source: {main_csv}); mean summed LLM latency {llm:.1f} s (source: {runs_file}).")
    lines += ["", "## Net GPU energy", ""]
    for r, label in cells:
        lines.append(
            f"* {label}: {r['mean_net_energy_wh']:.3f} Wh per run (95% CI {r['net_energy_ci_low']:.3f} to"
            f" {r['net_energy_ci_high']:.3f}), {r['energy_per_success_wh']:.3f} Wh per successful run, counter cross"
            f" check {r['mean_net_energy_counter_wh']:.3f} and {r['energy_per_success_counter_wh']:.3f} Wh."
            f" Source: {main_csv}."
        )
    lines += ["", "## Policy violations", ""]
    for r, label in cells:
        all_rate = r["violations_total"] / r["n_runs"]
        lines.append(f"* {label}: {pct(r['trap_violation_rate'])} of {r['trap_runs']} trap runs,"
                     f" {r['violations_total']} violations in all, {pct(all_rate)} of {r['n_runs']} runs."
                     f" Source: {main_csv}.")
    lines += ["", "## Dropped, malformed and text tool calls", ""]
    for r, label in cells:
        lines.append(f"* {label}: {r['dropped_calls_total']} dropped ({pct(r['dropped_rate'])} of runs),"
                     f" {r['malformed_calls_total']} malformed ({pct(r['malformed_rate'])} of runs),"
                     f" {r['text_calls_total']} text ({pct(r['text_tool_rate'])} of runs). Source: {main_csv}.")
    lines += ["", "## Delegations and replans", ""]
    for r, label in cells:
        if r["arch"] == "supervisor":
            lines.append(f"* {label}: {r['mean_delegations']:.2f} delegations per run. Source: {main_csv}.")
        elif r["arch"] == "plan_execute":
            lines.append(f"* {label}: {r['mean_replans']:.2f} replans per run. Source: {main_csv}.")
    lines += ["", "## Reply quality and run errors", ""]
    for r, label in cells:
        lines.append(f"* {label}: wrong script {pct(r['wrong_script_rate'])} of runs, empty reply"
                     f" {pct(r['empty_reply_rate'])} of runs, budget exceeded {pct(r['budget_exceeded_rate'])} of runs."
                     f" Source: {main_csv}.")
    errors = rows[rows["stop_reason"] == "llm_error"]
    lines.append(f"* Failed runs: {len(failures_df)} of {len(rows)}. Source: {failures_csv}.")
    lines.append(f"* Runs ended by an endpoint error: {len(errors)}, all scored as failed. Source: {runs_file}.")
    for _, r in errors.iterrows():
        lines.append(f"* Endpoint error run: {model_label(r['model'])}, {arch_label(r['arch'])}, {r['task_id']},"
                     f" seed {r['seed']}. Source: {runs_file}.")
    lines.append("")
    return "\n".join(lines)


def blind_sheets(analysis_dir, guide_source, out_dir):
    sheet = pd.read_csv(analysis_dir / "failures_to_label.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    ids = [f"L{i:03d}" for i in range(1, len(sheet) + 1)]
    key = pd.DataFrame({"blind_id": ids, "source_run_id": sheet["run_id"], "arch": sheet["arch"],
                        "model": sheet["model"]})
    key.to_csv(out_dir / "coordinator_key_do_not_share.csv", index=False, encoding="utf-8-sig")
    blind = sheet.drop(columns=["arch", "model"]).assign(run_id=ids)
    blind = blind.assign(label_1="", label_2="", notes="")
    for reviewer in (1, 2):
        drop = "label_2" if reviewer == 1 else "label_1"
        blind.drop(columns=[drop]).to_csv(out_dir / f"failures_to_label_reviewer{reviewer}.csv", index=False,
                                          encoding="utf-8-sig")
    shutil.copyfile(guide_source, out_dir / "labeling_guide.md")
    return len(blind)


def build(tag, results_dir=RESULTS_DIR):
    data = load_run(Path(results_dir) / tag, tag)
    analysis_dir = data.run_dir / "analysis"
    folders = {name: ASSETS_DIR / name for name in ("figures", "tables", "numbers", "labeling")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    main_df = pd.read_csv(analysis_dir / "main_results.csv")
    tests_df = pd.read_csv(analysis_dir / "mcnemar.csv")
    failures_df = pd.read_csv(analysis_dir / "failures_auto.csv", encoding="utf-8-sig")

    fig_category(data, folders["figures"])
    fig_language(data, folders["figures"])
    fig_success_energy(main_df, data, folders["figures"])
    fig_tokens_latency(data, folders["figures"])
    fig_violations(data, folders["figures"])
    fig_pass_k(data, folders["figures"])
    copy_tables(analysis_dir, folders["tables"])
    (folders["tables"] / "main_results_compact.tex").write_text(
        latex.compact_table(main_df, report.pass_k_value(data), len(data.seeds)), encoding="utf-8")
    (folders["numbers"] / "key_numbers.md").write_text(
        key_numbers(data, main_df, tests_df, failures_df), encoding="utf-8")
    if (folders["labeling"] / "failures_to_label_reviewer1.csv").exists():
        return 0
    return blind_sheets(analysis_dir, analysis_dir / "labeling_guide.md", folders["labeling"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="main")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    labeled = build(args.tag, args.results_dir)
    print(f"paper assets written to {ASSETS_DIR}; "
          + (f"{labeled} failed runs in each labeling sheet" if labeled else "existing labeling sheets kept"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
