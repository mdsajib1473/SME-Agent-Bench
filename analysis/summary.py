"""Plain language summary.md of the key numbers, for drafting the Results section."""

import math

from analysis.load import arch_label, language_label, model_label
from analysis.signals import SIGNALS


def pct(value):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value * 100:.0f}%"


def wh(value):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f} Wh"


def pval(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def odds(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "undefined (no discordant tasks)"
    return "infinite (all discordant tasks favour the first)" if math.isinf(value) else f"{value:.2f}"


def build_summary(data, main_df, tests_df, breakdown_df, signals_df, font_check, files, k):
    seeds = len(data.seeds)
    n_tasks = data.rows["task_id"].nunique()
    lines = [
        f"# Results summary: {data.tag}",
        "",
        f"{len(data.rows)} runs: {len(data.models)} model(s), {len(data.archs)} architecture(s), {n_tasks} tasks,"
        f" {seeds} seed(s) per task ({', '.join(str(s) for s in data.seeds)}).",
        "Success rates are per task means with 95% bootstrap confidence intervals over tasks (10,000 resamples, seed 0).",
        (f"pass^{k} is the share of tasks solved in all {k} seeds" if k > 1
         else "pass^1 equals the success rate, because each task has a single seed")
        + (f"; only {seeds} seed(s) are available, so pass^k uses k = {k} instead of 3." if seeds < 3 else "."),
        "Energy is net GPU board energy above idle; energy per success is the total net energy of a model and"
        " architecture divided by its number of successful runs.",
        "",
        "## Key numbers",
        "",
    ]
    for model in data.models:
        cells = main_df[main_df["model"] == model]
        if cells.empty:
            continue
        lines.append(f"### {model_label(model)}")
        lines.append("")
        for _, row in cells.iterrows():
            lines.append(
                f"* {arch_label(row['arch'])}: success {pct(row['success_rate'])}"
                f" (95% CI {pct(row['success_ci_low'])} to {pct(row['success_ci_high'])}), pass^{k}"
                f" {pct(row['pass_k'])}, policy violations on trap tasks {pct(row['trap_violation_rate'])}"
                f" ({int(row['violations_total'])} violation(s) in all), {row['mean_llm_calls']:.1f} LLM calls per run,"
                f" median wall time {row['wall_median_s']:.1f} s, {wh(row['mean_net_energy_wh'])} per run,"
                f" {wh(row['energy_per_success_wh'])} per successful run"
                f" (counter cross check {wh(row['energy_per_success_counter_wh'])})."
            )
        best = cells.loc[cells["success_rate"].idxmax()]
        cheapest = cells.loc[cells["mean_net_energy_wh"].idxmin()] if cells["mean_net_energy_wh"].notna().any() else None
        lines.append(f"* Highest success: {arch_label(best['arch'])} ({pct(best['success_rate'])}).")
        if cheapest is not None:
            lines.append(f"* Lowest energy per run: {arch_label(cheapest['arch'])} ({wh(cheapest['mean_net_energy_wh'])}).")
        extras = []
        for _, row in cells.iterrows():
            if row["arch"] == "supervisor" and row["mean_delegations"] == row["mean_delegations"]:
                extras.append(f"Supervisor made {row['mean_delegations']:.1f} delegations per run")
            if row["arch"] == "plan_execute" and row["mean_replans"] == row["mean_replans"]:
                extras.append(f"Plan and Execute made {row['mean_replans']:.2f} replans per run")
        if extras:
            lines.append(f"* {'; '.join(extras)}.")
        lines.append("")

    lines += ["## Statistical tests", ""]
    if tests_df.empty:
        lines += ["No pairs could be compared.", ""]
    for _, row in tests_df.iterrows():
        if row["family"] == "architectures":
            name = f"{model_label(row['group'])}, {arch_label(row['a'])} vs {arch_label(row['b'])}"
        else:
            name = f"{arch_label(row['group'])}, {model_label(row['a'])} vs {model_label(row['b'])}"
        verdict = "significant" if row["p_holm"] < 0.05 else "not significant"
        lines.append(
            f"* {name}: {row['a_only']} task(s) solved only by the first, {row['b_only']} only by the second"
            f" (odds ratio {odds(row['odds_ratio'])}), exact McNemar p = {pval(row['p_exact'])},"
            f" Holm adjusted p = {pval(row['p_holm'])}, {verdict} at 0.05."
        )
    lines.append("")

    lines += ["## Breakdowns", ""]
    language = breakdown_df[breakdown_df["dimension"] == "language"]
    for model in data.models:
        for arch in data.archs:
            mine = language[(language["model"] == model) & (language["arch"] == arch)]
            if mine.empty:
                continue
            parts = [f"{language_label(r['level'])} {pct(r['success_rate'])} (n = {r['n_runs']})" for _, r in mine.iterrows()]
            lines.append(f"* {model_label(model)} {arch_label(arch)}: {', '.join(parts)}.")
    traps = breakdown_df[breakdown_df["dimension"] == "is_trap"]
    for model in data.models:
        for arch in data.archs:
            mine = traps[(traps["model"] == model) & (traps["arch"] == arch)].set_index("level")
            if True in mine.index and False in mine.index:
                lines.append(
                    f"* {model_label(model)} {arch_label(arch)}: trap tasks {pct(mine.loc[True, 'success_rate'])},"
                    f" other tasks {pct(mine.loc[False, 'success_rate'])}."
                )
    lines += ["", "Category and difficulty breakdowns are in breakdowns.csv and fig_category.", ""]

    lines += ["## Failure signals", ""]
    if signals_df.empty:
        lines += ["No failed runs.", ""]
    else:
        lines.append(f"{len(signals_df)} failed runs. Share of failed runs showing each automatic signal:")
        lines.append("")
        for model in data.models:
            for arch in data.archs:
                mine = signals_df[(signals_df["model"] == model) & (signals_df["arch"] == arch)]
                if mine.empty:
                    continue
                counts = []
                for name in SIGNALS:
                    values = mine[name].replace("", None).dropna().astype(int)
                    if values.sum():
                        counts.append(f"{name} {values.sum()}/{len(values)}")
                lines.append(f"* {model_label(model)} {arch_label(arch)} ({len(mine)} failed): "
                             + (", ".join(counts) if counts else "no signal fired") + ".")
        lines.append("")

    lines += ["## Figures and fonts", ""]
    for name, subtypes in font_check.items():
        lines.append(f"* {name}: font types {', '.join(subtypes) or 'none'}"
                     + ("; contains Type 3 fonts." if "Type3" in subtypes else "; no Type 3 fonts."))
    lines.append("")

    lines += ["## Limits of this run", ""]
    lines.append(f"* Seeds: {seeds} per task" + (" (fewer than the planned 3; pass^k and the majority vote use the seeds"
                                                  " available, and a tie counts as failure)." if seeds < 3 else "."))
    lines.append(f"* Tasks: {n_tasks}. Runs: {len(data.rows)}.")
    expected = n_tasks * seeds * len(data.models) * len(data.archs)
    if expected != len(data.rows):
        lines.append(f"* The design has {expected} runs but {len(data.rows)} are recorded; some cells are incomplete.")
    if not data.notes:
        lines.append("* No fallbacks were needed; every field was recorded by the runner.")
    for note in data.notes:
        lines.append(f"* {note}")
    lines += ["", "## Files", ""]
    lines += [f"* {name}" for name in files]
    lines.append("")
    return "\n".join(lines)
