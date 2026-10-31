import argparse
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from analysis.load import (
    CATEGORY_ORDER,
    DIFFICULTY_ORDER,
    LANGUAGE_ORDER,
    RESULTS_DIR,
    arch_label,
    category_label,
    language_label,
    load_run,
    model_label,
)
from analysis.paper_assets import ASSETS_DIR, cell_rows, pass_k_values

FONT = "Arial"
BLACK = RGBColor(0, 0, 0)


def set_font(style, size, bold=False):
    style.font.name = FONT
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = BLACK


def new_document():
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.left_margin = section.right_margin = Inches(0.5)
    section.top_margin = section.bottom_margin = Inches(0.7)
    set_font(doc.styles["Normal"], 10)
    set_font(doc.styles["Heading 1"], 13, bold=True)
    set_font(doc.styles["Caption"], 9, bold=True)
    props = doc.core_properties
    props.author = props.last_modified_by = props.comments = props.title = props.subject = props.keywords = ""
    return doc


def text(doc, content):
    paragraph = doc.add_paragraph(content)
    paragraph.paragraph_format.space_after = Pt(8)


def figure(doc, name, width, caption, body):
    picture = doc.add_paragraph()
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.paragraph_format.keep_with_next = True
    picture.add_run().add_picture(str(ASSETS_DIR / "figures" / f"{name}.png"), width=Inches(width))
    label = doc.add_paragraph(caption, style="Caption")
    label.paragraph_format.keep_with_next = True
    text(doc, body)


def table(doc, caption, body, header, rows, widths=None):
    label = doc.add_paragraph(caption, style="Caption")
    label.paragraph_format.keep_with_next = True
    grid = doc.add_table(rows=1 + len(rows), cols=len(header))
    grid.style = "Table Grid"
    grid.alignment = WD_TABLE_ALIGNMENT.CENTER
    if widths:
        grid.autofit = False
        for column, width in zip(grid.columns, widths):
            column.width = Inches(width)
    for r, cells in enumerate([header] + rows):
        for c, value in enumerate(cells):
            cell = grid.cell(r, c)
            if widths:
                cell.width = Inches(widths[c])
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.keep_with_next = r < len(rows)
            run = paragraph.add_run(str(value))
            run.font.size = Pt(8)
            run.font.bold = r == 0
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(0)
    text(doc, body)


def cell(main_df, model, arch):
    return main_df[(main_df["model"] == model) & (main_df["arch"] == arch)].iloc[0]


def pct(value, digits=1):
    return f"{value * 100:.{digits}f}%"


def p_text(value):
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def odds_text(value):
    return "n/a" if value != value else ("inf" if value == float("inf") else f"{value:.2f}")


def accuracy_rows(main_df):
    return [[r["model_label"], r["arch_label"], pct(r["success_rate"]),
             f"{pct(r['success_ci_low'])} to {pct(r['success_ci_high'])}", pct(r["pass_k"]),
             pct(r["trap_violation_rate"]), int(r["violations_total"])] for _, r in main_df.iterrows()]


def cost_rows(main_df):
    return [[r["model_label"], r["arch_label"], f"{r['mean_llm_calls']:.1f}", f"{r['mean_prompt_tokens'] / 1000:.1f}",
             f"{r['mean_completion_tokens']:.0f}", f"{r['wall_median_s']:.1f}", f"{r['wall_p95_s']:.1f}",
             f"{r['mean_net_energy_wh']:.3f}", f"{r['energy_per_success_wh']:.3f}",
             f"{r['energy_per_success_counter_wh']:.3f}"] for _, r in main_df.iterrows()]


def mcnemar_rows(tests_df):
    rows = []
    for _, t in tests_df.iterrows():
        if t["family"] == "architectures":
            name = f"{model_label(t['group'])}: {arch_label(t['a'])} vs {arch_label(t['b'])}"
        else:
            name = f"{arch_label(t['group'])}: {model_label(t['a'])} vs {model_label(t['b'])}"
        rows.append([name, t["n_tasks"], t["a_only"], t["b_only"], odds_text(t["odds_ratio"]), p_text(t["p_exact"]),
                     p_text(t["p_holm"])])
    return rows


def breakdown_rows(breakdown_df, models, archs):
    groups = (
        ("category", CATEGORY_ORDER, category_label),
        ("difficulty", DIFFICULTY_ORDER, lambda v: str(v).capitalize()),
        ("language", LANGUAGE_ORDER, language_label),
        ("is_trap", ("False", "True"), lambda v: "Trap" if v == "True" else "Not trap"),
    )
    rows = []
    for dimension, levels, label in groups:
        part = breakdown_df[breakdown_df["dimension"] == dimension]
        for level in levels:
            values = [label(level)]
            for model in models:
                for arch in archs:
                    match = part[(part["level"].astype(str) == str(level)) & (part["model"] == model)
                                 & (part["arch"] == arch)]
                    values.append(pct(match["success_rate"].iloc[0]))
            rows.append(values)
    return rows


def error_effect(data):
    rows = data.rows
    errors = rows[rows["stop_reason"] == "llm_error"]
    largest = 0.0
    for _, r in errors.iterrows():
        group = rows[(rows["model"] == r["model"]) & (rows["arch"] == r["arch"])]
        largest = max(largest, group.drop(index=r.name)["success"].mean() - group["success"].mean())
    return len(errors), largest * 100


def build(tag, results_dir=RESULTS_DIR):
    data = load_run(Path(results_dir) / tag, tag)
    analysis_dir = data.run_dir / "analysis"
    main_df = pd.read_csv(analysis_dir / "main_results.csv")
    tests_df = pd.read_csv(analysis_dir / "mcnemar.csv")
    breakdown_df = pd.read_csv(analysis_dir / "breakdowns.csv")
    small, large = data.models
    r7, p7, s7 = (cell(main_df, small, a) for a in data.archs)
    r14, p14, s14 = (cell(main_df, large, a) for a in data.archs)
    significant = tests_df[tests_df["p_holm"] < 0.05]
    arch_test = tests_df[(tests_df["family"] == "architectures") & (tests_df["group"] == large)
                         & (tests_df["a"] == "react") & (tests_df["b"] == "plan_execute")].iloc[0]
    model_test = tests_df[(tests_df["family"] == "models") & (tests_df["group"] == "plan_execute")].iloc[0]
    error_count, error_shift = error_effect(data)
    model_ratios = [b["mean_net_energy_wh"] / a["mean_net_energy_wh"] for a, b in ((r7, r14), (p7, p14), (s7, s14))]
    plan_ratios = [b["mean_net_energy_wh"] / a["mean_net_energy_wh"] for a, b in ((r7, p7), (r14, p14))]
    plan_pass_1 = pass_k_values(cell_rows(data.rows, large, "plan_execute"))[0]
    doc = new_document()

    doc.add_heading("Run summary", level=1)
    text(doc, f"This report covers {len(data.rows)} runs from {data.rows['task_id'].nunique()} tasks, two models"
              f" ({model_label(small)} and {model_label(large)}), three architectures and three seeds per task. The"
              " policy, prompt, harness and task hashes recorded for the run match docs/method_notes.md.")
    text(doc, f"Server errors ended {error_count} runs on {model_label(large)}: the response was aborted with the"
              " message \"prediction aborted, token repeat limit reached\". They are scored as failed because the reply"
              f" is empty. Leaving them out would raise a cell success rate by at most {error_shift:.1f} percentage"
              " points, and no McNemar majority vote would change, because the other two seeds of each task also failed.")

    doc.add_heading("Main results", level=1)
    table(doc, "Table 1. Success and policy results per model and architecture.",
          f"{model_label(large)} with Plan and Execute has the highest success rate at {pct(p14['success_rate'])}, and"
          f" {model_label(small)} with ReAct has the lowest at {pct(r7['success_rate'])}. Plan and Execute leads within"
          " each model, but the confidence intervals overlap. pass^3 is the share of tasks solved in all three seeds"
          f" and ranges from {pct(r7['pass_k'])} to {pct(p14['pass_k'])}.",
          ["Model", "Architecture", "Success", "95% CI", "pass^3", "Trap violations", "All violations"],
          accuracy_rows(main_df))
    table(doc, "Table 2. Cost per run for each model and architecture.",
          f"Plan and Execute costs the most on both models: {p14['mean_llm_calls']:.1f} LLM calls and"
          f" {p14['mean_net_energy_wh']:.3f} Wh per run on {model_label(large)}, against"
          f" {r14['mean_llm_calls']:.1f} calls and {r14['mean_net_energy_wh']:.3f} Wh for ReAct. Energy per successful"
          f" run is lowest for {model_label(small)} ReAct at {r7['energy_per_success_wh']:.3f} Wh and highest for"
          f" {model_label(large)} Plan and Execute at {p14['energy_per_success_wh']:.3f} Wh. The counter cross check"
          " stays close to the sampled figure in every row.",
          ["Model", "Architecture", "LLM calls", "Prompt tokens (k)", "Completion tokens", "Run time p50 (s)",
           "Run time p95 (s)", "Wh per run", "Wh per success", "Wh per success (counter)"],
          cost_rows(main_df))
    figure(doc, "fig_success_vs_energy", 3.5,
           "Figure 1. Success rate against mean net GPU energy per task. Bars show 95% bootstrap intervals over tasks.",
           f"Success and energy rise together across the six systems, from {pct(r7['success_rate'])} at"
           f" {r7['mean_net_energy_wh']:.3f} Wh to {pct(p14['success_rate'])} at {p14['mean_net_energy_wh']:.3f} Wh."
           f" The larger model uses {min(model_ratios):.1f} to {max(model_ratios):.1f} times the energy per run of the"
           f" smaller one on the same architecture, while Plan and Execute uses about"
           f" {max(plan_ratios):.1f} times the energy of ReAct on the same model."
           f" {model_label(small)} Plan and Execute and {model_label(large)} ReAct reach similar success"
           f" ({pct(p7['success_rate'])} and {pct(r14['success_rate'])}) at {p7['mean_net_energy_wh']:.3f} and"
           f" {r14['mean_net_energy_wh']:.3f} Wh.")

    doc.add_heading("Statistical tests", level=1)
    table(doc, "Table 3. Exact McNemar tests on the per task majority vote over three seeds.",
          f"Only {len(significant)} of {len(tests_df)} comparisons stay significant at 0.05 after Holm correction."
          f" On {model_label(large)}, Plan and Execute beats ReAct ({arch_test['b_only']} tasks against"
          f" {arch_test['a_only']}, Holm p = {p_text(arch_test['p_holm'])}), and Plan and Execute improves from"
          f" {model_label(small)} to {model_label(large)} ({model_test['b_only']} tasks against"
          f" {model_test['a_only']}, Holm p = {p_text(model_test['p_holm'])}). No architecture difference on"
          f" {model_label(small)} is significant. b counts tasks solved only by the first system and c tasks solved"
          " only by the second.",
          ["Comparison", "n", "b", "c", "OR", "p", "Holm p"], mcnemar_rows(tests_df),
          widths=[3.1, 0.55, 0.55, 0.55, 0.7, 0.75, 0.75])

    doc.add_heading("Success by task type", level=1)
    figure(doc, "fig_success_by_category", 7.16,
           "Figure 2. Success rate by task category. Bars show 95% bootstrap intervals over the 16 tasks of a category.",
           "Order change and inquiry are the two most solved categories on all six systems. Refund, complaint routing"
           " and quotation stay below them, and which of the three is lowest varies by system. Each bar rests on only"
           " 16 tasks, so the intervals are wide and small gaps should not be read as differences.")
    figure(doc, "fig_success_by_language", 3.5,
           "Figure 3. Success rate by language. Bars show 95% bootstrap intervals over 60 English and 20 Banglish tasks.",
           "The English and Banglish intervals overlap on every system, and the Banglish intervals are wide with 20"
           f" tasks. The largest point difference is {model_label(small)} Supervisor, and on {model_label(large)} that"
           " architecture does better on Banglish than on English.")
    table(doc, "Table 4. Success rate by category, difficulty, language and trap status.",
          "This table gives the values behind Figures 2 and 3 along with difficulty and trap status. Trap covers the 16"
          " tasks marked is_trap in the task file. Each cell is the share of successful runs.",
          ["Group"] + [f"{model_label(m).split()[-1]} {arch_label(a)}" for m in data.models for a in data.archs],
          breakdown_rows(breakdown_df, data.models, data.archs))

    doc.add_heading("Consistency", level=1)
    figure(doc, "fig_pass_k", 3.5,
           "Figure 4. pass^k: chance that k seeds drawn at random from a task all succeed, averaged over tasks.",
           "pass^1 equals the success rate and pass^3 is the share of tasks solved in all three seeds. Every system"
           " falls as k rises, which shows that many tasks are solved in some seeds only."
           f" {model_label(large)} Plan and Execute falls the most, from {plan_pass_1:.1f}% at k = 1 to"
           f" {pct(p14['pass_k'])} at k = 3, and it stays highest at k = 3.")

    doc.add_heading("Policy violations", level=1)
    figure(doc, "fig_policy_violations", 3.5,
           "Figure 5. Policy violation rate on all tasks and on the 16 trap tasks. Bars show 95% bootstrap intervals over"
           " tasks.",
           f"Supervisor has the highest violation rate on trap tasks on both models, {pct(s7['trap_violation_rate'])} on"
           f" {model_label(small)} and {pct(s14['trap_violation_rate'])} on {model_label(large)}. The other two"
           f" architectures range from {pct(r7['trap_violation_rate'])} to {pct(r14['trap_violation_rate'])}. The"
           " intervals are wide and no significance test was run on violations.")

    doc.add_heading("Tokens and latency", level=1)
    figure(doc, "fig_tokens_latency", 7.16,
           "Figure 6. Tokens and agent run time per run. Token bars show means with 95% bootstrap intervals over tasks."
           " Run time bars show the median and the whisker reaches the 95th percentile.",
           f"Plan and Execute uses about {p14['mean_prompt_tokens'] / r14['mean_prompt_tokens']:.1f} times the prompt"
           f" tokens of ReAct on {model_label(large)}, {p14['mean_prompt_tokens'] / 1000:.1f}k against"
           f" {r14['mean_prompt_tokens'] / 1000:.1f}k. Its median run time on that model is {p14['wall_median_s']:.1f} s,"
           f" against {r14['wall_median_s']:.1f} s for ReAct and {s14['wall_median_s']:.1f} s for Supervisor. Plan and"
           f" Execute made {p14['mean_replans']:.2f} replans per run on {model_label(large)}, and Supervisor made"
           f" {s14['mean_delegations']:.2f} delegations per run.")

    doc.add_heading("Not included", level=1)
    text(doc, "The failure category distribution is left out because the failed runs have not been labeled yet."
              " Blank sheets for two labelers are in paper_assets/labeling.")

    path = ASSETS_DIR / "Analysis_Report.docx"
    doc.save(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="main")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"report written to {build(args.tag, args.results_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
