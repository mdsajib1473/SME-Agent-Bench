"""Analysis pipeline tests on small synthetic run folders with known answers."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import agreement, figures, report, run_all, signals, stats
from analysis.load import load_run, text_sha256

CATEGORIES = ("refund", "order_change", "complaint_routing", "quotation", "inquiry")


def make_tasks(n):
    tasks = []
    for i in range(n):
        tasks.append({
            "task_id": f"T{i:02d}",
            "category": CATEGORIES[i % len(CATEGORIES)],
            "difficulty": ("easy", "medium", "hard")[i % 3],
            "language": "en" if i % 2 == 0 else "banglish",
            "is_trap": i % 4 == 0,
            "instruction": f"My order ORD-10{i:02d}, phone 0171234567{i % 10}. Please refund 500 taka.",
            "gold_actions": [{"name": "issue_refund", "arguments": {"order_id": f"ORD-10{i:02d}"}}],
            "required_outputs": [500],
            "forbidden_actions": [],
        })
    return tasks


def make_row(task, model, arch, seed, success, energy=0.01, **extra):
    row = {
        "task_id": task["task_id"], "category": task["category"], "difficulty": task["difficulty"],
        "language": task["language"], "is_trap": task["is_trap"], "model": model, "arch": arch, "seed": seed,
        "success": success, "state_match": success, "output_match": True, "policy_violation": False,
        "empty_reply": False, "wrong_script": False, "llm_calls": 2, "tool_calls": 1, "malformed_tool_calls": 0,
        "text_tool_calls": 0, "dropped_tool_calls": 0, "prompt_tokens": 1000, "completion_tokens": 100,
        "wall_time_s": 5.0, "llm_latency_s": 4.9, "energy_wh": energy + 0.02, "net_energy_wh": energy,
        "energy_sampled_raw_wh": energy, "energy_counter_wh": energy + 0.03, "net_energy_counter_wh": energy * 1.1,
        "budget_exceeded": False, "llm_timeout": False, "stop_reason": "final_answer",
        "delegations": 1 if arch == "supervisor" else None, "replans": 0 if arch == "plan_execute" else None,
        "settle_seconds": 5, "idle_w_used": 12.0, "harness_sha256": "h" * 64,
        "trace_file": f"traces/{model}__{arch}__{task['task_id']}__s{seed}.json",
    }
    row.update(extra)
    return row


def write_run(base, tag, rows, tasks, traces=None):
    run_dir = base / tag
    (run_dir / "traces").mkdir(parents=True)
    tasks_path = base / f"{tag}_tasks.jsonl"
    tasks_path.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    (run_dir / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    meta = {"tasks_path": str(tasks_path), "tasks_sha256": text_sha256(tasks_path), "idle_power_w": 12.0,
            "harness_sha256": "h" * 64}
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for name, trace in (traces or {}).items():
        (run_dir / "traces" / f"{name}.json").write_text(json.dumps(trace), encoding="utf-8")
    return run_dir


@pytest.fixture
def two_arch_run(tmp_path):
    """One model, react and supervisor, 10 tasks, 1 seed.

    react solves T00 to T07, supervisor solves T06 to T09: both 2, react only 6,
    supervisor only 2, neither 0.
    """
    tasks = make_tasks(10)
    rows = [make_row(t, "m-7b", "react", 0, i <= 7, energy=0.02) for i, t in enumerate(tasks)]
    rows += [make_row(t, "m-7b", "supervisor", 0, i >= 6, energy=0.05) for i, t in enumerate(tasks)]
    write_run(tmp_path, "syn", rows, tasks)
    return tmp_path


# Statistics with hand computed answers


def test_bootstrap_interval_contains_the_true_rate_and_is_reproducible():
    values = [1.0] * 60 + [0.0] * 40
    low, high = stats.bootstrap_ci(values, n_resamples=10_000, seed=0)
    assert low < 0.6 < high
    assert 0.08 < high - low < 0.25
    assert stats.bootstrap_ci(values, n_resamples=10_000, seed=0) == (low, high)
    assert all(math.isnan(v) for v in stats.bootstrap_ci([]))


def test_mcnemar_exact_on_hand_computed_tables():
    assert stats.mcnemar_exact(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 2**10)
    assert stats.mcnemar_exact(6, 2) == pytest.approx(2 * (1 + 8 + 28) / 2**8)
    assert stats.mcnemar_exact(5, 0) == pytest.approx(2 / 2**5)
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(3, 3) == 1.0
    assert stats.discordant_odds_ratio(8, 2) == 4.0
    assert math.isinf(stats.discordant_odds_ratio(5, 0))
    assert math.isnan(stats.discordant_odds_ratio(0, 0))


def test_holm_adjustment():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    adjusted = stats.holm([0.02, math.nan])
    assert adjusted[0] == pytest.approx(0.02) and math.isnan(adjusted[1])


def test_pass_hat_k_and_majority_vote():
    tasks = [[True, True, True], [True, False, True], [False, False, False], [True, True, True]]
    assert stats.pass_hat_k(tasks, 3) == (0.5, 4)
    assert stats.pass_hat_k(tasks, 2) == (0.5, 4)
    assert stats.pass_hat_k(tasks, 1) == (0.75, 4)
    assert stats.pass_hat_k([[True], [True, True]], 2) == (1.0, 1)
    assert stats.majority_success([True, True, False]) is True
    assert stats.majority_success([True, False, False]) is False
    assert stats.majority_success([True, False]) is False
    assert stats.majority_success([True]) is True


def test_cohen_kappa_known_value():
    assert stats.cohen_kappa(["A", "A", "B", "B"], ["A", "B", "B", "B"]) == pytest.approx(0.5)
    assert stats.cohen_kappa(["A", "B"], ["A", "B"]) == pytest.approx(1.0)
    assert stats.cohen_kappa(["A", "A"], ["A", "A"]) == 1.0


# Tables on the synthetic run


def test_main_results_success_energy_per_success_and_pass_k(two_arch_run):
    data = load_run(two_arch_run / "syn")
    main = report.main_results(data, n_resamples=500).set_index("arch")
    assert main.loc["react", "success_rate"] == pytest.approx(0.8)
    assert main.loc["supervisor", "success_rate"] == pytest.approx(0.4)
    assert main.loc["react", "pass_k"] == pytest.approx(0.8) and main.loc["react", "k"] == 1
    assert main.loc["react", "energy_per_success_wh"] == pytest.approx(10 * 0.02 / 8)
    assert main.loc["supervisor", "energy_per_success_wh"] == pytest.approx(10 * 0.05 / 4)
    assert main.loc["supervisor", "energy_per_success_counter_wh"] == pytest.approx(10 * 0.055 / 4)
    assert main.loc["react", "trap_runs"] == 3
    assert main.loc["supervisor", "mean_delegations"] == 1.0
    assert math.isnan(main.loc["react", "mean_delegations"])


def test_energy_per_success_is_undefined_without_successes():
    assert math.isnan(report.energy_per_success(pd.Series([0.1, 0.2]), 0))
    assert report.energy_per_success(pd.Series([0.1, 0.2, None]), 2) == pytest.approx(0.15)


def test_mcnemar_on_the_synthetic_table(two_arch_run):
    tests = report.mcnemar_tests(load_run(two_arch_run / "syn"))
    row = tests.iloc[0]
    assert (row["both_success"], row["a_only"], row["b_only"], row["both_fail"]) == (2, 6, 2, 0)
    assert row["odds_ratio"] == 3.0
    assert row["p_exact"] == pytest.approx(74 / 256)
    assert row["p_holm"] == pytest.approx(74 / 256)


def test_majority_vote_with_three_seeds_and_two_seed_ties(tmp_path):
    tasks = make_tasks(4)
    pattern_a = {"T00": [1, 1, 0], "T01": [1, 0, 0], "T02": [1, 1, 1], "T03": [0, 0, 0]}
    pattern_b = {"T00": [0, 0, 0], "T01": [1, 1, 0], "T02": [1, 1, 1], "T03": [0, 0, 1]}
    rows = []
    for arch, pattern in (("react", pattern_a), ("supervisor", pattern_b)):
        for t in tasks:
            rows += [make_row(t, "m-7b", arch, s, bool(v)) for s, v in enumerate(pattern[t["task_id"]])]
    write_run(tmp_path, "three", rows, tasks)
    row = report.mcnemar_tests(load_run(tmp_path / "three")).iloc[0]
    assert (row["both_success"], row["a_only"], row["b_only"]) == (1, 1, 1)
    main = report.main_results(load_run(tmp_path / "three"), n_resamples=200).set_index("arch")
    assert main.loc["react", "k"] == 3
    assert main.loc["react", "pass_k"] == pytest.approx(0.25)

    two_seed_rows = [r for r in rows if r["seed"] < 2]
    write_run(tmp_path, "two", two_seed_rows, tasks)
    data = load_run(tmp_path / "two")
    assert report.pass_k_value(data) == 2
    row = report.mcnemar_tests(data).iloc[0]
    # Seeds 0 and 1: react wins T00 (1,1) and ties T01 (1,0, a failure); supervisor wins T01 (1,1).
    assert (row["both_success"], row["a_only"], row["b_only"]) == (1, 1, 1)


# Failure signals and the labeling sheet


def supervisor_trace(task):
    return {
        "final_reply": "Done.",
        "score": {"diff": "refunds: missing row REF-0001", "missing_outputs": [500], "violations": []},
        "trace": [
            {"type": "message", "agent": "supervisor", "role": "user", "content": task["instruction"]},
            {"type": "llm_call", "agent": "supervisor", "tool_calls": [
                {"name": "SupportAgent", "raw_arguments": json.dumps({"instruction": "refund ORD-1000 please"})}]},
            *[{"type": "tool_call", "agent": "SupportAgent", "name": "get_order",
               "arguments": {"order_id": "ORD-1000"}, "malformed": False, "result": {"ok": True}}] * 3,
            {"type": "delegation", "index": 1, "specialist": "SupportAgent",
             "instruction": "refund ORD-1000 please", "refused": None},
        ],
    }


def test_failure_signals_for_a_supervisor_run(tmp_path):
    tasks = make_tasks(2)
    rows = [make_row(tasks[0], "m-7b", "supervisor", 0, False, output_match=False),
            make_row(tasks[1], "m-7b", "supervisor", 0, True)]
    write_run(tmp_path, "sig", rows, tasks, {"m-7b__supervisor__T00__s0": supervisor_trace(tasks[0])})
    data = load_run(tmp_path / "sig")
    found = signals.failure_signals(data)
    assert len(found) == 1
    row = found.iloc[0]
    assert row["loop"] == 1
    assert row["handoff_missing_tool"] == 1
    assert row["handoff_missing_identifier"] == 1
    assert row["wrong_final_state"] == 1 and row["missing_required_output"] == 1
    assert row["budget_exceeded"] == 0 and row["policy_violation"] == 0
    assert "01712345670" in row["details"] and "issue_refund" in row["details"]
    summary = signals.trace_summary(data.traces[row["run_id"]], "supervisor")
    assert summary.startswith('1. delegate to SupportAgent: "refund ORD-1000 please"; 2. SupportAgent: get_order(')


def test_labeling_sheet_samples_per_cell_and_hides_the_other_reviewer(two_arch_run):
    data = load_run(two_arch_run / "syn")
    found = signals.failure_signals(data)
    sheet = signals.labeling_sheet(data, found, per_cell=1)
    assert len(sheet) == 2
    assert set(sheet["arch"]) == {"react", "supervisor"}
    assert list(sheet.columns) == list(signals.SHEET_COLUMNS)
    assert (sheet["label_1"] == "").all() and (sheet["label_2"] == "").all()
    assert sheet.equals(signals.labeling_sheet(data, found, per_cell=1))
    assert "label_2" not in signals.reviewer_copy(sheet, 1).columns
    assert "label_1" not in signals.reviewer_copy(sheet, 2).columns
    assert len(signals.labeling_sheet(data, found, per_cell=25)) == len(found) == 8


# The whole pipeline


def test_run_all_writes_every_output_without_type3_fonts(two_arch_run):
    data, out, written, font_check = run_all.run("syn", two_arch_run, n_resamples=200)
    expected = {
        "main_results.csv", "breakdowns.csv", "mcnemar.csv", "table_main_results.tex", "table_mcnemar.tex",
        "fig_cost_accuracy.pdf", "fig_cost_accuracy.png", "fig_category.pdf", "fig_category.png",
        "fig_language.pdf", "fig_language.png", "failures_auto.csv", "failures_to_label.csv",
        "failures_to_label_reviewer1.csv", "failures_to_label_reviewer2.csv", "labeling_guide.md", "summary.md",
    }
    assert expected <= set(written)
    for name in expected:
        assert (out / name).exists()
    for pdf in ("fig_cost_accuracy.pdf", "fig_category.pdf", "fig_language.pdf"):
        assert not figures.type3_fonts(out / pdf)
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Limits of this run" in summary
    assert "only 1 seed(s) are available" in summary
    tex = (out / "table_main_results.tex").read_text(encoding="utf-8")
    assert "\\toprule" in tex and "\\footnotesize" in tex and "\\setlength{\\tabcolsep}{3pt}" in tex
    for name in ("summary.md", "table_main_results.tex", "table_mcnemar.tex", "labeling_guide.md"):
        text = (out / name).read_text(encoding="utf-8")
        assert "\u2014" not in text and "\u2013" not in text and " - " not in text
        if name != "labeling_guide.md":  # the guide quotes a command line with -- flags
            assert "--" not in text


def test_run_all_keeps_a_sheet_that_already_holds_labels(two_arch_run):
    _, out, _, _ = run_all.run("syn", two_arch_run, n_resamples=100)
    sheet = pd.read_csv(out / "failures_to_label.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sheet.loc[0, "label_1"] = "LOOP"
    sheet.to_csv(out / "failures_to_label.csv", index=False, encoding="utf-8-sig")
    data, _, _, _ = run_all.run("syn", two_arch_run, n_resamples=100)
    kept = pd.read_csv(out / "failures_to_label.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert kept.loc[0, "label_1"] == "LOOP"
    assert (out / "failures_to_label.new.csv").exists()
    assert any("already holds labels" in note for note in data.notes)


def test_older_run_fields_fall_back_and_are_noted(tmp_path):
    tasks = make_tasks(3)
    rows = []
    for i, t in enumerate(tasks):
        row = make_row(t, "m-7b", "react", 0, i == 0)
        for field in ("wrong_script", "empty_reply", "dropped_tool_calls", "net_energy_counter_wh",
                      "energy_counter_wh", "energy_sampled_raw_wh", "settle_seconds", "idle_w_used",
                      "harness_sha256"):
            row.pop(field)
        row["counter_energy_wh"] = 0.05
        rows.append(row)
    trace = {"final_reply": "আপনার order", "trace": [
        {"type": "llm_call", "agent": "react", "finish_reason": "stop", "content": "", "tool_calls": [],
         "completion_tokens": 12}]}
    write_run(tmp_path, "old", rows, tasks, {"m-7b__react__T01__s0": trace})
    (tmp_path / "old" / "meta.json").write_text(json.dumps({"idle_power_w": 12.0}), encoding="utf-8")
    data = load_run(tmp_path / "old")
    by_task = data.rows.set_index("task_id")
    assert by_task.loc["T01", "wrong_script"] is True
    assert by_task.loc["T01", "dropped_tool_calls"] == 1
    assert by_task.loc["T00", "wrong_script"] is None
    assert by_task.loc["T00", "net_energy_counter_wh"] == pytest.approx(0.05 - 12.0 * 5.0 / 3600.0)
    text = " ".join(data.notes)
    for phrase in ("wrong_script is missing", "dropped_tool_calls is missing", "net_energy_counter_wh is missing",
                   "power lag correction", "settle wait", "harness_sha256"):
        assert phrase in text


def test_empty_or_missing_run_folder_stops_cleanly(tmp_path, capsys):
    assert run_all.main(["--tag", "nothing", "--results-dir", str(tmp_path)]) == 1
    (tmp_path / "blank").mkdir()
    (tmp_path / "blank" / "runs.jsonl").write_text("\n", encoding="utf-8")
    assert run_all.main(["--tag", "blank", "--results-dir", str(tmp_path)]) == 1
    assert "nothing to analyse" in capsys.readouterr().out
    assert not (tmp_path / "blank" / "analysis").exists()


# Agreement


def test_agreement_from_the_sheet_and_from_reviewer_copies(tmp_path, capsys):
    sheet = pd.DataFrame({
        "run_id": ["a", "b", "c", "d", "e"],
        "label_1": ["loop", "LOOP", "wrong tool", "WRONG_TOOL", "OTHER"],
        "label_2": ["LOOP", "WRONG_TOOL", "WRONG_TOOL", "WRONG_TOOL", ""],
    })
    path = tmp_path / "sheet.csv"
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    result = agreement.agreement(agreement.load_labels(path))
    assert result["n_labeled"] == 4 and result["n_skipped"] == 1
    assert result["kappa"] == pytest.approx(0.5)
    assert result["confusion"].loc["LOOP", "WRONG_TOOL"] == 1

    sheet[["run_id", "label_1"]].to_csv(tmp_path / "r1.csv", index=False, encoding="utf-8-sig")
    sheet[["run_id", "label_2"]].to_csv(tmp_path / "r2.csv", index=False, encoding="utf-8-sig")
    assert agreement.main(["--reviewer1", str(tmp_path / "r1.csv"), "--reviewer2", str(tmp_path / "r2.csv")]) == 0
    out = capsys.readouterr().out
    assert "Cohen's kappa 0.500" in out and "WRONG_TOOL" in out
