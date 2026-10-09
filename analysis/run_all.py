"""Run the whole analysis for one run folder.

Reads results/<tag>/runs.jsonl and meta.json (and the traces) and writes every
table, figure and sheet to results/<tag>/analysis/. No GPU is needed.

Usage (PowerShell):
    .venv\\Scripts\\python.exe -m analysis.run_all --tag main
"""

import argparse
import shutil
import sys
from pathlib import Path

from analysis import figures, latex, report, signals, stats
from analysis.load import RESULTS_DIR, EmptyRunError, load_run
from analysis.summary import build_summary

GUIDE_PATH = Path(__file__).resolve().parent / "labeling_guide.md"


def write_sheet(sheet, path, notes):
    """Write a labeling sheet unless an existing one already holds labels."""
    if signals.has_labels(path):
        fresh = path.with_name(path.stem + ".new.csv")
        sheet.to_csv(fresh, index=False, encoding="utf-8-sig")
        notes.append(f"{path.name} already holds labels and was kept; the new sample went to {fresh.name}.")
        return fresh
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run(tag, results_dir=RESULTS_DIR, n_resamples=stats.N_RESAMPLES):
    run_dir = Path(results_dir) / tag
    data = load_run(run_dir, tag)
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def keep(path):
        written.append(Path(path).name)
        return path

    k = report.pass_k_value(data)
    main_df = report.main_results(data, n_resamples)
    main_df.to_csv(keep(out / "main_results.csv"), index=False)
    breakdown_df = report.breakdowns(data)
    breakdown_df.to_csv(keep(out / "breakdowns.csv"), index=False)
    tests_df = report.mcnemar_tests(data)
    tests_df.to_csv(keep(out / "mcnemar.csv"), index=False)

    uncorrected = any("lag correction" in note for note in data.notes)
    energy_note = " Sampled energy without the power lag correction (older run)." if uncorrected else ""
    keep(out / "table_main_results.tex").write_text(
        latex.main_table(main_df, tag, k, len(data.seeds), energy_note), encoding="utf-8")
    keep(out / "table_mcnemar.tex").write_text(latex.mcnemar_table(tests_df, tag, len(data.seeds)), encoding="utf-8")

    font_check = {}
    for paths in (
        figures.cost_accuracy(main_df, out, data.models, data.archs),
        figures.category_bars(breakdown_df, out, data.models, data.archs),
        figures.language_bars(breakdown_df, out, data.models, data.archs),
    ):
        for path in paths:
            keep(path)
            if path.suffix == ".pdf":
                font_check[path.name] = figures.pdf_font_subtypes(path)

    signals_df = signals.failure_signals(data)
    signals_df.to_csv(keep(out / "failures_auto.csv"), index=False, encoding="utf-8-sig")
    sheet = signals.labeling_sheet(data, signals_df)
    keep(write_sheet(sheet, out / "failures_to_label.csv", data.notes))
    for reviewer in (1, 2):
        keep(write_sheet(signals.reviewer_copy(sheet, reviewer), out / f"failures_to_label_reviewer{reviewer}.csv",
                         data.notes))
    shutil.copyfile(GUIDE_PATH, keep(out / "labeling_guide.md"))

    keep(out / "summary.md")
    (out / "summary.md").write_text(
        build_summary(data, main_df, tests_df, breakdown_df, signals_df, font_check, written, k), encoding="utf-8")
    return data, out, written, font_check


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="main")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--resamples", type=int, default=stats.N_RESAMPLES)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        data, out, written, font_check = run(args.tag, args.results_dir, args.resamples)
    except EmptyRunError as error:
        print(f"STOP: nothing to analyse: {error}")
        return 1
    print(f"{len(data.rows)} runs from results/{args.tag}: models {data.models}, archs {data.archs},"
          f" seeds {data.seeds}")
    print(f"wrote {len(written)} files to {out}")
    for name, subtypes in font_check.items():
        print(f"  {name}: fonts {', '.join(subtypes)}" + ("  TYPE 3 FOUND" if "Type3" in subtypes else ""))
    for note in data.notes:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
