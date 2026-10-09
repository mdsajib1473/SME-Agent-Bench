"""Inter-rater agreement on the failure labels: Cohen's kappa and a confusion table.

Reads label_1 and label_2 from results/<tag>/analysis/failures_to_label.csv, or
merges the two reviewer copies by run_id when they are given. Runs where either
label is empty are left out and counted.

Usage (PowerShell):
    .venv\\Scripts\\python.exe -m analysis.agreement --tag main
    .venv\\Scripts\\python.exe -m analysis.agreement --tag main --reviewer1 r1.csv --reviewer2 r2.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from analysis.load import RESULTS_DIR
from analysis.stats import cohen_kappa

KNOWN_LABELS = (
    "SYSTEM_DESIGN", "MISALIGNMENT", "VERIFICATION", "WRONG_TOOL", "WRONG_ARGUMENTS", "PREMATURE_STOP",
    "LOOP", "POLICY_BREACH", "HANDOFF_ERROR", "OTHER",
)


def normalize(label):
    return re.sub(r"[\s\-]+", "_", str(label).strip()).upper()


def read_sheet(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def load_labels(sheet_path=None, reviewer1=None, reviewer2=None):
    """DataFrame with run_id, label_1 and label_2 (normalized; empty string when missing)."""
    if reviewer1 or reviewer2:
        if not (reviewer1 and reviewer2):
            raise ValueError("give both reviewer copies")
        first = read_sheet(reviewer1)[["run_id", "label_1"]]
        second = read_sheet(reviewer2)[["run_id", "label_2"]]
        labels = first.merge(second, on="run_id", how="outer").fillna("")
    else:
        labels = read_sheet(sheet_path)[["run_id", "label_1", "label_2"]]
    for column in ("label_1", "label_2"):
        labels[column] = labels[column].map(lambda v: normalize(v) if str(v).strip() else "")
    return labels


def agreement(labels):
    complete = labels[(labels["label_1"] != "") & (labels["label_2"] != "")]
    kappa = cohen_kappa(complete["label_1"], complete["label_2"])
    observed = (complete["label_1"] == complete["label_2"]).mean() if len(complete) else float("nan")
    order = [l for l in KNOWN_LABELS if l in set(complete["label_1"]) | set(complete["label_2"])]
    order += sorted((set(complete["label_1"]) | set(complete["label_2"])) - set(order))
    table = pd.crosstab(
        pd.Categorical(complete["label_1"], categories=order),
        pd.Categorical(complete["label_2"], categories=order),
        rownames=["reviewer 1"], colnames=["reviewer 2"], dropna=False,
    )
    unknown = sorted((set(complete["label_1"]) | set(complete["label_2"])) - set(KNOWN_LABELS))
    return {
        "n_labeled": len(complete),
        "n_skipped": len(labels) - len(complete),
        "kappa": kappa,
        "observed_agreement": observed,
        "confusion": table,
        "unknown_labels": unknown,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="main")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--sheet", type=Path, help="sheet with both label columns (default: the run's sheet)")
    parser.add_argument("--reviewer1", type=Path)
    parser.add_argument("--reviewer2", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sheet = args.sheet or args.results_dir / args.tag / "analysis" / "failures_to_label.csv"
    try:
        labels = load_labels(sheet, args.reviewer1, args.reviewer2)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"STOP: cannot read labels: {error}")
        return 1
    result = agreement(labels)
    if result["n_labeled"] == 0:
        print(f"STOP: no run has both labels yet ({result['n_skipped']} run(s) in the sheet).")
        return 1
    print(f"{result['n_labeled']} run(s) with both labels; {result['n_skipped']} skipped (a label is empty)")
    print(f"observed agreement {result['observed_agreement']:.3f}, Cohen's kappa {result['kappa']:.3f}")
    if result["unknown_labels"]:
        print(f"labels not in the guide: {', '.join(result['unknown_labels'])}")
    print()
    print("confusion table (rows reviewer 1, columns reviewer 2)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(result["confusion"].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
