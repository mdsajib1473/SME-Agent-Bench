"""IEEE friendly figures: 3.5 in wide, 300 dpi, PDF and PNG, grayscale safe, TrueType fonts."""

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.load import arch_label, category_label, language_label, model_label, ordered, CATEGORY_ORDER

WIDTH_IN = 3.5
STYLE = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 0.8,
    "hatch.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}
ARCH_STYLE = {
    "react": {"marker": "o", "hatch": "", "face": "0.25"},
    "plan_execute": {"marker": "s", "hatch": "////", "face": "0.6"},
    "supervisor": {"marker": "^", "hatch": "....", "face": "0.9"},
}
FALLBACK_STYLES = [{"marker": "D", "hatch": "xxxx", "face": "0.45"}, {"marker": "v", "hatch": "\\\\\\\\", "face": "0.75"}]
LANGUAGE_STYLE = {"en": {"hatch": "", "face": "0.35"}, "banglish": {"hatch": "////", "face": "0.85"}}


def arch_style(arch, archs):
    if arch in ARCH_STYLE:
        return ARCH_STYLE[arch]
    extra = [a for a in archs if a not in ARCH_STYLE]
    return FALLBACK_STYLES[extra.index(arch) % len(FALLBACK_STYLES)]


def wrap(label):
    return textwrap.fill(label, 10)


def save(fig, out_dir, name):
    paths = []
    for suffix in ("pdf", "png"):
        path = Path(out_dir) / f"{name}.{suffix}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def cost_accuracy(main_df, out_dir, models, archs):
    """Mean net energy per task against success rate, one point per model and architecture."""
    with matplotlib.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.4))
        plotted = main_df.dropna(subset=["mean_net_energy_wh"])
        for _, row in plotted.iterrows():
            style = arch_style(row["arch"], archs)
            filled = models.index(row["model"]) % 2 == 0
            x, y = row["mean_net_energy_wh"], row["success_rate"] * 100
            xerr = [[max(x - row["net_energy_ci_low"], 0)], [max(row["net_energy_ci_high"] - x, 0)]]
            yerr = [[max(y - row["success_ci_low"] * 100, 0)], [max(row["success_ci_high"] * 100 - y, 0)]]
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr, fmt=style["marker"], markersize=5,
                markerfacecolor="black" if filled else "white", markeredgecolor="black",
                ecolor="0.4", elinewidth=0.6, capsize=1.5, markeredgewidth=0.8,
            )
        ax.set_xlabel("Net GPU energy per task (Wh)")
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(-5, 105)
        ax.set_xlim(left=0)
        arch_handles = [Line2D([], [], linestyle="none", marker=arch_style(a, archs)["marker"], color="black",
                               markerfacecolor="0.6", markersize=5, label=arch_label(a)) for a in archs]
        model_handles = [Patch(facecolor="black" if i % 2 == 0 else "white", edgecolor="black", linewidth=0.6,
                               label=f"{model_label(m)} ({'filled' if i % 2 == 0 else 'open'})")
                         for i, m in enumerate(models)]
        # Legends fill column by column; interleave so architectures form the first row and models the second.
        ncol = max(len(arch_handles), len(model_handles))
        blank = Patch(facecolor="none", edgecolor="none", label=" ")
        handles = []
        for j in range(ncol):
            handles.append(arch_handles[j] if j < len(arch_handles) else blank)
            handles.append(model_handles[j] if j < len(model_handles) else blank)
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=ncol,
                  handletextpad=0.3, columnspacing=0.9, handlelength=1.0)
        return save(fig, out_dir, "fig_cost_accuracy")


def grouped_bars(ax, groups, series, values, styles, labels_for_legend):
    width = 0.8 / max(len(series), 1)
    x = np.arange(len(groups))
    for i, key in enumerate(series):
        heights = [values.get((g, key), np.nan) for g in groups]
        style = styles[key]
        ax.bar(x - 0.4 + width * (i + 0.5), [0 if h != h else h for h in heights], width,
               color=style["face"], hatch=style["hatch"], edgecolor="black", linewidth=0.5,
               label=labels_for_legend.get(key, key))
        for xi, h in zip(x, heights):
            if h != h:
                ax.text(xi - 0.4 + width * (i + 0.5), 2, "n/a", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    # Headroom above 100 % holds the panel's model name clear of the bars.
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20))


def bar_panels(breakdown_df, dimension, out_dir, name, models, groups, series, values_key, styles, tick_labels,
               legend_labels):
    """One panel per model, stacked, bars grouped by `groups` with one bar per `series` entry."""
    rows = breakdown_df[breakdown_df["dimension"] == dimension]
    with matplotlib.rc_context(STYLE):
        fig, axes = plt.subplots(len(models), 1, figsize=(WIDTH_IN, 1.35 * len(models) + 0.45),
                                 sharex=True, squeeze=False)
        for ax, model in zip(axes[:, 0], models):
            mine = rows[rows["model"] == model]
            values = {values_key(r): r["success_rate"] * 100 for _, r in mine.iterrows()}
            grouped_bars(ax, groups, series, values, styles, legend_labels)
            ax.text(0.01, 0.99, model_label(model), transform=ax.transAxes, ha="left", va="top", fontsize=7)
        axes[-1, 0].set_xticklabels(tick_labels)
        fig.supylabel("Success rate (%)", fontsize=8, x=0.01)
        axes[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series), frameon=False,
                          handlelength=1.6, columnspacing=1.0)
        return save(fig, out_dir, name)


def category_bars(breakdown_df, out_dir, models, archs):
    categories = ordered(breakdown_df[breakdown_df["dimension"] == "category"]["level"], CATEGORY_ORDER)
    return bar_panels(
        breakdown_df, "category", out_dir, "fig_category", models, categories, archs,
        lambda r: (r["level"], r["arch"]), {a: arch_style(a, archs) for a in archs},
        [wrap(category_label(c)) for c in categories], {a: arch_label(a) for a in archs},
    )


def language_bars(breakdown_df, out_dir, models, archs):
    languages = ordered(breakdown_df[breakdown_df["dimension"] == "language"]["level"], ("en", "banglish"))
    return bar_panels(
        breakdown_df, "language", out_dir, "fig_language", models, archs, languages,
        lambda r: (r["arch"], r["level"]),
        {lang: LANGUAGE_STYLE.get(lang, {"hatch": "xxxx", "face": "0.6"}) for lang in languages},
        [arch_label(a) for a in archs], {lang: language_label(lang) for lang in languages},
    )


FONT_SUBTYPE = re.compile(rb"/Subtype\s*/(Type3|Type1C?|TrueType|Type0|CIDFontType[02]|OpenType)")


def pdf_font_subtypes(path):
    return sorted({m.group(1).decode() for m in FONT_SUBTYPE.finditer(Path(path).read_bytes())})


def type3_fonts(path):
    return "Type3" in pdf_font_subtypes(path)
