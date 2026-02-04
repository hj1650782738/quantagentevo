#!/usr/bin/env python3
"""
Generate a publication-quality figure (ARR + MDD vs iteration) and compact LaTeX tables
from `iter迭代效果分析.csv`.

Outputs:
  - paper/figs/iteration_arr_mdd.pdf (and .png)
  - paper/tables/iter_convergence_summary.tex
  - paper/tables/iter_convergence_representative.tex
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    iteration: int
    arr_pct: float  # e.g. 29.63
    mdd_pct: float  # e.g. -9.24


def _parse_pct(s: str) -> float:
    s = s.strip()
    if not s.endswith("%"):
        raise ValueError(f"Expected percentage with trailing %; got: {s!r}")
    return float(s[:-1])


def read_rows(csv_path: Path) -> list[Row]:
    rows: list[Row] = []
    # Use utf-8-sig to be robust to BOM in CSV headers.
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r:
                continue
            # Normalize keys defensively (strip whitespace, handle accidental BOM-like prefixes).
            rk = {str(k).strip().lstrip("\ufeff"): v for k, v in r.items()}
            it = int(rk["Iteration"])
            arr = _parse_pct(rk["ARR"])
            mdd = _parse_pct(rk["MDD"])
            rows.append(Row(iteration=it, arr_pct=arr, mdd_pct=mdd))
    rows.sort(key=lambda x: x.iteration)
    return rows


def save_plot(rows: list[Row], out_pdf: Path, out_png: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter

    # Use a clean, paper-friendly style (no extra dependency like seaborn).
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        # Fallback for older matplotlib without the seaborn-v0_8 styles.
        plt.style.use("seaborn-whitegrid")

    mpl.rcParams.update(
        {
            "figure.figsize": (8.6, 3.8),
            "savefig.transparent": False,
            # Typography (academic look)
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,  # embed TrueType fonts (editable text in PDF)
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    x = [r.iteration for r in rows]
    arr = [r.arr_pct for r in rows]
    mdd_abs = [abs(r.mdd_pct) for r in rows]

    # We manage layout manually to reserve space for a figure-level legend.
    fig, ax1 = plt.subplots()

    # Academic, color-blind friendly palette (Okabe–Ito)
    # Swap colors per paper preference: ARR (vermillion), |MDD| (blue).
    color_arr = "#D55E00"  # vermillion
    color_mdd = "#0072B2"  # blue

    # Grid: dashed major + dotted minor for readability
    ax1.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.28)
    ax1.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.18)

    l1 = ax1.plot(
        x,
        arr,
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        color=color_arr,
        label="ARR (excess, %)",
        zorder=3,
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("ARR (%)", color=color_arr)
    ax1.tick_params(axis="y", labelcolor=color_arr)
    ax1.set_xticks(x)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Show percent sign on both y-axes (values are already in % units).
    pct_fmt = FuncFormatter(lambda v, pos: f"{v:.0f}%")
    ax1.yaxis.set_major_formatter(pct_fmt)

    ax2 = ax1.twinx()
    l2 = ax2.plot(
        x,
        mdd_abs,
        marker="s",
        markersize=5.0,
        linewidth=2.0,
        color=color_mdd,
        label="|MDD| (%, lower is better)",
        zorder=2,
    )
    ax2.set_ylabel("|MDD| (%)", color=color_mdd)
    ax2.tick_params(axis="y", labelcolor=color_mdd)
    ax2.yaxis.set_major_formatter(pct_fmt)
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Ensure ARR (ax1) is visually on top of |MDD| (ax2) when elements overlap.
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # Highlight best points (max ARR, min |MDD|)
    best_arr_idx = max(range(len(arr)), key=lambda i: arr[i])
    best_mdd_idx = min(range(len(mdd_abs)), key=lambda i: mdd_abs[i])

    ax1.scatter(
        [x[best_arr_idx]],
        [arr[best_arr_idx]],
        s=64,
        color=color_arr,
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
    )
    ax2.scatter(
        [x[best_mdd_idx]],
        [mdd_abs[best_mdd_idx]],
        s=64,
        color=color_mdd,
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
    )

    # Smart annotation offsets to avoid overlapping with curves / boundaries.
    arr_ymin, arr_ymax = ax1.get_ylim()
    mdd_ymin, mdd_ymax = ax2.get_ylim()

    arr_best_y = arr[best_arr_idx]
    mdd_best_y = mdd_abs[best_mdd_idx]

    arr_offset_y = -18 if arr_best_y > (arr_ymax - 0.7) else 14
    mdd_offset_y = 14 if mdd_best_y < (mdd_ymin + 0.7) else -18

    ax1.annotate(
        f"max ARR: {arr_best_y:.2f}%",
        (x[best_arr_idx], arr_best_y),
        textcoords="offset points",
        xytext=(10, arr_offset_y),
        ha="left",
        va="bottom" if arr_offset_y > 0 else "top",
        fontsize=9,
        color=color_arr,
        zorder=10,
        clip_on=False,
        arrowprops=dict(arrowstyle="->", color=color_arr, lw=0.8, shrinkA=0, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color_arr, lw=0.7, alpha=0.95),
    )
    ax2.annotate(
        f"min |MDD|: {mdd_best_y:.2f}%",
        (x[best_mdd_idx], mdd_best_y),
        textcoords="offset points",
        xytext=(10, mdd_offset_y),
        ha="left",
        va="bottom" if mdd_offset_y > 0 else "top",
        fontsize=9,
        color=color_mdd,
        zorder=10,
        clip_on=False,
        arrowprops=dict(arrowstyle="->", color=color_mdd, lw=0.8, shrinkA=0, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color_mdd, lw=0.7, alpha=0.95),
    )

    # Tighten y-limits slightly so trends are visually comparable.
    ax1.set_ylim(min(arr) - 1.0, max(arr) + 1.0)
    ax2.set_ylim(min(mdd_abs) - 0.8, max(mdd_abs) + 0.8)

    # Subtle highlight for the apparent plateau region (11--13)
    ax1.axvspan(11 - 0.25, 13 + 0.25, color="#888888", alpha=0.08, zorder=0)

    title = "Factor pool performance"
    ax1.set_title(title)

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    # Figure-level legend above axes, with white background to avoid any overlap.
    leg = fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        ncol=2,
        columnspacing=1.4,
        handlelength=2.6,
        fancybox=True,
        borderpad=0.35,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#DDDDDD")
    leg.get_frame().set_linewidth(0.7)
    leg.set_zorder(20)

    # Reserve space at the top for the legend.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def save_tables(rows: list[Row], out_summary: Path, out_repr: Path) -> None:
    # Summary table: all iterations
    summary_lines: list[str] = []
    summary_lines.append("% Auto-generated. Do not edit manually.")
    summary_lines.append("\\small")
    summary_lines.append("\\setlength{\\tabcolsep}{4pt}")
    summary_lines.append("\\renewcommand{\\arraystretch}{1.06}")
    summary_lines.append("\\begin{tabular}{ccc}")
    summary_lines.append("\\toprule")
    summary_lines.append("\\textbf{Iteration} & \\textbf{ARR (\\%)} & \\textbf{MDD (\\%)} \\\\")
    summary_lines.append("\\midrule")
    for r in rows:
        summary_lines.append(f"{r.iteration} & {r.arr_pct:.2f} & {r.mdd_pct:.2f} \\\\")
    summary_lines.append("\\bottomrule")
    summary_lines.append("\\end{tabular}")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Representative table: selected rounds (early / mid / best-risk / best-return)
    by_it = {r.iteration: r for r in rows}
    selected: list[int] = []

    notes = {}
    # Determine programmatically for correctness.
    arr = {r.iteration: r.arr_pct for r in rows}
    mdd_abs = {r.iteration: abs(r.mdd_pct) for r in rows}
    it_max_arr = max(arr, key=arr.get)
    it_min_abs_mdd = min(mdd_abs, key=mdd_abs.get)

    # Mid iteration: closest to the median iteration number.
    iters_sorted = sorted(arr.keys())
    median_it = iters_sorted[len(iters_sorted) // 2]

    for it in (rows[0].iteration, median_it, it_min_abs_mdd, it_max_arr):
        if it in by_it and it not in selected:
            selected.append(it)
    selected.sort()

    for it in selected:
        if it == rows[0].iteration:
            notes[it] = "First reported round in this appendix (Section 5.5 covers Iter 1--5)."
        elif it == median_it:
            notes[it] = "Mid stage; performance approaches a plateau."
        elif it == it_min_abs_mdd:
            notes[it] = "Lowest drawdown magnitude among Iter 5--15."
        elif it == it_max_arr:
            notes[it] = "Highest ARR among Iter 5--15."
        else:
            notes[it] = "Representative round."

    repr_lines: list[str] = []
    repr_lines.append("% Auto-generated. Do not edit manually.")
    repr_lines.append("\\scriptsize")
    repr_lines.append("\\setlength{\\tabcolsep}{3pt}")
    repr_lines.append("\\renewcommand{\\arraystretch}{1.06}")
    repr_lines.append("\\begin{tabular}{cccp{6.6cm}}")
    repr_lines.append("\\toprule")
    repr_lines.append("\\textbf{Iteration} & \\textbf{ARR (\\%)} & \\textbf{MDD (\\%)} & \\textbf{Note} \\\\")
    repr_lines.append("\\midrule")
    for it in selected:
        r = by_it[it]
        note = _latex_escape(notes[it])
        repr_lines.append(f"{r.iteration} & {r.arr_pct:.2f} & {r.mdd_pct:.2f} & {note} \\\\")
    repr_lines.append("\\bottomrule")
    repr_lines.append("\\end{tabular}")
    out_repr.write_text("\n".join(repr_lines) + "\n", encoding="utf-8")


def main() -> None:
    base = Path(__file__).resolve().parents[1]  # .../paper
    csv_path = base / "iter迭代效果分析.csv"
    rows = read_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows loaded from {csv_path}")

    out_pdf = base / "figs" / "iteration_arr_mdd.pdf"
    out_png = base / "figs" / "iteration_arr_mdd.png"
    save_plot(rows, out_pdf=out_pdf, out_png=out_png)

    out_summary = base / "tables" / "iter_convergence_summary.tex"
    out_repr = base / "tables" / "iter_convergence_representative.tex"
    save_tables(rows, out_summary=out_summary, out_repr=out_repr)

    print(f"Saved figure: {out_pdf}")
    print(f"Saved figure: {out_png}")
    print(f"Saved tables: {out_summary}, {out_repr}")


if __name__ == "__main__":
    main()


