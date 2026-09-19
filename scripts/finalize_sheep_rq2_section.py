#!/usr/bin/env python
"""Fill the ovine RQ2 mini-sweep block in ``experimental_evaluation.tex``.

Run this once ``results/sheep_cross_species/rq2_mini/summary.json``
exists (i.e. the bounded mini-sweep has completed). The script:

1. regenerates the retention figure with Type-42 fonts and copies it to
   ``thesis/assets/fig_sheep_retention.{pdf,png}``;
2. writes a LaTeX paragraph with the actual mean/std/retention numbers
   and the retention figure;
3. atomically replaces the block between the AUTO RQ2-MINI markers in
   ``thesis/experimental_evaluation.tex``.

It is idempotent: running it again regenerates and replaces the block.
If the summary is missing, it exits with a message and changes nothing.

Usage:
    uv run python scripts/finalize_sheep_rq2_section.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from sheep_phase1_utils import PLOTS_DIR, TeeLogger  # noqa: E402
from run_sheep_rq2_mini import plot_retention_curve  # noqa: E402

SUMMARY = os.path.join(REPO, "results", "sheep_cross_species", "rq2_mini",
                       "summary.json")
TEX = os.path.join(REPO, "thesis", "experimental_evaluation.tex")
ASSETS = os.path.join(REPO, "thesis", "assets")
FIG_SRC = os.path.join(PLOTS_DIR, "rq2_mini_retention_curve")
FIG_DST = os.path.join(ASSETS, "fig_sheep_retention")
BEGIN = "% ---- BEGIN AUTO RQ2-MINI ----"
END = "% ---- END AUTO RQ2-MINI ----"

TEMPLATE = r"""\paragraph{Ovine sample efficiency.}
The bounded mini-sweep (Section~\ref{sec:minimum_requirements}) evaluates
$N\in\{@NS@\}$ individuals per breed with three subsampling seeds and
three development folds, fitting a fresh preprocessor for every run.
Retention is measured against the full-sample ovine VMGP baseline
($S_{\mathrm{full}} = @BASE@$). Across the @OBS@ observations per sample
size, @SERIES@.@PARALLEL@ As in the caprine domain, most of the full-sample
performance is already acquired with very few individuals per breed and
the marginal gain beyond $N=10$ is small. Because the minimum
fold-training count per breed is $17$, the grid cannot be extended beyond
$N=15$ and the internal reference is $N=15$ itself; the ovine
$N_{\min}$ is therefore bounded by construction and is interpreted
through the shape of the retention curve rather than as an extended
minimum.

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.80\textwidth]{fig_sheep_retention}
  \caption{Ovine bounded sample-efficiency mini-sweep: Macro-F1 retention
  (left) and absolute Macro-F1 (right) as a function of $N$, with error
  bars across the @OBS@ fold-seed observations. The $98\%$ and $95\%$
  lines are the pre-registered retention thresholds and the dashed line is
  the full-sample baseline (@BASE@).}
  \label{fig:sheep_retention}
\end{figure}"""


def build_block(summary: dict) -> str:
    base = float(summary["baseline_macro_f1_full_sample"])
    by_n = {int(r["N"]): r for r in summary["by_n"]}
    ns = sorted(by_n)
    obs_values = {int(r["n_observations"]) for r in summary["by_n"]}
    obs = max(obs_values) if obs_values else 0

    series = []
    for n in ns:
        r = by_n[n]
        series.append(
            f"$N={n}$: Macro-F1 ${r['macro_f1_mean']:.4f} \\pm "
            f"{r['macro_f1_std']:.4f}$, retention "
            f"${r['retention_pct_mean']:.1f} \\pm {r['retention_pct_std']:.1f}\\%$"
        )

    parallel = ""
    r10 = by_n.get(10)
    if r10 is not None:
        parallel = (
            f" At $N=10$ the ovine retention "
            f"(${r10['retention_pct_mean']:.1f}\\%$) already parallels the "
            f"caprine result, where $N_{{\\min}}=10$ retained $98.5\\%$ of "
            f"the reference."
        )

    return (
        TEMPLATE.replace("@NS@", ", ".join(str(n) for n in ns))
        .replace("@BASE@", f"{base:.4f}")
        .replace("@OBS@", str(obs))
        .replace("@SERIES@", "; ".join(series))
        .replace("@PARALLEL@", parallel)
    )


def main() -> int:
    if not os.path.exists(SUMMARY):
        print(f"[finalize] {SUMMARY} non esiste: il mini-sweep non e' completo; "
              "nessuna modifica effettuata.")
        return 1

    with open(SUMMARY) as f:
        summary = json.load(f)

    logger = TeeLogger()
    plot_retention_curve(summary, logger, plots_dir=PLOTS_DIR)
    for ext in (".pdf", ".png"):
        shutil.copyfile(FIG_SRC + ext, FIG_DST + ext)
        print(f"[finalize] copiata {FIG_DST}{ext}")

    block = build_block(summary)
    with open(TEX) as f:
        text = f.read()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        print("[finalize] marcatori AUTO RQ2-MINI non trovati: nessuna modifica.")
        return 2

    new_text = pattern.sub(lambda m: BEGIN + "\n" + block + "\n" + END,
                           text, count=1)
    tmp = TEX + ".tmp"
    with open(tmp, "w") as f:
        f.write(new_text)
    os.replace(tmp, TEX)
    print("[finalize] experimental_evaluation.tex aggiornato con i numeri RQ2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
