"""\
src/presentation_results.py

Generate slide-ready results for:
- Resource availability (basic vs advanced)
- Resource selector strategies (Random, Round-Robin, Shortest-Queue)
- Park & Song allocation

Outputs are written to the project-level `outputs/` directory.

Usage (from repo root):
    .venv\\Scripts\\python.exe src\\presentation_results.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Match the palette/labels used by src/generate_analysis_report.py
TUM_BLUE = "#0065BD"
TUM_LIGHT = "#64A0C8"
LIGHT_BG = "#F7F9FC"
DARK_GRAY = "#333333"
MID_GRAY = "#666666"
GREEN = "#27AE60"

CONFIG_COLORS = {
    "r_rma": "#0065BD",
    "r_rra": "#64A0C8",
    "r_shq": "#27AE60",
    "park_song": "#8E44AD",
}
CONFIG_LABELS = {
    "r_rma": "Random",
    "r_rra": "Round-Robin",
    "r_shq": "Shortest Queue",
    "park_song": "Park & Song",
}

# Ensure `src/` is importable when invoked as a script.
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from evaluation import compute_all_metrics  # noqa: E402
from resource_availability_1_5 import ResourceAvailabilityModel  # noqa: E402
from run_simulation import train_if_missing  # noqa: E402
from run_evaluation import _build_engine, SIM_START, SIM_END  # noqa: E402


@dataclass(frozen=True)
class AvailabilitySummary:
    mode: str
    sample_freq: str
    n_samples: int
    mean_available: float
    p10_available: float
    median_available: float
    p90_available: float


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _data_paths(project_root: Path) -> Tuple[Path, Path]:
    data_dir = project_root / "data"
    csv_path = data_dir / "bpi2017.csv"
    bpmn_path = data_dir / "Signavio_Model.bpmn"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing data file: {csv_path}")
    if not bpmn_path.exists():
        raise FileNotFoundError(f"Missing BPMN model: {bpmn_path}")

    return csv_path, bpmn_path


def _outputs_dir(project_root: Path) -> Path:
    out = project_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sample_times_utc(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")
    if t1.tzinfo is None:
        t1 = t1.tz_localize("UTC")
    else:
        t1 = t1.tz_convert("UTC")

    # inclusive="left" ensures we don't include the end boundary.
    return pd.date_range(t0, t1, freq=freq, inclusive="left")


def export_availability_results(
    *,
    project_root: Path,
    csv_path: Path,
    mode: str,
    sample_freq: str,
    out_prefix: str,
) -> AvailabilitySummary:
    """Export time-series + heatmap-friendly aggregates for availability."""

    outputs_dir = _outputs_dir(project_root)

    availability_cache = str(project_root / "models" / f"availability_{mode}_1_5.pkl")
    model = ResourceAvailabilityModel(
        csv_path=str(csv_path),
        mode=mode,
        cache_path=availability_cache,
    )

    times = _sample_times_utc(SIM_START, SIM_END, sample_freq)
    counts: List[int] = []
    wd_list: List[int] = []
    hr_bucket_list: List[int] = []

    for ts in times:
        avail = model.get_available_resources(ts)
        counts.append(int(len(avail)))
        wd_list.append(int(ts.weekday()))
        if mode == "basic":
            hr_bucket_list.append(int(ts.hour))
        else:
            hr_bucket_list.append(int(ts.hour // 2))

    ts_df = pd.DataFrame(
        {
            "time_utc": times,
            "available_resources": counts,
            "weekday": wd_list,
            "hour_bucket": hr_bucket_list,
        }
    )

    ts_csv = outputs_dir / f"{out_prefix}_{mode}_timeseries.csv"
    ts_df.to_csv(ts_csv, index=False)

    # Heatmap-friendly aggregation: mean availability per weekday × hour bucket.
    heat_df = (
        ts_df.groupby(["weekday", "hour_bucket"], observed=True)["available_resources"]
        .mean()
        .rename("mean_available")
        .reset_index()
        .sort_values(["weekday", "hour_bucket"], ascending=[True, True])
    )
    heat_csv = outputs_dir / f"{out_prefix}_{mode}_heatmap.csv"
    heat_df.to_csv(heat_csv, index=False)

    s = pd.Series(counts, dtype=float)
    summary = AvailabilitySummary(
        mode=mode,
        sample_freq=sample_freq,
        n_samples=int(len(s)),
        mean_available=float(s.mean()) if len(s) else float("nan"),
        p10_available=float(s.quantile(0.10)) if len(s) else float("nan"),
        median_available=float(s.quantile(0.50)) if len(s) else float("nan"),
        p90_available=float(s.quantile(0.90)) if len(s) else float("nan"),
    )

    return summary


def _run_or_reuse_simulation(
    *,
    project_root: Path,
    outputs_dir: Path,
    config: str,
    mode: str,
    out_csv_name: str,
) -> Path:
    # If the standard evaluation log already exists for advanced mode, reuse it.
    # This avoids re-running long simulations (especially Park & Song).
    if mode == "advanced":
        standard_eval = outputs_dir / f"eval_{config}.csv"
        if standard_eval.exists():
            print(f"  [USE ] {standard_eval.name} (existing)")
            return standard_eval

    out_csv = outputs_dir / out_csv_name
    if out_csv.exists():
        print(f"  [SKIP] {out_csv.name} already exists")
        return out_csv

    print(f"  [RUN ] {config:<10} mode={mode:<8} -> {out_csv.name}")
    engine = _build_engine(
        project_root=project_root,
        config=config,
        out_csv=str(out_csv),
        mode=mode,
    )
    engine.run()
    return out_csv


def export_allocation_metrics(
    *,
    project_root: Path,
    configs: Iterable[str],
    mode: str,
    out_csv_name: str,
    log_prefix: str,
) -> Path:
    """Run/Reuse simulations and export a single metrics table as CSV."""

    outputs_dir = _outputs_dir(project_root)

    t_start = pd.Timestamp(SIM_START)
    t_end = pd.Timestamp(SIM_END)

    rows: List[Dict[str, object]] = []

    for config in configs:
        log_path = _run_or_reuse_simulation(
            project_root=project_root,
            outputs_dir=outputs_dir,
            config=config,
            mode=mode,
            out_csv_name=f"{log_prefix}_{mode}_{config}_2016.csv",
        )

        metrics = compute_all_metrics(str(log_path), sim_start=t_start, sim_end=t_end)
        row = {"mode": mode, "config": config}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = outputs_dir / out_csv_name
    df.to_csv(out_path, index=False)
    return out_path


def _make_availability_chart(
    *,
    summary_csv: Path,
    out_path: Path,
) -> None:
    df = pd.read_csv(summary_csv)
    # Expected columns from AvailabilitySummary.__dict__
    # mode, sample_freq, n_samples, mean_available, p10_available, median_available, p90_available
    order = ["basic", "advanced"]
    df["mode"] = df["mode"].astype(str).str.lower()
    df["_order"] = df["mode"].apply(lambda m: order.index(m) if m in order else 999)
    df = df.sort_values(["_order", "mode"]).drop(columns=["_order"])

    modes = df["mode"].astype(str).tolist()
    median = df["median_available"].astype(float).to_numpy()
    p10 = df["p10_available"].astype(float).to_numpy()
    p90 = df["p90_available"].astype(float).to_numpy()
    mean = df["mean_available"].astype(float).to_numpy()

    yerr = [median - p10, p90 - median]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=220)
    x = list(range(len(modes)))

    ax.set_facecolor(LIGHT_BG)
    fig.patch.set_facecolor("white")

    ax.bar(x, median, color=TUM_BLUE, alpha=0.75, edgecolor="white", linewidth=0.8, label="Median")
    ax.errorbar(x, median, yerr=yerr, fmt="none", ecolor=DARK_GRAY, elinewidth=1.4, capsize=7, label="P10–P90")
    ax.scatter(x, mean, color=TUM_LIGHT, s=35, zorder=3, label="Mean")

    ax.set_xticks(x, [m.capitalize() for m in modes])
    ax.set_ylabel("Available resources", color=MID_GRAY)
    ax.set_title("Resource Availability (Basic vs Advanced) — 2016", fontsize=14, fontweight="bold", color=DARK_GRAY)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _make_cycle_time_chart(
    *,
    metrics_csv: Path,
    out_path: Path,
) -> None:
    df = pd.read_csv(metrics_csv)
    # focus on allocation comparison: avg cycle time
    df["config"] = df["config"].astype(str)
    df = df.sort_values("avg_cycle_time_h", ascending=True)
    configs = df["config"].tolist()
    labels = [CONFIG_LABELS.get(c, c) for c in configs]
    values = df["avg_cycle_time_h"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=220)
    ax.set_facecolor(LIGHT_BG)
    fig.patch.set_facecolor("white")

    bars = ax.barh(labels, values, color=MID_GRAY, alpha=0.88, edgecolor="white", linewidth=0.8)
    for bar, cfg in zip(bars, configs):
        bar.set_facecolor(CONFIG_COLORS.get(cfg, MID_GRAY))
    ax.invert_yaxis()

    # highlight the best (lowest avg cycle time)
    if len(bars) > 0:
        bars[0].set_edgecolor(GREEN)
        bars[0].set_linewidth(2.4)

    ax.set_xlabel("Avg cycle time (hours) — lower is better", color=MID_GRAY)
    ax.set_title("Allocation Methods — Avg Cycle Time (2016, advanced)", fontsize=14, fontweight="bold", color=DARK_GRAY)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # value labels
    vmax = float(values.max()) if len(values) else 1.0
    for y, v in enumerate(values):
        ax.text(v + vmax * 0.01, y, f"{v:.2f}h", va="center", fontsize=10, color=DARK_GRAY)

    # legend (method colors)
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=CONFIG_COLORS.get(cfg, MID_GRAY), edgecolor="white", label=CONFIG_LABELS.get(cfg, cfg))
        for cfg in ["r_rma", "r_rra", "r_shq", "park_song"]
        if cfg in set(configs)
    ]
    if legend_items:
        ax.legend(handles=legend_items, frameon=False, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    project_root = _project_root()
    csv_path, _bpmn_path = _data_paths(project_root)

    print("[SETUP] Ensuring trained models exist (1.3 + 1.4)…")
    train_if_missing(project_root, csv_path)

    outputs_dir = _outputs_dir(project_root)

    print("\n[AVAILABILITY] Exporting basic vs advanced availability summaries…")
    # Basic: hourly buckets; Advanced: 2-hour buckets. Sampling at 1H keeps comparisons simple.
    avail_summaries = []
    for mode in ("basic", "advanced"):
        summary = export_availability_results(
            project_root=project_root,
            csv_path=csv_path,
            mode=mode,
            sample_freq="1h",
            out_prefix="presentation_availability",
        )
        avail_summaries.append(summary.__dict__)
        print(
            f"  [OK ] {mode:<8} mean={summary.mean_available:.2f} "
            f"(p10={summary.p10_available:.0f}, p50={summary.median_available:.0f}, p90={summary.p90_available:.0f})"
        )

    avail_summary_path = outputs_dir / "presentation_availability_summary.csv"
    pd.DataFrame(avail_summaries).to_csv(avail_summary_path, index=False)

    print("\n[ALLOCATION] Exporting selector + Park&Song metrics (advanced availability)…")
    metrics_path = export_allocation_metrics(
        project_root=project_root,
        configs=["r_rma", "r_rra", "r_shq", "park_song"],
        mode="advanced",
        out_csv_name="presentation_metrics_advanced.csv",
        log_prefix="presentation_eval",
    )

    print("\n[CHARTS] Rendering slide charts (PNG)…")
    availability_png = outputs_dir / "presentation_chart_availability_basic_vs_advanced.png"
    cycle_time_png = outputs_dir / "presentation_chart_avg_cycle_time.png"
    _make_availability_chart(summary_csv=avail_summary_path, out_path=availability_png)
    _make_cycle_time_chart(metrics_csv=metrics_path, out_path=cycle_time_png)

    print("\n[DONE] Wrote:")
    print(f"  - {avail_summary_path}")
    print(f"  - {metrics_path}")
    print(f"  - {availability_png}")
    print(f"  - {cycle_time_png}")
    print("  - outputs/presentation_availability_{basic,advanced}_timeseries.csv")
    print("  - outputs/presentation_availability_{basic,advanced}_heatmap.csv")
    print("  - outputs/presentation_eval_advanced_{config}_2016.csv (logs)\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
