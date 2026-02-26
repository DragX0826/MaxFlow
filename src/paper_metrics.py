#!/usr/bin/env python3
"""
Generate paper-ready benchmark tables:
1) Main table (SR@2, SR@5, median RMSD, crash rate, fallback rate)
2) Stability table (seed variance + 95% CI)
3) Efficiency table (same-time-budget SR@2)
4) Ranking table (Spearman(logZ, -RMSD) + top-k hit rate)

Usage example:
python src/paper_metrics.py \
  --run fksmc_socm="results/astex10_fksmc_socm_3seed/**/benchmark_results.csv" \
  --run socm="results/astex10_socm_3seed/**/benchmark_results.csv" \
  --exclude_targets 1glh \
  --output_dir results/paper_tables
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ASTEX10_DEFAULT = [
    "1aq1", "1b8o", "1cvu", "1d3p", "1eve",
    "1f0r", "1fc0", "1fpu", "1glh", "1gpk",
]


T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def ci95(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    n = len(vals)
    if n <= 1:
        return float("nan")
    std = float(np.std(vals, ddof=1))
    df = n - 1
    t = T_CRIT_95[df] if df in T_CRIT_95 else 1.96
    return t * std / math.sqrt(n)


def parse_targets(raw: str) -> List[str]:
    if not raw:
        return ASTEX10_DEFAULT
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def infer_seed(path: str) -> int:
    m = re.search(r"seed[_-]?(\d+)", path.lower())
    if m:
        return int(m.group(1))
    return 0


def resolve_csvs(path_pattern: str) -> List[str]:
    matches = sorted(glob.glob(path_pattern, recursive=True))
    out: List[str] = []
    for p in matches:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "**", "benchmark_results.csv"), recursive=True)))
        elif p.lower().endswith(".csv"):
            out.append(p)
    if not out and os.path.isdir(path_pattern):
        out = sorted(glob.glob(os.path.join(path_pattern, "**", "benchmark_results.csv"), recursive=True))
    if not out and os.path.isfile(path_pattern):
        out = [path_pattern]
    return sorted(set(out))


def read_run(method: str, csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "pdb_id" not in df.columns or "best_rmsd" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["method"] = method
    df["csv_path"] = csv_path
    if "seed" not in df.columns:
        df["seed"] = infer_seed(csv_path)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(0).astype(int)
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()

    for col in ["best_rmsd", "time_sec", "log_Z_final", "mmff_fallback_rate"]:
        if col not in df.columns:
            df[col] = np.nan
    df["best_rmsd"] = pd.to_numeric(df["best_rmsd"], errors="coerce")
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce").fillna(0.0)
    df["log_Z_final"] = pd.to_numeric(df["log_Z_final"], errors="coerce")
    df["mmff_fallback_rate"] = pd.to_numeric(df["mmff_fallback_rate"], errors="coerce")
    return df


def summarize_seed_metrics(df_rows: pd.DataFrame, expected_targets: List[str]) -> pd.DataFrame:
    expected = set(expected_targets)
    rows = []
    for (method, seed), g in df_rows.groupby(["method", "seed"], sort=False):
        # Deduplicate target entries (e.g., repeated runs): keep the best RMSD row.
        g2 = (
            g.sort_values(["best_rmsd", "time_sec"], na_position="last")
            .groupby("pdb_id", as_index=False)
            .first()
        )
        g2 = g2[g2["pdb_id"].isin(expected)].copy()
        n_expected = len(expected)
        n_success = g2["pdb_id"].nunique()
        crash_rate = 1.0 - (n_success / n_expected if n_expected > 0 else 0.0)

        rmsd = g2["best_rmsd"].dropna()
        sr2 = float((rmsd < 2.0).mean()) if len(rmsd) else float("nan")
        sr5 = float((rmsd < 5.0).mean()) if len(rmsd) else float("nan")
        med = float(np.median(rmsd)) if len(rmsd) else float("nan")
        mean_r = float(np.mean(rmsd)) if len(rmsd) else float("nan")
        fallback = float(g2["mmff_fallback_rate"].dropna().mean()) if g2["mmff_fallback_rate"].notna().any() else float("nan")
        total_time = float(g2["time_sec"].sum())

        rows.append({
            "method": method,
            "seed": int(seed),
            "n_expected": n_expected,
            "n_success": n_success,
            "crash_rate": crash_rate,
            "sr2": sr2,
            "sr5": sr5,
            "median_rmsd": med,
            "mean_rmsd": mean_r,
            "fallback_rate": fallback,
            "total_time_sec": total_time,
        })
    return pd.DataFrame(rows)


def aggregate_with_ci(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in seed_df.groupby("method", sort=False):
        def m(col: str) -> float:
            return float(g[col].mean()) if col in g else float("nan")

        def v(col: str) -> float:
            vals = g[col].dropna()
            return float(vals.var(ddof=1)) if len(vals) > 1 else float("nan")

        rows.append({
            "method": method,
            "n_seeds": int(g["seed"].nunique()),
            "sr2": m("sr2"),
            "sr2_ci95": ci95(g["sr2"].tolist()),
            "sr5": m("sr5"),
            "sr5_ci95": ci95(g["sr5"].tolist()),
            "median_rmsd": m("median_rmsd"),
            "median_rmsd_ci95": ci95(g["median_rmsd"].tolist()),
            "crash_rate": m("crash_rate"),
            "crash_rate_ci95": ci95(g["crash_rate"].tolist()),
            "fallback_rate": m("fallback_rate"),
            "fallback_rate_ci95": ci95(g["fallback_rate"].tolist()),
            "time_sec": m("total_time_sec"),
            "time_sec_ci95": ci95(g["total_time_sec"].tolist()),
            "sr2_var": v("sr2"),
            "median_rmsd_var": v("median_rmsd"),
            "crash_rate_var": v("crash_rate"),
        })
    return pd.DataFrame(rows)


def efficiency_same_budget(df_rows: pd.DataFrame, expected_targets: List[str]) -> pd.DataFrame:
    order = {t: i for i, t in enumerate(expected_targets)}
    expected = set(expected_targets)

    per_seed_method: Dict[Tuple[int, str], pd.DataFrame] = {}
    for (method, seed), g in df_rows.groupby(["method", "seed"], sort=False):
        g2 = (
            g[g["pdb_id"].isin(expected)]
            .sort_values(["best_rmsd", "time_sec"], na_position="last")
            .groupby("pdb_id", as_index=False)
            .first()
        )
        g2["target_order"] = g2["pdb_id"].map(order).fillna(10**6).astype(int)
        g2 = g2.sort_values("target_order")
        g2["cum_time"] = g2["time_sec"].cumsum()
        per_seed_method[(seed, method)] = g2

    seeds = sorted({k[0] for k in per_seed_method.keys()})
    methods = sorted({k[1] for k in per_seed_method.keys()})
    rows = []
    for seed in seeds:
        totals = []
        for method in methods:
            key = (seed, method)
            if key in per_seed_method:
                totals.append(float(per_seed_method[key]["time_sec"].sum()))
        if not totals:
            continue
        budget = min(totals)
        for method in methods:
            key = (seed, method)
            if key not in per_seed_method:
                continue
            g2 = per_seed_method[key]
            within = g2[g2["cum_time"] <= budget].copy()
            finished = len(within)
            if finished == 0:
                sr2_budget = float("nan")
            else:
                sr2_budget = float((within["best_rmsd"] < 2.0).mean())
            rows.append({
                "method": method,
                "seed": seed,
                "time_budget_sec": budget,
                "finished_targets": finished,
                "coverage": finished / len(expected_targets) if expected_targets else float("nan"),
                "sr2_at_budget": sr2_budget,
            })
    eff_seed = pd.DataFrame(rows)
    if eff_seed.empty:
        return eff_seed
    out = []
    for method, g in eff_seed.groupby("method", sort=False):
        out.append({
            "method": method,
            "n_seeds": int(g["seed"].nunique()),
            "time_budget_sec": float(g["time_budget_sec"].mean()),
            "coverage": float(g["coverage"].mean()),
            "coverage_ci95": ci95(g["coverage"].tolist()),
            "sr2_at_budget": float(g["sr2_at_budget"].mean()),
            "sr2_at_budget_ci95": ci95(g["sr2_at_budget"].tolist()),
        })
    return pd.DataFrame(out)


def ranking_table(df_rows: pd.DataFrame, expected_targets: List[str]) -> pd.DataFrame:
    expected = set(expected_targets)
    rows = []
    for method, g in df_rows.groupby("method", sort=False):
        g = g[g["pdb_id"].isin(expected)].copy()
        g = g[g["log_Z_final"].notna() & g["best_rmsd"].notna()].copy()
        if g.empty:
            rows.append({
                "method": method,
                "n_points": 0,
                "spearman_logZ_vs_negRMSD": float("nan"),
                "n_targets_ranked": 0,
                "top1_hit_rate": float("nan"),
                "top2_hit_rate": float("nan"),
            })
            continue

        # Spearman(logZ, -RMSD): higher is better.
        rx = g["log_Z_final"].rank(method="average")
        ry = (-g["best_rmsd"]).rank(method="average")
        rho = float(rx.corr(ry))

        # Ranking quality across seeds for each target:
        # choose seed with highest logZ, check if it matches best RMSD (top-1 / top-2).
        hit1 = 0
        hit2 = 0
        n_t = 0
        for target, gt in g.groupby("pdb_id"):
            if gt["seed"].nunique() < 2:
                continue
            n_t += 1
            idx_pick = gt["log_Z_final"].idxmax()
            ranks = gt["best_rmsd"].rank(method="min", ascending=True)
            r = float(ranks.loc[idx_pick])
            if r <= 1:
                hit1 += 1
            if r <= 2:
                hit2 += 1
        rows.append({
            "method": method,
            "n_points": int(len(g)),
            "spearman_logZ_vs_negRMSD": rho,
            "n_targets_ranked": n_t,
            "top1_hit_rate": (hit1 / n_t) if n_t else float("nan"),
            "top2_hit_rate": (hit2 / n_t) if n_t else float("nan"),
        })
    return pd.DataFrame(rows)


def write_md(out_path: str, title: str, df: pd.DataFrame):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No data._\n")
            return
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            f.write(df.to_csv(index=False))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Compute paper tables from benchmark_results.csv files")
    parser.add_argument("--run", action="append", required=True,
                        help="Method input in form method_name=path_or_glob_to_csv_or_dir")
    parser.add_argument("--targets", type=str, default=",".join(ASTEX10_DEFAULT),
                        help="Expected targets (comma-separated). Default: Astex-10")
    parser.add_argument("--exclude_targets", type=str, default="1glh",
                        help="Targets to exclude from metrics (comma-separated)")
    parser.add_argument("--output_dir", type=str, default="results/paper_tables")
    args = parser.parse_args()

    expected = parse_targets(args.targets)
    exclude = set(parse_targets(args.exclude_targets))
    expected_eval = [t for t in expected if t not in exclude]
    if not expected_eval:
        raise SystemExit("No evaluation targets left after exclusion.")

    frames = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Invalid --run: {spec}. Expected method=path.")
        method, path_spec = spec.split("=", 1)
        method = method.strip()
        csvs = resolve_csvs(path_spec.strip())
        if not csvs:
            print(f"[WARN] No csv found for method={method}, spec={path_spec}")
            continue
        for csv_path in csvs:
            df = read_run(method, csv_path)
            if not df.empty:
                frames.append(df)
    if not frames:
        raise SystemExit("No valid benchmark_results.csv rows loaded.")

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows[~all_rows["pdb_id"].isin(exclude)].copy()

    seed_metrics = summarize_seed_metrics(all_rows, expected_eval)
    main_tbl = aggregate_with_ci(seed_metrics)[[
        "method", "n_seeds",
        "sr2", "sr2_ci95",
        "sr5", "sr5_ci95",
        "median_rmsd", "median_rmsd_ci95",
        "crash_rate", "crash_rate_ci95",
        "fallback_rate", "fallback_rate_ci95",
    ]].copy()
    stability_tbl = aggregate_with_ci(seed_metrics)[[
        "method", "n_seeds",
        "sr2_var", "sr2_ci95",
        "median_rmsd_var", "median_rmsd_ci95",
        "crash_rate_var", "crash_rate_ci95",
    ]].copy()
    eff_tbl = efficiency_same_budget(all_rows, expected_eval)
    rank_tbl = ranking_table(all_rows, expected_eval)

    os.makedirs(args.output_dir, exist_ok=True)
    seed_metrics.to_csv(os.path.join(args.output_dir, "seed_metrics.csv"), index=False)
    main_tbl.to_csv(os.path.join(args.output_dir, "main_table.csv"), index=False)
    stability_tbl.to_csv(os.path.join(args.output_dir, "stability_table.csv"), index=False)
    eff_tbl.to_csv(os.path.join(args.output_dir, "efficiency_table.csv"), index=False)
    rank_tbl.to_csv(os.path.join(args.output_dir, "ranking_table.csv"), index=False)

    write_md(os.path.join(args.output_dir, "main_table.md"), "Main Table", main_tbl)
    write_md(os.path.join(args.output_dir, "stability_table.md"), "Claim 1 Stability Table", stability_tbl)
    write_md(os.path.join(args.output_dir, "efficiency_table.md"), "Claim 2 Efficiency Table", eff_tbl)
    write_md(os.path.join(args.output_dir, "ranking_table.md"), "Claim 3 Ranking Table", rank_tbl)

    print("\n[Done] tables written to:", args.output_dir)
    print(" -", os.path.join(args.output_dir, "main_table.csv"))
    print(" -", os.path.join(args.output_dir, "stability_table.csv"))
    print(" -", os.path.join(args.output_dir, "efficiency_table.csv"))
    print(" -", os.path.join(args.output_dir, "ranking_table.csv"))
    print("\n[Note] Excluded targets:", ",".join(sorted(exclude)))


if __name__ == "__main__":
    main()
