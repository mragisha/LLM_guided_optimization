"""
Main experiment runner.

Usage:
  python run_experiment.py --smoke          # quick test: 2 tasks, 2 seeds, T0+T1
  python run_experiment.py --highdim        # recommended: high-dim tasks only, all treatments
  python run_experiment.py --all            # everything (slow, expensive)
  python run_experiment.py --summarize      # just re-print summary from saved results
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data import load_task, compute_chebyshev, classify_dim, get_all_tasks
from treatments import (t0_ucb_gpm, t1_single_step_llm, t2_iterative_llm,
                        t1b_synthcore, t3_constraint_single,
                        t3_constraint_ensemble, t4_llm_dim_reduction)
from stats import scott_knott

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Treatment registry ────────────────────────────────────────────────────────
# Each entry: (function, budget, uses_llm)
TREATMENTS = {
    "T0_UCB_GPM":            (lambda df,x,y,d,c,s: t0_ucb_gpm(df,x,y,d,c,s, budget=20),           False),
    "T1_SingleStep_LLM":     (lambda df,x,y,d,c,s: t1_single_step_llm(df,x,y,d,c,s, budget=20),    True),
    "T2_Iterative_LLM":      (lambda df,x,y,d,c,s: t2_iterative_llm(df,x,y,d,c,s, rounds=20, k=4), True),
    "T1b_SynthCore":         (lambda df,x,y,d,c,s: t1b_synthcore(df,x,y,d,c,s, M=20, k=4),         True),
    "T3a_Constraint_Single": (lambda df,x,y,d,c,s: t3_constraint_single(df,x,y,d,c,s, budget=20),  True),
    "T3b_Constraint_Ensemble":(lambda df,x,y,d,c,s: t3_constraint_ensemble(df,x,y,d,c,s, M=20,k=4),True),
    "T4_LLM_DimReduction":   (lambda df,x,y,d,c,s: t4_llm_dim_reduction(df,x,y,d,c,s, top_k=5, budget=20), True),
}

# ── Runner ────────────────────────────────────────────────────────────────────

def run(task_filter=None, treatment_names=None, seeds=None, out_file="raw_results.csv"):
    all_tasks = get_all_tasks()
    if task_filter:
        all_tasks = [(n, p) for n, p in all_tasks if task_filter(n, p)]
    if treatment_names is None:
        treatment_names = list(TREATMENTS.keys())
    if seeds is None:
        seeds = list(range(20))

    records = []
    out_path = os.path.join(RESULTS_DIR, out_file)

    # load existing results so we can append without re-running
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        done = set(zip(existing['task'], existing['treatment']))
    else:
        existing = pd.DataFrame()
        done = set()

    for name, path in all_tasks:
        df, x_cols, y_cols, directions = load_task(path)
        cheb = compute_chebyshev(df, y_cols, directions)
        dim = classify_dim(len(x_cols))
        print(f"\n{'='*60}")
        print(f"Task: {name}  |  x={len(x_cols)} ({dim})  |  rows={len(df)}")

        for tname in treatment_names:
            if (name, tname) in done:
                print(f"  [skip] {tname} already done")
                continue

            fn, uses_llm = TREATMENTS[tname]
            scores = []
            for seed in seeds:
                try:
                    score = fn(df, x_cols, y_cols, directions, cheb, seed)
                    scores.append(float(score))
                    print(f"  {tname} seed={seed:02d} -> {score:.4f}")
                except Exception as e:
                    print(f"  {tname} seed={seed:02d} ERROR: {e}")
                    scores.append(float('nan'))

            records.append({
                "task": name, "dim": dim, "n_x": len(x_cols),
                "treatment": tname,
                "scores": json.dumps(scores),
                "median": float(np.nanmedian(scores)),
                "uses_llm": uses_llm,
            })

            # save after every treatment so progress is never lost
            new_df = pd.DataFrame(records)
            combined = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
            combined.to_csv(out_path, index=False)
            print(f"  [saved]")

    if not records:
        print("\nNo new results (all already done).")

    return pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame()


# ── Summarize ─────────────────────────────────────────────────────────────────

def summarize(results_csv="raw_results.csv"):
    path = os.path.join(RESULTS_DIR, results_csv)
    if not os.path.exists(path):
        print("No results file found. Run the experiment first.")
        return

    df = pd.read_csv(path)
    strata = ['low', 'medium', 'high']
    treatments = df['treatment'].unique()
    rows = []

    for stratum in strata:
        sub = df[df['dim'] == stratum]
        tasks = sub['task'].unique()
        total = len(tasks)
        rank0 = {t: 0 for t in treatments}

        for task in tasks:
            task_rows = sub[sub['task'] == task]
            groups = {}
            for _, row in task_rows.iterrows():
                scores = [s for s in json.loads(row['scores']) if not np.isnan(s)]
                if scores:
                    groups[row['treatment']] = scores
            if len(groups) < 2:
                continue
            ranks = scott_knott(groups)
            for t, r in ranks.items():
                if r == 0:
                    rank0[t] += 1

        for t in treatments:
            pct = round(100 * rank0[t] / total, 1) if total > 0 else 0.0
            rows.append({"treatment": t, "stratum": stratum,
                         "rank0_%": pct, "n_tasks": total})

    summary = pd.DataFrame(rows)
    pivot = summary.pivot(index='treatment', columns='stratum', values='rank0_%')
    pivot = pivot.reindex(columns=['low', 'medium', 'high'])
    print("\n=== Rank-0 Frequency (%) — higher is better ===")
    print(pivot.to_string())
    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print("\nSaved to results/summary.csv")
    return pivot


# ── Entry points ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--highdim"

    if mode == "--smoke":
        # Quick sanity check: 2 tasks, T0+T1 only, 3 seeds
        print("Running smoke test...")
        run(
            task_filter=lambda n, p: n in ["SS-A", "SS-B"],
            treatment_names=["T0_UCB_GPM", "T1_SingleStep_LLM"],
            seeds=[0, 1, 2],
            out_file="smoke_results.csv"
        )
        summarize("smoke_results.csv")

    elif mode == "--highdim":
        # Multi-objective high-dim tasks only (paper focus).
        # Skips single-objective tasks (HSMGP, SQL, X264) — not relevant to multi-obj paper.
        def is_relevant(name, path):
            df_tmp = pd.read_csv(path)
            y = [c for c in df_tmp.columns if c.endswith('+') or c.endswith('-')]
            x = [c for c in df_tmp.columns if c not in y]
            return classify_dim(len(x)) == 'high' and len(y) >= 2

        print("Running high-dim multi-objective experiment...")
        print("Tasks: SS-M, SS-N, SS-Q, SS-R, SS-T, SS-U, SS-V, SS-W (8 tasks)")
        print("Seeds: 10  |  Saves after every treatment — safe to stop and resume.\n")
        run(
            task_filter=is_relevant,
            treatment_names=list(TREATMENTS.keys()),
            seeds=list(range(10)),
            out_file="raw_results.csv"
        )
        summarize()

    elif mode == "--all":
        # Full experiment: all tasks, all treatments, 20 seeds
        print("Running FULL experiment (all tasks, all treatments, 20 seeds)...")
        print("WARNING: This will be slow and expensive in API calls.\n")
        run(
            treatment_names=list(TREATMENTS.keys()),
            seeds=list(range(20)),
            out_file="raw_results.csv"
        )
        summarize()

    elif mode == "--summarize":
        summarize()

    else:
        print(__doc__)
