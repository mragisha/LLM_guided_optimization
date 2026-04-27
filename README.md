# Rescuing LLM-Guided Optimization on High-Dimensional Software Configuration Tasks

---

## Overview

Software configuration optimization often requires balancing multiple objectives (throughput, latency, energy) under a tight labeling budget, where each evaluation can take hours of compute. Prior work showed that LLM warm-starts help on low- and medium-dimensional tasks but **fail on high-dimensional problems** (`|x| > 11`), where a plain Gaussian Process baseline regains the lead.

This project tests three rescue mechanisms on 8 high-dimensional tasks from the [MOOT benchmark](https://arxiv.org/abs/2511.16882):

| Hypothesis | Treatment | Idea |
|---|---|---|
| H1 | T2 — Iterative LLM | Re-prompt the LLM each round with the full evaluated trajectory (OPRO-style) |
| H2 | T3a/T3b — Constraint-Aware | Encode parameter types and ranges explicitly in the prompt |
| H3 | T4 — LLM DimReduction | Let the LLM rank parameters by importance; optimize only the top-k |

All are compared against a GP-UCB baseline (T0), a single-step LLM warm-start (T1), and an in-house SynthCore reproduction (T1b).

---

## Key Results

| Treatment | Rank-0 Frequency | Description |
|---|:---:|---|
| **T3b Constraint Ensemble** | **100%** | Never fell out of the best group across all 8 tasks |
| T2 Iterative LLM | 87.5% | Best point-estimate on 5/8 tasks; up to 30% improvement over baseline |
| T1b SynthCore | 87.5% | Strong on high-dimensional tight-cluster tasks |
| T3a Constraint Single | 87.5% | |
| T0 UCB-GP (baseline) | 75.0% | Best on binary/integer-heavy tasks (SS-V, SS-W) |
| T1 SingleStep LLM | 75.0% | Reproduces the known high-dim failure |
| T4 LLM DimReduction | 75.0% | |

**Headline finding:** Constraint-aware ensemble prompting (T3b) is the most robust treatment. LLM methods significantly *hurt* performance on tasks dominated by binary/integer parameters (SS-V, SS-W), where nearest-neighbor snapping of continuous proposals injects noise.

---

## Repository Structure

```
.
├── src/
│   ├── run_experiment.py   # Main experiment runner
│   ├── treatments.py       # All 7 treatment implementations
│   ├── data.py             # MOOT data loading + Chebyshev metric
│   └── stats.py            # Scott-Knott + Cliff's Delta
├── moot/                   # MOOT benchmark tasks (submodule)
│   └── optimize/config/    # SS-* CSV files used in this study
├── results/
│   ├── raw_results.csv     # Per-seed Chebyshev scores for all runs
│   ├── summary.csv         # Scott-Knott rank-0 frequency per treatment
│   └── smoke_results.csv   # Quick sanity-check run output
├── report_final.tex        # Complete paper (LaTeX, ACM sigconf)
├── refs.bib                # Bibliography
└── .env                    # API keys (not committed)
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn openai python-dotenv
```

### 3. Set your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

---

## Running the Experiment

### Full experiment (8 high-dim tasks, all 7 treatments, 10 seeds)

```bash
python src/run_experiment.py --highdim
```

Results are saved to `results/raw_results.csv` after every treatment. The run is **safe to stop and resume** — already-completed (task, treatment) pairs are skipped automatically.

### Quick smoke test (2 tasks, T0 + T1 only, 3 seeds)

```bash
python src/run_experiment.py --smoke
```

### Re-print the rank-0 frequency summary from saved results

```bash
python src/run_experiment.py --summarize
```

### Full experiment (all tasks, 20 seeds) — slow and expensive

```bash
python src/run_experiment.py --all
```

---

## Treatments

| ID | Name | Labels | LLM Calls | Description |
|---|---|:---:|:---:|---|
| T0 | UCB-GP | 20 | 0 | Gaussian Process + UCB acquisition (baseline) |
| T1 | SingleStep LLM | 20 | 1 | Single LLM call to warm-start, then GP-UCB |
| T2 | Iterative LLM | 84 | 20 | OPRO-style: LLM re-prompted each round with trajectory |
| T1b | SynthCore | 84 | 20 | 20 independent few-shot sessions, best result returned |
| T3a | Constraint Single | 20 | 1 | T1 with explicit parameter-type constraints in prompt |
| T3b | Constraint Ensemble | 84 | 20 | T1b with constraint-aware prompt |
| T4 | LLM DimReduction | 20 | 2 | LLM ranks parameters; optimize top-5 only |

> **Budget note:** T2, T1b, T3b use 84 labels (matched LLM-call protocol). T0, T1, T3a, T4 use 20 labels (matched label protocol). Cross-group comparisons are label-asymmetric; see the paper for discussion.

---

## Evaluation Metric

All objectives are normalized to [0, 1] with direction flipped so 0 = optimal. The **Chebyshev distance** from the utopia point is:

```
δ(y) = max_j  y_j_normalized       (lower is better)
```

Statistical ranking uses **Scott-Knott** recursive bi-clustering with a 1000-resample bootstrap test (α = 0.05) and Cliff's Delta effect size. **Rank-0 frequency** (% of tasks in the best statistical group) is the primary cross-task metric.

---

## Data

Tasks come from the [MOOT repository](https://github.com/timm/moot) — specifically the 8 Specific Software Configuration (SS-*) tasks in the high-dimensional stratum (`|x| > 11`):

| Task | Features | Objectives |
|---|:---:|:---:|
| SS-M | 17 | 2 |
| SS-N | 17 | 2 |
| SS-Q | 13 | 2 |
| SS-R | 14 | 2 |
| SS-T | 12 | 2 |
| SS-U | 21 | 3 |
| SS-V | 16 | 2 |
| SS-W | 16 | 2 |

---


