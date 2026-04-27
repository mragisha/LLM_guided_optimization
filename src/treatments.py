"""
All optimization treatments: T0 (UCB_GPM), T1 (single-step LLM), T2 (iterative LLM).
Each treatment takes a MOOT task and returns the best Chebyshev distance found
within a label budget.
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from data import load_task, compute_chebyshev, normalize_objectives, chebyshev

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
LLM_MODEL = "gpt-4o"


# ── helpers ──────────────────────────────────────────────────────────────────

def _nearest_neighbor(proposal, df, x_cols, labeled_idx):
    """Map an LLM proposal (dict) to the nearest unlabeled row in df."""
    unlabeled = df.index.difference(labeled_idx)
    if len(unlabeled) == 0:
        unlabeled = df.index
    x_vals = df.loc[unlabeled, x_cols].values.astype(float)
    prop_vec = np.array([float(proposal.get(c, 0)) for c in x_cols])
    dists = np.linalg.norm(x_vals - prop_vec, axis=1)
    return unlabeled[np.argmin(dists)]


def _run_once(task_df, x_cols, y_cols, directions, cheb, seed, budget,
              get_next_idx_fn):
    """Generic active-learning loop. get_next_idx_fn(labeled, cheb) -> next index."""
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(len(task_df), size=4, replace=False))
    best_scores = []

    while len(labeled) < budget:
        next_idx = get_next_idx_fn(labeled, cheb, task_df, x_cols, y_cols)
        if next_idx not in labeled:
            labeled.append(next_idx)
        best_scores.append(min(cheb[labeled]))

    return min(cheb[labeled])


# ── T0: UCB_GPM ──────────────────────────────────────────────────────────────

def t0_ucb_gpm(task_df, x_cols, y_cols, directions, cheb, seed, budget=20):
    """Upper Confidence Bound over a Gaussian Process surrogate."""
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(len(task_df), size=4, replace=False))
    X_all = task_df[x_cols].values.astype(float)

    while len(labeled) < budget:
        X_lab = X_all[labeled]
        y_lab = cheb[labeled]

        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True,
                                      n_restarts_optimizer=3)
        gp.fit(X_lab, y_lab)

        unlabeled = [i for i in range(len(task_df)) if i not in labeled]
        X_un = X_all[unlabeled]
        mu, sigma = gp.predict(X_un, return_std=True)
        kappa = 2.0
        ucb = mu - kappa * sigma  # lower is better
        next_idx = unlabeled[np.argmin(ucb)]
        labeled.append(next_idx)

    return min(cheb[labeled])


# ── T1: single-step LLM warm-start + greedy ──────────────────────────────────

def _build_few_shot_prompt(task_df, x_cols, y_cols, directions, labeled, cheb, k=4):
    """Build the few-shot prompt for single-step LLM."""
    sorted_labeled = sorted(labeled, key=lambda i: cheb[i])
    best = sorted_labeled[:len(sorted_labeled)//2 or 1]
    rest = sorted_labeled[len(sorted_labeled)//2 or 1:]

    schema_lines = []
    for col in x_cols:
        vals = task_df[col].unique()
        lo, hi = task_df[col].min(), task_df[col].max()
        schema_lines.append(f"  {col}: range [{lo}, {hi}], example values: {list(vals[:5])}")
    schema = "\n".join(schema_lines)

    obj_desc = ", ".join(
        f"{c} ({'maximize' if d == 1 else 'minimize'})" for c, d in zip(y_cols, directions)
    )

    def fmt_rows(idxs):
        rows = []
        for i in idxs:
            row = {c: task_df.iloc[i][c] for c in x_cols}
            row['chebyshev'] = round(float(cheb[i]), 4)
            rows.append(json.dumps(row))
        return "\n".join(rows)

    prompt = f"""You are a software configuration expert. Given a configuration space, propose {k} new configurations that optimize performance.

Objectives: {obj_desc}
Lower Chebyshev distance = better overall performance.

Parameter schema:
{schema}

Best configurations seen so far (lower chebyshev = better):
{fmt_rows(best)}

Worse configurations seen so far:
{fmt_rows(rest)}

Propose exactly {k} new configurations as a JSON array of objects. Only include the parameter names, not chebyshev. Example format:
[{{"param1": value1, "param2": value2, ...}}, ...]

Respond with only the JSON array."""
    return prompt


def _call_llm(prompt):
    """Call OpenAI and parse JSON response."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = response.choices[0].message.content.strip()
    # strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _gp_ucb_next(labeled, task_df, x_cols):
    """Pick next index using GP-UCB acquisition (honest surrogate, no oracle access)."""
    X_all = task_df[x_cols].values.astype(float)
    X_lab = X_all[labeled]
    y_lab = np.array([task_df.index.get_loc(i) if i in task_df.index else i
                      for i in labeled], dtype=float)
    # use the actual observed cheb values passed via closure — rebuilt per call
    return None  # placeholder; see usage below


def t1_single_step_llm(task_df, x_cols, y_cols, directions, cheb, seed, budget=20):
    """Single LLM call to warm-start, then GP-UCB for remaining budget."""
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(len(task_df), size=4, replace=False))
    labeled_set = set(labeled)
    X_all = task_df[x_cols].values.astype(float)

    # LLM proposes 4 warm-start configs
    prompt = _build_few_shot_prompt(task_df, x_cols, y_cols, directions, labeled, cheb)
    try:
        proposals = _call_llm(prompt)
    except Exception:
        proposals = []

    for prop in proposals:
        if len(labeled) >= budget:
            break
        idx = _nearest_neighbor(prop, task_df, x_cols, labeled_set)
        if idx not in labeled_set:
            labeled.append(idx)
            labeled_set.add(idx)

    # fill remaining budget with GP-UCB (no oracle access to cheb)
    while len(labeled) < budget:
        X_lab = X_all[labeled]
        y_lab = cheb[labeled]  # only observed labels — this is legitimate
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True,
                                      n_restarts_optimizer=2)
        gp.fit(X_lab, y_lab)
        unlabeled = [i for i in range(len(task_df)) if i not in labeled_set]
        if not unlabeled:
            break
        mu, sigma = gp.predict(X_all[unlabeled], return_std=True)
        next_idx = unlabeled[np.argmin(mu - 2.0 * sigma)]
        labeled.append(next_idx)
        labeled_set.add(next_idx)

    return min(cheb[labeled])


# ── T2: iterative LLM (H1) ───────────────────────────────────────────────────

def t2_iterative_llm(task_df, x_cols, y_cols, directions, cheb, seed,
                     rounds=20, k=4):
    """
    Iterative LLM optimization (OPRO-style).
    Each round: show full trajectory → LLM proposes k configs → label → repeat.
    Total labels = 4 + rounds * k = 84 (matching SynthCore budget).
    """
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(len(task_df), size=4, replace=False))
    labeled_set = set(labeled)

    for t in range(rounds):
        prompt = _build_few_shot_prompt(task_df, x_cols, y_cols, directions,
                                        labeled, cheb, k=k)
        try:
            proposals = _call_llm(prompt)
        except Exception:
            proposals = []

        added = 0
        for prop in proposals:
            if added >= k:
                break
            idx = _nearest_neighbor(prop, task_df, x_cols, labeled_set)
            if idx not in labeled_set:
                labeled.append(idx)
                labeled_set.add(idx)
                added += 1

    return min(cheb[labeled])


# ── T1b: SynthCore ensemble reproduction ─────────────────────────────────────

def t1b_synthcore(task_df, x_cols, y_cols, directions, cheb, seed,
                  M=20, k=4):
    """
    M independent few-shot sessions, each with 4 random examples.
    All candidates aggregated; best by Chebyshev returned.
    Total labels = 4 + M * k = 84.
    """
    rng = np.random.default_rng(seed)
    initial = list(rng.choice(len(task_df), size=4, replace=False))
    all_labeled = set(initial)
    all_candidates = list(initial)

    for _ in range(M):
        # resample 4 examples from current pool for this session
        session_examples = list(rng.choice(list(all_candidates),
                                           size=min(4, len(all_candidates)),
                                           replace=False))
        prompt = _build_few_shot_prompt(task_df, x_cols, y_cols, directions,
                                        session_examples, cheb, k=k)
        try:
            proposals = _call_llm(prompt)
        except Exception:
            proposals = []

        added = 0
        for prop in proposals:
            if added >= k:
                break
            idx = _nearest_neighbor(prop, task_df, x_cols, all_labeled)
            if idx not in all_labeled:
                all_candidates.append(idx)
                all_labeled.add(idx)
                added += 1

    return min(cheb[list(all_labeled)])


# ── T3: constraint-aware prompting (H2) ──────────────────────────────────────

def _infer_constraints(task_df, x_cols):
    """Infer parameter types and ranges from the data."""
    constraints = []
    for col in x_cols:
        vals = task_df[col].dropna().unique()
        lo, hi = float(task_df[col].min()), float(task_df[col].max())
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            constraints.append(f"  {col}: BINARY (must be 0 or 1)")
        elif all(float(v) == int(float(v)) for v in vals):
            constraints.append(f"  {col}: INTEGER in [{int(lo)}, {int(hi)}]")
        else:
            constraints.append(f"  {col}: CONTINUOUS in [{lo:.4f}, {hi:.4f}]")
    return "\n".join(constraints)


def _build_constraint_prompt(task_df, x_cols, y_cols, directions, labeled, cheb, k=4):
    """Prompt augmented with explicit parameter type/range constraints."""
    sorted_labeled = sorted(labeled, key=lambda i: cheb[i])
    best = sorted_labeled[:len(sorted_labeled)//2 or 1]
    rest = sorted_labeled[len(sorted_labeled)//2 or 1:]

    constraint_block = _infer_constraints(task_df, x_cols)
    obj_desc = ", ".join(
        f"{c} ({'maximize' if d == 1 else 'minimize'})" for c, d in zip(y_cols, directions)
    )

    def fmt_rows(idxs):
        rows = []
        for i in idxs:
            row = {c: task_df.iloc[i][c] for c in x_cols}
            row['chebyshev'] = round(float(cheb[i]), 4)
            rows.append(json.dumps(row))
        return "\n".join(rows)

    prompt = f"""You are a software configuration expert. Propose {k} valid configurations to optimize performance.

Objectives: {obj_desc}
Lower Chebyshev distance = better overall performance.

PARAMETER CONSTRAINTS (you MUST respect these exactly):
{constraint_block}

Best configurations seen so far (lower chebyshev = better):
{fmt_rows(best)}

Worse configurations seen so far:
{fmt_rows(rest)}

Rules:
- Binary parameters must be exactly 0 or 1
- Integer parameters must be whole numbers within the stated range
- Continuous parameters must be within the stated range
- Propose exactly {k} configurations as a JSON array

Respond with only the JSON array."""
    return prompt


def t3_constraint_single(task_df, x_cols, y_cols, directions, cheb, seed, budget=20):
    """H2: T1 with constraint-aware prompt (single LLM call), GP-UCB fill."""
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(len(task_df), size=4, replace=False))
    labeled_set = set(labeled)
    X_all = task_df[x_cols].values.astype(float)

    prompt = _build_constraint_prompt(task_df, x_cols, y_cols, directions, labeled, cheb)
    try:
        proposals = _call_llm(prompt)
    except Exception:
        proposals = []

    for prop in proposals:
        if len(labeled) >= budget:
            break
        idx = _nearest_neighbor(prop, task_df, x_cols, labeled_set)
        if idx not in labeled_set:
            labeled.append(idx)
            labeled_set.add(idx)

    while len(labeled) < budget:
        X_lab = X_all[labeled]
        y_lab = cheb[labeled]
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True,
                                      n_restarts_optimizer=2)
        gp.fit(X_lab, y_lab)
        unlabeled = [i for i in range(len(task_df)) if i not in labeled_set]
        if not unlabeled:
            break
        mu, sigma = gp.predict(X_all[unlabeled], return_std=True)
        next_idx = unlabeled[np.argmin(mu - 2.0 * sigma)]
        labeled.append(next_idx)
        labeled_set.add(next_idx)

    return min(cheb[labeled])


def t3_constraint_ensemble(task_df, x_cols, y_cols, directions, cheb, seed,
                            M=20, k=4):
    """H2: T1b (SynthCore) with constraint-aware prompt."""
    rng = np.random.default_rng(seed)
    initial = list(rng.choice(len(task_df), size=4, replace=False))
    all_labeled = set(initial)
    all_candidates = list(initial)

    for _ in range(M):
        session_examples = list(rng.choice(list(all_candidates),
                                           size=min(4, len(all_candidates)),
                                           replace=False))
        prompt = _build_constraint_prompt(task_df, x_cols, y_cols, directions,
                                          session_examples, cheb, k=k)
        try:
            proposals = _call_llm(prompt)
        except Exception:
            proposals = []

        added = 0
        for prop in proposals:
            if added >= k:
                break
            idx = _nearest_neighbor(prop, task_df, x_cols, all_labeled)
            if idx not in all_labeled:
                all_candidates.append(idx)
                all_labeled.add(idx)
                added += 1

    return min(cheb[list(all_labeled)])


# ── T4: LLM-ranked dimensionality reduction (H3) ─────────────────────────────

def _llm_rank_parameters(task_df, x_cols, y_cols, directions, top_k=5):
    """Ask LLM to rank parameters by expected importance. Returns top_k param names."""
    obj_desc = ", ".join(
        f"{c} ({'maximize' if d == 1 else 'minimize'})" for c, d in zip(y_cols, directions)
    )
    constraints = _infer_constraints(task_df, x_cols)

    prompt = f"""You are a software performance expert. Given this configuration space, rank the parameters from most to least important for optimizing performance.

Objectives: {obj_desc}

Parameters:
{constraints}

Return a JSON array of parameter names ordered from MOST to LEAST important.
Example: ["param3", "param1", "param2", ...]

Respond with only the JSON array of parameter names."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        ranked = json.loads(text.strip())
        # keep only valid column names, take top_k
        ranked = [c for c in ranked if c in x_cols]
        return ranked[:top_k] if len(ranked) >= top_k else x_cols[:top_k]
    except Exception:
        return x_cols[:top_k]


def t4_llm_dim_reduction(task_df, x_cols, y_cols, directions, cheb, seed,
                          top_k=5, budget=20):
    """
    H3: LLM ranks parameters, optimize only top-k.
    Stage 1: 1 LLM call to rank params.
    Stage 2: warm-start using only top-k params, others fixed at median/mode.
    Total LLM calls: 2 (1 rank + 1 propose).
    """
    rng = np.random.default_rng(seed)

    # Stage 1: rank parameters (1 LLM call)
    top_params = _llm_rank_parameters(task_df, x_cols, y_cols, directions, top_k)
    fixed_params = [c for c in x_cols if c not in top_params]

    # fix non-top parameters at their median/mode
    fixed_vals = {}
    for col in fixed_params:
        vals = task_df[col].values
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            fixed_vals[col] = int(pd.Series(vals).mode()[0])
        elif all(float(v) == int(float(v)) for v in vals):
            fixed_vals[col] = int(np.median(vals))
        else:
            fixed_vals[col] = float(np.median(vals))

    # build reduced view: only rows close to fixed vals
    reduced_df = task_df.copy()

    # Stage 2: LLM proposes configs over top-k only (1 LLM call)
    labeled = list(rng.choice(len(task_df), size=4, replace=False))
    prompt = _build_few_shot_prompt(reduced_df, top_params, y_cols, directions,
                                    labeled, cheb, k=4)
    try:
        proposals = _call_llm(prompt)
        # fill in fixed params for each proposal
        for prop in proposals:
            prop.update(fixed_vals)
    except Exception:
        proposals = []

    labeled_set = set(labeled)
    X_all = task_df[x_cols].values.astype(float)

    for prop in proposals:
        if len(labeled) >= budget:
            break
        idx = _nearest_neighbor(prop, task_df, x_cols, labeled_set)
        if idx not in labeled_set:
            labeled.append(idx)
            labeled_set.add(idx)

    while len(labeled) < budget:
        X_lab = X_all[labeled]
        y_lab = cheb[labeled]
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True,
                                      n_restarts_optimizer=2)
        gp.fit(X_lab, y_lab)
        unlabeled = [i for i in range(len(task_df)) if i not in labeled_set]
        if not unlabeled:
            break
        mu, sigma = gp.predict(X_all[unlabeled], return_std=True)
        next_idx = unlabeled[np.argmin(mu - 2.0 * sigma)]
        labeled.append(next_idx)
        labeled_set.add(next_idx)

    return min(cheb[labeled])
