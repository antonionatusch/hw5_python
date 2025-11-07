# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ------------ config ------------
K_FOLDS      = 5
k_grid       = [10, 20, 30, 40, 50]
lambda1_grid = [0.01, 0.1, 0.5, 1.0, 1.5]
lambda2_grid = [0.01, 0.1, 0.5, 1.0, 1.5]
T_max        = 30
seed         = 42
ROUND_PRED   = True
# --------------------------------

ALLOWED = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

# ===== dense CSV loaders =====
def _first_line(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readline()

def _has_header(line):
    toks = [t.strip() for t in line.split(",")]
    def is_num(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    return any(not is_num(t) for t in toks)

def load_dense_matrix(path):
    """Load users×items CSV; zeros mean missing."""
    line1 = _first_line(path)
    skip = 1 if _has_header(line1) else 0
    arr = np.loadtxt(path, delimiter=",", skiprows=skip)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype(float)

def dense_to_entries(X):
    I, J = np.nonzero(X)
    vals = X[I, J]
    return [(int(i), int(j), float(r)) for i, j, r in zip(I, J, vals)]

def index_by_user_and_item(entries):
    by_user, by_item = defaultdict(list), defaultdict(list)
    for i, j, r in entries:
        by_user[i].append((j, r))
        by_item[j].append((i, r))
    return by_user, by_item

# ===== metrics =====
def rmse_on_entries(U, V, entries, round_pred=False, seen_users=None, seen_items=None):
    errs = []
    for i, j, r in entries:
        if (seen_users is not None and i not in seen_users) or (seen_items is not None and j not in seen_items):
            continue
        pred = float(U[i] @ V[:, j])
        if round_pred:
            pred = float(ALLOWED[np.abs(ALLOWED - pred).argmin()])
        errs.append((pred - r) ** 2)
    return np.sqrt(np.mean(errs)) if errs else np.nan

def seen_sets(entries):
    su = {i for i, _, _ in entries}
    si = {j for _, j, _ in entries}
    return su, si

# ===== ALS (normal-equation updates) =====
def als(n_users, n_items, train_entries, k, lambda_u, lambda_v,
        n_updates, seed, evaluate=None, round_pred=False):
    rng = np.random.default_rng(seed)
    U = 0.1 * rng.standard_normal((n_users, k))
    V = 0.1 * rng.standard_normal((k, n_items))

    by_user, by_item = index_by_user_and_item(train_entries)
    I = np.eye(k)
    history = []

    for _ in range(n_updates):
        # update U
        for i in range(n_users):
            obs = by_user.get(i, [])
            if not obs:
                continue
            js = [j for (j, _) in obs]
            rs = np.array([r for (_, r) in obs], dtype=float)
            Vobs = V[:, js]                     # (k,s)
            A = Vobs @ Vobs.T + lambda_u * I    # (k,k)
            b = Vobs @ rs                       # (k,)
            U[i] = np.linalg.solve(A, b)

        # update V
        for j in range(n_items):
            obs = by_item.get(j, [])
            if not obs:
                continue
            is_ = [i for (i, _) in obs]
            rs = np.array([r for (_, r) in obs], dtype=float)
            Uobs = U[is_, :].T                  # (k,s)
            A = Uobs @ Uobs.T + lambda_v * I    # (k,k)
            b = Uobs @ rs                       # (k,)
            V[:, j] = np.linalg.solve(A, b)

        if evaluate is not None:
            history.append(evaluate(U, V, round_pred=round_pred))

    return U, V, history

# ===== K-fold helpers =====
def kfold_indices(n, K, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    return [list(f) for f in folds]

def complement_indices(n, idx_subset):
    """
    Return indices in [0, n) not in idx_subset.
    Accepts list/ndarray/set; converts to integer array for indexing.
    """
    if not isinstance(idx_subset, np.ndarray):
        idx_subset = np.array(list(idx_subset), dtype=int)
    mask = np.ones(n, dtype=bool)
    if idx_subset.size:
        mask[idx_subset] = False
    return np.flatnonzero(mask).tolist()

# ===== main pipeline =====
Xtr = load_dense_matrix("rate_train.csv")
Xte = load_dense_matrix("rate_test.csv")
assert Xtr.shape == Xte.shape, "Train and test matrices must have identical shape"
n_users, n_items = Xtr.shape

train_entries = dense_to_entries(Xtr)
test_entries  = dense_to_entries(Xte)
seen_users_full, seen_items_full = seen_sets(train_entries)

folds = kfold_indices(len(train_entries), K_FOLDS, seed=seed)

best = dict(score=np.inf, k=None, lam1=None, lam2=None, T=None, curve=None)

for k in k_grid:
    for lam1 in lambda1_grid:
        for lam2 in lambda2_grid:
            fold_curves = []
            for fi, val_idx in enumerate(folds):
                tr_idx = complement_indices(len(train_entries), val_idx)  # <— fixed
                sub_train   = [train_entries[t] for t in tr_idx]
                val_entries = [train_entries[t] for t in val_idx]
                su, si = seen_sets(sub_train)

                def val_eval(U, V, round_pred=False, _su=su, _si=si):
                    return rmse_on_entries(
                        U, V, val_entries, round_pred=ROUND_PRED,
                        seen_users=_su, seen_items=_si
                    )

                _, _, val_hist = als(
                    n_users, n_items, sub_train,
                    k=k, lambda_u=lam1, lambda_v=lam2,
                    n_updates=T_max, seed=seed + fi,
                    evaluate=val_eval, round_pred=ROUND_PRED
                )
                fold_curves.append(np.array(val_hist))

            avg_curve = np.nanmean(np.vstack(fold_curves), axis=0)
            t_star = int(np.nanargmin(avg_curve)) + 1
            best_score = float(np.nanmin(avg_curve))
            if best_score < best["score"]:
                best.update(dict(score=best_score, k=k, lam1=lam1, lam2=lam2,
                                 T=t_star, curve=avg_curve.copy()))

print("Best hyperparameters (via CV):")
print(f"  k = {best['k']}, λ₁ = {best['lam1']}, λ₂ = {best['lam2']}, T* = {best['T']}")
print(f"  CV best RMSE = {best['score']:.4f}")

def test_eval(U, V, round_pred=False):
    return rmse_on_entries(
        U, V, test_entries, round_pred=ROUND_PRED,
        seen_users=seen_users_full, seen_items=seen_items_full
    )

U_best, V_best, test_rmse_hist = als(
    n_users, n_items, train_entries,
    k=best["k"], lambda_u=best["lam1"], lambda_v=best["lam2"],
    n_updates=best["T"], seed=seed + 777,
    evaluate=test_eval, round_pred=ROUND_PRED
)

print(f"Final test RMSE at T*={best['T']}: {test_rmse_hist[-1]:.4f}")

plt.figure(figsize=(7, 4.5))
xs = np.arange(1, len(test_rmse_hist) + 1)
plt.plot(xs, test_rmse_hist, marker='o', linewidth=2, label='ALS (test RMSE)')
plt.xlabel('Number of ALS Updates')
plt.ylabel('Testing Error (RMSE)')
plt.title(f'ALS learning curve — k={best["k"]}, λ₁={best["lam1"]}, λ₂={best["lam2"]}, T*={best["T"]}, rounding={ROUND_PRED}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
