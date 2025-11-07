# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------- config you choose -----------------
k          = 20         # rank
lambda_1   = 0.5        # L2 on U
lambda_2   = 0.5        # L2 on V
n_updates  = 30         # ALS sweeps (each = update U then V)
seed       = 42
do_round   = True       # round predictions to {1,1.5,...,5} per tip
# ----------------------------------------------------

ALLOWED = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

def round_to_allowed(x):
    idx = np.abs(ALLOWED[None, :] - x[:, None]).argmin(axis=1)
    return ALLOWED[idx]

def dense_to_entries(X):
    """Convert dense matrix X (0 = missing) → list of (i,j,r)."""
    I, J = np.nonzero(X)           # observed entries only
    return list(zip(I, J, X[I, J].astype(float)))

def index_by_user_and_item(entries):
    by_user, by_item = defaultdict(list), defaultdict(list)
    for i, j, r in entries:
        by_user[i].append((j, r))
        by_item[j].append((i, r))
    return by_user, by_item

def rmse_on_entries(U, V, entries, round_pred=False):
    errs = []
    for i, j, r in entries:
        pred = float(U[i] @ V[:, j])
        if round_pred:
            pred = float(ALLOWED[np.abs(ALLOWED - pred).argmin()])
        errs.append((pred - r) ** 2)
    return np.sqrt(np.mean(errs)) if errs else np.nan

def als(n_users, n_items, train_entries, k, lambda_u, lambda_v,
        n_updates, seed, evaluate=None, round_pred=False):
    rng = np.random.default_rng(seed)
    U = 0.1 * rng.standard_normal((n_users, k))
    V = 0.1 * rng.standard_normal((k, n_items))

    by_user, by_item = index_by_user_and_item(train_entries)
    I = np.eye(k)
    hist = []

    for t in range(n_updates):
        # ----- update U (row-wise normal equations) -----
        for i in range(n_users):
            obs = by_user.get(i, [])
            if not obs:
                continue
            js = [j for (j, _) in obs]
            rs = np.array([r for (_, r) in obs], dtype=float)   # (s,)
            Vobs = V[:, js]                                     # (k,s)
            A = Vobs @ Vobs.T + lambda_u * I                    # (k,k)
            b = Vobs @ rs                                       # (k,)
            U[i] = np.linalg.solve(A, b)

        # ----- update V (column-wise normal equations) -----
        for j in range(n_items):
            obs = by_item.get(j, [])
            if not obs:
                continue
            is_ = [i for (i, _) in obs]
            rs = np.array([r for (_, r) in obs], dtype=float)   # (s,)
            Uobs = U[is_, :].T                                  # (k,s)
            A = Uobs @ Uobs.T + lambda_v * I                    # (k,k)
            b = Uobs @ rs                                       # (k,)
            V[:, j] = np.linalg.solve(A, b)

        if evaluate is not None:
            hist.append(evaluate(U, V, round_pred=round_pred))

    return U, V, hist

# ----------------- load your dense matrices -----------------
# Each CSV is a users×movies matrix; 0 = missing
Xtr = np.loadtxt('rate_train.csv', delimiter=',')
Xte = np.loadtxt('rate_test.csv', delimiter=',')


n_users, n_items = Xtr.shape
train_entries = dense_to_entries(Xtr)
test_entries  = dense_to_entries(Xte)

def evaluator(U, V, round_pred=False):
    return rmse_on_entries(U, V, test_entries, round_pred=round_pred)

# ----------------- run ALS + plot Figure 1 -----------------
U, V, rmse_hist = als(
    n_users, n_items, train_entries,
    k=k, lambda_u=lambda_1, lambda_v=lambda_2,
    n_updates=n_updates, seed=seed,
    evaluate=evaluator, round_pred=do_round
)

plt.figure()
plt.plot(range(1, len(rmse_hist)+1), rmse_hist,
         marker='o', linewidth=2, label='ALS (Test RMSE)')
plt.xlabel('Number of ALS Updates')
plt.ylabel('Testing Error (RMSE)')
plt.title(f'k={k}, λ₁={lambda_1}, λ₂={lambda_2}, rounding={do_round}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
