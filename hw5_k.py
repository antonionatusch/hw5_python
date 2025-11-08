# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

ALLOWED = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

k_values   = [8, 16, 24, 32, 48]
lambda_1   = 0.5
lambda_2   = 0.5
n_updates  = 30
seed       = 42
ROUND_PRED = True

def seen_sets(entries):
    su = {i for i,_,_ in entries}
    si = {j for _,j,_ in entries}
    return su, si


def _first_line(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readline()

def _has_header(line):
    toks = [t.strip() for t in line.split(",")]
    def is_num(s):
        try: float(s); return True
        except ValueError: return False
    return any(not is_num(t) for t in toks)

def load_dense_matrix(path):
    line1 = _first_line(path)
    skip = 1 if _has_header(line1) else 0
    arr = np.loadtxt(path, delimiter=",", skiprows=skip)
    if arr.ndim == 1: arr = arr[None, :]
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
    for _ in range(n_updates):
        for i in range(n_users):
            obs = by_user.get(i, [])
            if obs:
                js = [j for (j, _) in obs]
                rs = np.array([r for (_, r) in obs], dtype=float)
                Vobs = V[:, js]
                A = Vobs @ Vobs.T + lambda_u * I
                b = Vobs @ rs
                U[i] = np.linalg.solve(A, b)
        for j in range(n_items):
            obs = by_item.get(j, [])
            if obs:
                is_ = [i for (i, _) in obs]
                rs = np.array([r for (_, r) in obs], dtype=float)
                Uobs = U[is_, :].T
                A = Uobs @ Uobs.T + lambda_v * I
                b = Uobs @ rs
                V[:, j] = np.linalg.solve(A, b)
        if evaluate is not None:
            hist.append(evaluate(U, V, round_pred=round_pred))
    return U, V, hist

Xtr = load_dense_matrix("rate_train.csv")
Xte = load_dense_matrix("rate_test.csv")
assert Xtr.shape == Xte.shape
n_users, n_items = Xtr.shape
train_entries = dense_to_entries(Xtr)
test_entries  = dense_to_entries(Xte)

seen_users, seen_items = seen_sets(train_entries)

def evaluator(U, V, round_pred=False):
    filt = [(i, j, r) for (i, j, r) in test_entries
           if i in seen_users and j in seen_items]
    if not filt:
       return float("nan")
   
    return rmse_on_entries(U, V, filt, round_pred=ROUND_PRED)

errs = []
for k in k_values:
    _, _, hist = als(
        n_users, n_items, train_entries,
        k=k, lambda_u=lambda_1, lambda_v=lambda_2,
        n_updates=n_updates, seed=seed,
        evaluate=evaluator, round_pred=ROUND_PRED
    )
    errs.append(hist[-1])

plt.figure(figsize=(7,4.5))
plt.plot(k_values, errs, marker='o', linewidth=2)
plt.xlabel('Rank k')
plt.ylabel('Testing Error (RMSE)')
plt.title(f'Figure 2 — Test RMSE vs k (λ₁={lambda_1}, λ₂={lambda_2}, T={n_updates}, rounding={ROUND_PRED})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
