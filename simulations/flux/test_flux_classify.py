"""Recurrence classifier must be deterministic and order-invariant; simulate exposes fI."""
import numpy as np
from flux_sim import classify, simulate

def test_permutation_invariance():
    G = np.array([[0], [1], [0], [0]]); labels = np.array([1, 1, 0, 0])
    base = classify(G, labels)
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = rng.permutation(len(labels)); assert classify(G[p], labels[p]) == base
    for _ in range(30):
        n = int(rng.integers(4, 14)); G = rng.integers(0, 2, size=(n, int(rng.integers(1, 10))))
        labels = np.array([1] * (n // 2) + [0] * (n - n // 2)); base = classify(G, labels)
        for _ in range(10):
            p = rng.permutation(n); assert classify(G[p], labels[p]) == base

def test_simulate_meta():
    t = dict(t01_23=250000, t0_1=100000, t2_3=50000, t_inv=100000)
    _, _, m = simulate("recurrent", 0.1, 240, 1e-8, 1e-8, 0.0, t, seed=1)
    assert m["fI"] is not None and 0.0 <= m["fI"] <= 1.0
    _, _, m = simulate("single", 0.1, 240, 1e-8, 1e-8, 0.0, t, seed=1)
    assert m["fI"] is None

if __name__ == "__main__":
    test_permutation_invariance(); test_simulate_meta()
    print("OK: classify order-invariant; simulate exposes fI")
