"""Regression test: the recurrence classifier must be permutation-invariant (audit #4)."""
import numpy as np
from flux_sim import classify

def test_permutation_invariance_minimal():
    # Audit repro: previously classify(G,labels)=1 but classify(G[p],labels[p])=2.
    G = np.array([[0], [1], [0], [0]])
    labels = np.array([1, 1, 0, 0])
    base = classify(G, labels)
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.permutation(len(labels))
        assert classify(G[p], labels[p]) == base, "classify is order-dependent"
    return base

def test_permutation_invariance_random():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = rng.integers(4, 12)
        G = rng.integers(0, 2, size=(n, rng.integers(1, 8)))
        labels = np.array([1] * (n // 2) + [0] * (n - n // 2))
        base = classify(G, labels)
        for _ in range(10):
            p = rng.permutation(n)
            assert classify(G[p], labels[p]) == base

if __name__ == "__main__":
    b = test_permutation_invariance_minimal()
    test_permutation_invariance_random()
    print(f"OK: permutation-invariant (minimal repro stable score={b})")
