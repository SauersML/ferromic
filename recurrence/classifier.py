"""The recurrence classifier: standardized logistic warm start + partial-AUC-at-low-FPR
refinement, L2-anchored to the warm start. Deterministic.

pAUC refinement: maximize a smooth surrogate of the partial AUC over FPR in [0, FMAX]
(Narasimhan & Agarwal tight-pAUC): for each positive score s_i and each of the
top-ceil(FMAX*n_neg) highest-scoring negatives s_j, add sigmoid((s_i - s_j)/TAU).
The top-FMAX negative set is re-selected from current scores each optimizer step.
Objective minimized:  -pAUC_smooth(w,b) + LAMBDA * ||w - w0||^2 .
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

FMAX = 0.10       # low-FPR region the classifier optimizes
TAU = 0.5         # sigmoid temperature on standardized-score scale
LAMBDA = 0.05     # L2 anchor to the logistic warm start


@dataclass
class Model:
    feature_names: list
    mean: list
    sd: list
    impute: list      # median used for NaN imputation (on raw scale)
    w: list           # weights on standardized features
    b: float
    warm_w: list
    warm_b: float

    def score(self, Xraw):
        X = self._prep(Xraw)
        return expit(X @ np.array(self.w) + self.b)

    def warm_score(self, Xraw):
        X = self._prep(Xraw)
        return expit(X @ np.array(self.warm_w) + self.warm_b)

    def _prep(self, Xraw):
        Xraw = np.asarray(Xraw, dtype=np.float64).copy()
        med = np.array(self.impute)
        idx = np.where(~np.isfinite(Xraw))
        Xraw[idx] = np.take(med, idx[1])
        return (Xraw - np.array(self.mean)) / np.array(self.sd)


def _pauc_smooth_negobj(theta, Xpos, Xneg, w0, fmax=FMAX, tau=TAU, lam=LAMBDA):
    w = theta[:-1]
    b = theta[-1]
    sp = Xpos @ w + b
    sn = Xneg @ w + b
    k = max(1, int(np.ceil(fmax * len(sn))))
    topneg = np.sort(sn)[-k:]                      # hardest (highest-scoring) negatives
    diff = (sp[:, None] - topneg[None, :]) / tau
    pauc = expit(diff).mean()
    anchor = lam * np.sum((w - w0) ** 2)
    return -(pauc) + anchor


def fit(X, y, feature_names, seed=42):
    """X: (n,d) raw features (may contain NaN). y: (n,) 0/1. Returns Model."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    impute = np.nanmedian(X, axis=0)
    Xi = X.copy()
    bad = np.where(~np.isfinite(Xi))
    Xi[bad] = np.take(impute, bad[1])
    mean = Xi.mean(axis=0)
    sd = Xi.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Xi - mean) / sd

    lr = LogisticRegression(C=1.0, max_iter=5000, random_state=seed)
    lr.fit(Z, y)
    w0 = lr.coef_.ravel().copy()
    b0 = float(lr.intercept_[0])

    Zpos, Zneg = Z[y == 1], Z[y == 0]
    theta0 = np.concatenate([w0, [b0]])
    res = minimize(_pauc_smooth_negobj, theta0, args=(Zpos, Zneg, w0),
                   method="L-BFGS-B", options=dict(maxiter=2000, ftol=1e-9))
    w = res.x[:-1]
    b = float(res.x[-1])
    return Model(feature_names=list(feature_names), mean=mean.tolist(), sd=sd.tolist(),
                 impute=impute.tolist(), w=w.tolist(), b=b,
                 warm_w=w0.tolist(), warm_b=b0)


def power_at_fpr(y, s, fmax=FMAX):
    """TPR at the largest threshold with FPR <= fmax."""
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= fmax
    return float(tpr[ok].max()) if ok.any() else 0.0


def pauc_standardized(y, s, fmax=FMAX):
    """McClish standardized partial AUC over [0,fmax] (0.5=random, 1=perfect)."""
    try:
        return float(roc_auc_score(y, s, max_fpr=fmax))
    except ValueError:
        return float("nan")


def evaluate(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    return dict(
        auc=float(roc_auc_score(y, s)),
        pauc_fmax=pauc_standardized(y, s),
        power_at_fpr10=power_at_fpr(y, s),
        brier=float(brier_score_loss(y, np.clip(s, 0, 1))),
        n=int(len(y)), n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
    )
