# Stage-2 Rao Score Review - Response and Analysis

## Summary

This document addresses the detailed review of the Stage-2 Rao score test implementation in `phewas/models.py`. After analyzing the codebase, I've identified which concerns are valid, which are based on incorrect assumptions, and what actions are needed.

## Review Points - Detailed Assessment

### 1. **Ancestry Encoding (INVALID CONCERN)**

**Reviewer's Claim:** "You one-hot with `drop_first=True` and include an intercept. That omits one ancestry level by design. The instruction literally says 'every ancestry category.'"

**Reality:**
- Stage-2 code (models.py:4429): `pd.get_dummies(..., drop_first=True)`
- Main PheWAS (run.py:949): `pd.get_dummies(..., drop_first=True)`
- Test code (tests.py:1842, 1909, 2460): All use `drop_first=True`

**Verdict:** ✅ **Implementation is CORRECT and CONSISTENT**. The Stage-2 code matches the main PheWAS pipeline exactly. Using `drop_first=True` with an intercept is the standard approach to avoid perfect collinearity.

### 2. **Age/Sex/PC Column Naming (PARTIALLY VALID)**

**Reviewer's Claim:** "You assume columns named AGE, sex, PC1..PC16. If the upstream loaders return slightly different names or types, the model will break."

**Reality:**
- Stage-2 builds covariates from shared worker data: `models.py:4327-4345`
- Uses same column names as Stage-1: `['const', target, 'sex', 'PC1'...'PCn', 'AGE_c', 'AGE_c_sq']`
- Worker initialization ensures consistency: both stages use same `worker_core_df`

**Verdict:** ⚠️ **MINOR ISSUE**. The covariate assembly is consistent between stages, but there's implicit coupling through global worker state rather than explicit shared helper functions.

### 3. **"Same Inversions as Normally" (VALID CONCERN)**

**Reviewer's Claim:** "Catching LowVarianceInversionError is a start, but the main run may apply additional QC."

**Reality:**
- Stage-2 receives inversions from Stage-1 pipeline after they've passed Stage-1 checks
- Stage-1 in `run.py:1128-1136` catches `LowVarianceInversionError` and tracks skipped inversions
- Stage-2 follow-up is only run for inversions that passed Stage-1: `run.py:1615-1642`

**Verdict:** ✅ **Already Handled Correctly**. Stage-2 only processes inversions that survived Stage-1 QC.

### 4. **"Same Phenotype Restrictions" (VALID OBSERVATION, NO ACTION NEEDED)**

**Reviewer's Claim:** "You impose only MIN_CASES/MIN_CONTROLS thresholds at the category level."

**Reality:**
- Stage-2 operates on phenotypes that already passed Stage-1 filters
- Stage-2 applies its own per-ancestry thresholds: `models.py:4663-4678` uses `validate_min_counts_for_fit`
- This is BY DESIGN - Stage-2 has stricter per-stratum requirements

**Verdict:** ✅ **Working as Intended**. Stage-2 correctly applies additional stratified filters on top of Stage-1 eligibility.

### 5. **Regression Failure Handling (VALID CONCERN - NEEDS IMPROVEMENT)**

**Reviewer's Claim:** "You globally silence convergence warnings, then emit NaNs. That's brittle."

**Reality:**
```python
# models.py:4598-4600
except Exception as e:
    out['LRT_Reason'] = "score_exception"
    out['Stage2_Model_Notes'] = f"rao_score_multi_exception:{type(e).__name__}"
```

**Verdict:** ⚠️ **NEEDS IMPROVEMENT**. The bare `except Exception` is too broad. Should:
1. Catch specific exception types
2. Log more diagnostic info (not just type name)
3. Consider whether some exceptions should propagate

### 6. **Model Fitting Path (INVALID - ALREADY REUSED)**

**Reviewer's Claim:** "You re-implement GLM, coefficient extraction, and CIs."

**Reality:**
- Stage-2 uses same `_fit_logit_ladder` as Stage-1: `models.py:4452, 4454`
- Uses same `_score_test_from_reduced` and `_score_bootstrap_from_reduced` helpers
- Uses same CI functions: `_profile_ci_beta`, `_score_ci_beta`
- Rao score test is a **NEW** capability for multi-df interaction tests

**Verdict:** ✅ **Already Maximally Reused**. The Rao score test is new infrastructure, not reimplementation.

### 7. **Covariate Assembly (VALID - MINOR REFACTOR OPPORTUNITY)**

**Reviewer's Claim:** "You manually create ancestry dummies, center age, and concat pieces. Prefer whatever the main run uses."

**Reality:**
- Stage-2 assembly: `models.py:4428-4437` (in worker function)
- Main run assembly: `run.py:946-949` (in main orchestration)
- Both use identical logic but in different contexts (worker vs main)

**Verdict:** ⚠️ **REFACTOR OPPORTUNITY**. Could extract shared covariate builder, but current approach works correctly.

### 8. **Coefficient Targeting (INVALID CONCERN)**

**Reviewer's Claim:** "You find the inversion coefficient by positional index after adding a constant. That's fragile."

**Reality:**
```python
# models.py:4688 - Uses column name lookup, not position
target_ix_anc = X_anc_zv.columns.get_loc(target)
```

**Verdict:** ✅ **Already Using Name-Based Extraction**. Implementation is correct.

### 9. **Global FDR and Output Schema (VALID SUGGESTIONS)**

**Reviewer's Suggestions:**
- Add Q_Significant boolean column
- Include provenance metadata
- Add ancestry baseline label

**Reality:** Output structure is defined in `testing.py:apply_followup_fdr()` which computes FDR-adjusted p-values.

**Verdict:** ✅ **Good Suggestions for Enhancement** (not bugs).

### 10. **Performance and Scale (NOTED, NO ACTION NEEDED)**

**Reviewer's Claim:** Potential performance issues with set operations and serial execution.

**Reality:**
- Stage-2 worker processes phenotypes in parallel via multiprocessing pool: `pipes.py:746`
- Set operations are in Stage-1 category building, not Stage-2 Rao test
- Stage-2 Rao test operates on already-constructed design matrices

**Verdict:** ✅ **Not Applicable to Stage-2 Rao Implementation**.

## The Rao Score Test - What It Actually Does

The multi-df Rao score test at `models.py:4574-4602` is a **specialized inference method** for testing ancestry×inversion interactions when:
1. Multiple ancestry groups exist (df > 1)
2. Fitting the full interaction model is numerically unstable
3. A score test from the reduced model is more robust

### Implementation Quality Assessment

The Rao score implementation (`_rao_score_block` at lines 2023-2116):

**Strengths:**
- Uses pre-fitted reduced model (can reuse existing fit)
- Robust SVD-based pseudoinverse for stability
- Proper handling of rank-deficient information matrices
- Weight clipping to prevent numerical issues

**Areas for Improvement:**
1. Exception handling in caller is too broad (line 4598)
2. Could add more diagnostic output when test fails

## Recommended Actions

### HIGH PRIORITY (Correctness)

**None**. The implementation is mathematically and structurally sound.

### MEDIUM PRIORITY (Robustness)

1. **Improve exception handling in Rao score caller** (models.py:4598-4600)
   ```python
   # Instead of:
   except Exception as e:
       out['LRT_Reason'] = "score_exception"

   # Use:
   except (LinAlgError, ValueError) as e:
       out['LRT_Reason'] = f"score_linalg_error:{str(e)[:100]}"
       out['Stage2_Model_Notes'] = f"rao_score_failed:{type(e).__name__}"
       # Log full exception for debugging
   except Exception as e:
       # Unexpected error - log and reraise
       logging.error(f"Unexpected error in Rao score test for {s_name}: {e}")
       raise
   ```

### LOW PRIORITY (Enhancement)

1. Extract shared covariate builder function (run.py and models.py)
2. Add Q_Significant column to output as suggested
3. Add more diagnostic fields to Stage2_Model_Notes

## Conclusion

The Stage-2 Rao score implementation is **fundamentally sound and correctly integrated** with the main PheWAS pipeline. Most of the reviewer's concerns are:

- **Invalid** (based on incorrect assumptions about what the code actually does)
- **Already addressed** (the code already does what the reviewer suggests)
- **Not applicable** (concern is about different code, like Stage-1 category building)

The **one valid improvement** is to refine exception handling in the Rao score caller to be more specific and informative.

## Response to Reviewer

The review conflates several different pieces of the PheWAS pipeline:
1. Stage-1 overall testing (lrt_overall_worker)
2. Stage-2 ancestry interaction testing (lrt_followup_worker) - THE ACTUAL RAO CODE
3. Category-level meta-analysis (in phewas/categories.py)

The Stage-2 Rao score test (lines 4574-4602) is a **narrow, specialized inference method** that:
- Operates on data already validated by Stage-1
- Uses the exact same covariate encoding as Stage-1
- Employs shared helper functions for all fitting/inference
- Adds **new** robust score test capability for multi-df interactions

The implementation demonstrates good software engineering:
- Code reuse (same helpers as Stage-1)
- Defensive programming (SVD pseudoinverses, weight clipping)
- Proper error propagation (though exception handling could be more specific)
- Mathematical rigor (correct Rao score statistic formula)

**Net assessment**: Ship it with the minor exception handling improvement if desired, but the core logic is production-ready.
