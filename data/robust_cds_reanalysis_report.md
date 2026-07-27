# Robust reanalysis of CDS sequence identity

## Data structure

- Matched input: **422 orientation rows = 211 paired CDS strata** from **26 inversions**.
- Single-event: **7 inversions / 63 paired CDSs**. Recurrent: **19 inversions / 148 paired CDSs**.
- Rows dropped because the opposite orientation was unavailable: **172**.
- Nominal N implied by `freq_weights=n_pairs`: **486,275**. Independent units: **26**.
- Single-event inverted cells: median haplotype count **2**; **54/63** CDS estimates equal exactly 1.0.

## Results

1. **Original adjusted GLM reproduced:** single-event orientation Wald p = **0.0078**; interaction p = **0.0045**.
2. **Same GLM, exact inversion-block calibration:** two-sided p = **0.1484** (directional 0.0781); unadjusted 0.0547.
3. **Primary paired inversion-level analysis:** single-event mean difference = **8.26 percentage points**, exact two-sided p = **0.0938** (directional 0.0469), 95% t CI **-1.34 to 17.87**.
4. **Recurrent inversions:** **3.36 points**, p = **0.4263**.
5. **Recurrence interaction:** **4.90 points**, exact studentised permutation p = **0.4131**.
6. **After background-diversity adjustment:** residual **-1.33 points**, p = **0.7700**; adding length and haplotype count, **-0.18 points**, p = **0.9639**.
7. **k>=3 sensitivity:** **3** single-event inversions remain, p = **0.5000**.
8. **Nested chr2 loci as one block:** p = **0.1250**.

## Power

Seven single-event blocks, between-inversion SD 0.104, exact sign-flip rule:
power for the observed 8.26-point effect is **39.6% two-sided**
(56.3% directional); about **14.0 points** are needed for 80%.
The exact test cannot return p below 2/128 = 0.0156 at this unit count, whatever the effect size.

## Interpretation

The raw direction is compatible with higher CDS identity among single-event inverted
haplotypes, but the evidence is not robustly two-sided significant once the inversion is
the independent unit, and there is no support for a recurrence-by-orientation
interaction. The observed difference tracks what local background diversity alone
predicts (Pearson r = 0.566, p = 0.0026; Spearman rho = 0.690, p = 9.48e-05),
and the residual CDS-specific effect is null after that adjustment. The defensible
conclusion is descriptive: single-event inverted haplotypes show higher raw CDS pair
identity here, consistent with their lower background diversity rather than with a
CDS-specific conservation or selection effect.
