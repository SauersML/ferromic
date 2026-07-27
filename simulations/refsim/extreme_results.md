# Between-orientation flux sweep — reference classifier

Recurrence is called exactly as in `hsiehphLab/inversionSimulation`:
IQ-TREE ML tree over the full-length haplotype alignment, outgroup
collapsed, Fitch parsimony on the orientation trait, recurrent iff
`minMutHomoplasy >= 2`. The `m=0` column is the upstream model itself.

## single scenario — false-positive rate

**rho = 1e-08**

| depth | m=1e-06 | m=3e-06 | m=1e-05 | m=3e-05 | m=1e-04 |
|---|---|---|---|---|---|
| recent | 0.300 | 0.483 | 0.583 | 0.867 | 1.000 |
| young | 0.083 | 0.183 | 0.400 | 0.733 | 0.967 |

## recurrent scenario — detection rate

**rho = 1e-08**

| depth | m=1e-06 | m=3e-06 | m=1e-05 | m=3e-05 | m=1e-04 |
|---|---|---|---|---|---|
| recent | 0.917 | 0.867 | 0.933 | 0.883 | 0.950 |
| young | 0.883 | 0.900 | 0.833 | 0.917 | 0.933 |

## Marginal over the nine (depth x rho) cells

| scenario | m=1e-06 | m=3e-06 | m=1e-05 | m=3e-05 | m=1e-04 |
|---|---|---|---|---|---|
| single | 0.192 | 0.333 | 0.492 | 0.800 | 0.983 |
| recurrent | 0.900 | 0.883 | 0.883 | 0.900 | 0.942 |

## Lowest against highest flux, pooled over all cells

| scenario | rate at m_lo | rate at m_hi | z | p |
|---|---|---|---|---|
| single | 0.1917 (n=120) | 0.9833 (n=120) | 12.46 | 0.0000 |
| recurrent | 0.9000 (n=120) | 0.9417 (n=120) | 1.20 | 0.2319 |
