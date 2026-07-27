# Between-orientation flux sweep — reference classifier

Recurrence is called exactly as in `hsiehphLab/inversionSimulation`:
IQ-TREE ML tree over the full-length haplotype alignment, outgroup
collapsed, Fitch parsimony on the orientation trait, recurrent iff
`minMutHomoplasy >= 2`. The `m=0` column is the upstream model itself.

## single scenario — false-positive rate

**rho = 0e+00**

| depth | m=0e+00 | m=1e-09 | m=1e-08 | m=1e-07 | m=1e-06 |
|---|---|---|---|---|---|
| recent | 0.533 | 0.517 | 0.550 | 0.483 | 0.483 |
| young | 0.133 | 0.050 | 0.083 | 0.083 | 0.050 |
| old | 0.000 | 0.000 | 0.000 | 0.000 | 0.083 |

**rho = 1e-08**

| depth | m=0e+00 | m=1e-09 | m=1e-08 | m=1e-07 | m=1e-06 |
|---|---|---|---|---|---|
| recent | 0.333 | 0.283 | 0.400 | 0.233 | 0.233 |
| young | 0.017 | 0.067 | 0.000 | 0.033 | 0.083 |
| old | 0.000 | 0.000 | 0.000 | 0.017 | 0.017 |

**rho = 1e-06**

| depth | m=0e+00 | m=1e-09 | m=1e-08 | m=1e-07 | m=1e-06 |
|---|---|---|---|---|---|
| recent | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| young | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| old | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## recurrent scenario — detection rate

**rho = 0e+00**

| depth | m=0e+00 | m=1e-09 | m=1e-08 | m=1e-07 | m=1e-06 |
|---|---|---|---|---|---|
| recent | 0.933 | 0.900 | 0.950 | 0.883 | 0.883 |
| young | 0.867 | 0.917 | 0.833 | 0.867 | 0.850 |
| old | 0.933 | 0.883 | 0.883 | 0.900 | 0.950 |

**rho = 1e-08**

| depth | m=0e+00 | m=1e-09 | m=1e-08 | m=1e-07 | m=1e-06 |
|---|---|---|---|---|---|
| recent | 0.900 | 0.917 | 0.883 | 0.933 | 0.900 |
| young | 0.950 | 0.950 | 0.883 | 0.867 | 0.983 |
| old | 0.833 | 0.783 | 0.833 | 0.850 | 0.900 |

**rho = 1e-06**

| depth | m=0e+00 | m=1e-09 | m=1e-08 | m=1e-07 | m=1e-06 |
|---|---|---|---|---|---|
| recent | 0.917 | 0.917 | 0.883 | 0.867 | 0.867 |
| young | 0.900 | 0.917 | 0.800 | 0.900 | 0.950 |
| old | 0.783 | 0.817 | 0.767 | 0.800 | 0.883 |

## Marginal over the nine (depth x rho) cells

| scenario | m=0e+00 | m=1e-09 | m=1e-08 | m=1e-07 | m=1e-06 |
|---|---|---|---|---|---|
| single | 0.113 | 0.102 | 0.115 | 0.094 | 0.106 |
| recurrent | 0.891 | 0.889 | 0.857 | 0.874 | 0.907 |

## Lowest against highest flux, pooled over all cells

| scenario | rate at m_lo | rate at m_hi | z | p |
|---|---|---|---|---|
| single | 0.1130 (n=540) | 0.1056 (n=540) | -0.39 | 0.6964 |
| recurrent | 0.8907 (n=540) | 0.9074 (n=540) | 0.91 | 0.3633 |
