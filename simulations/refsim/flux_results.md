# Between-orientation flux sweep — reference classifier

Recurrence is called exactly as in `hsiehphLab/inversionSimulation`:
IQ-TREE ML tree over the full-length haplotype alignment, outgroup
collapsed, Fitch parsimony on the orientation trait, recurrent iff
`minMutHomoplasy >= 2`. The `m=0` column is the upstream model itself.

## single scenario — false-positive rate

**rho = 0e+00**

| depth | m=0e+00 | m=1e-09 | m=1e-08 | m=1e-07 | m=1e-06 |
|---|---|---|---|---|---|
| recent | 0.583 | 0.600 | 0.650 | 0.517 | 0.633 |
| young | 0.233 | 0.283 | 0.300 | 0.233 | 0.300 |
| old | 0.050 | 0.050 | 0.017 | 0.033 | 0.117 |

**rho = 1e-08**

| depth | m=0e+00 | m=1e-09 | m=1e-08 | m=1e-07 | m=1e-06 |
|---|---|---|---|---|---|
| recent | 0.433 | 0.500 | 0.517 | 0.567 | 0.583 |
| young | 0.117 | 0.167 | 0.200 | 0.133 | 0.183 |
| old | 0.000 | 0.017 | 0.000 | 0.017 | 0.050 |

**rho = 1e-06**

| depth | m=0e+00 | m=1e-09 | m=1e-08 | m=1e-07 | m=1e-06 |
|---|---|---|---|---|---|
| recent | 0.000 | 0.017 | 0.017 | 0.017 | 0.000 |
| young | 0.000 | 0.017 | 0.000 | 0.017 | 0.000 |
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
| single | 0.157 | 0.183 | 0.189 | 0.170 | 0.207 |
| recurrent | 0.891 | 0.889 | 0.857 | 0.874 | 0.907 |

## Lowest against highest flux, pooled over all cells

| scenario | rate at m_lo | rate at m_hi | z | p |
|---|---|---|---|---|
| single | 0.1574 (n=540) | 0.2074 (n=540) | 2.13 | 0.0334 |
| recurrent | 0.8907 (n=540) | 0.9074 (n=540) | 0.91 | 0.3633 |
