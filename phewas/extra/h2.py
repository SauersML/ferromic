Need to make .cov file:
FID and IID (must match .fam).
Remaining columns: all covariates
- PCs
-age
- age^2
- NOT ancestry categories since it is one ancestry at a time
- sex


need to make .pheno file:
FID IID  disease_name
id1 id1  1
id2 id2  0
id3 id3  0
...

Need to make .modelSnps file:
Tells BOLT-REML which SNP goes into which variance component.
format:
columns are:
SNP_ID    component_name

We can do the normal SNP IDs.

For inv calls, it cannot stay as an arbitrary float dosage for BOLT-REML. so we have to compute hard-calls. 

this must be one ancestry per BOLT-REML run


liability-scale heritability
