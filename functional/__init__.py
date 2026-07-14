"""ferromic functional-consequence analyses for polymorphic inversions.

Three independent analyses, each in its own submodule, package the code behind the paper's
functional-consequence supplement:

* :mod:`functional.coding`     coding functional scoring (AlphaMissense + ESM C + Evo 2).
* :mod:`functional.splice`     gene-localised AlphaGenome splice-disruption + GTEx validation.
* :mod:`functional.regulatory` measured Geuvadis/GTEx cis eQTL and sQTL by inversion dosage.

Everything is analysis + result tables only; interpretation lives with the study, not here.
All results are associational and haplotype-level (tag-SNP dosage indexes the inversion
haplotype, not the inversion per se).
"""
