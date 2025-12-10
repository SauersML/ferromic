Download https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/aggregated_phenotype_results.tsv

Format:

phenostring	phenocode	chrom	pos	ref	alt	rsids	nearest_genes	consequence	pval	beta	sebeta	af	case_af	control_af	tstat
Mild cognitive impairment	292.2	1	13104656	C	G	rs61149350	HNRNPCL2	non_coding_transcript_exon_variant	0.71	0.056	0.15	0.15	0.16	0.15	2.5
Mild cognitive impairment	292.2	1	25358993	G	A	rs3093638	TMEM50A	intron_variant	0.081	0.18	0.1	0.44	0.48	0.44	17.0
Mild cognitive impairment	292.2	1	108362984	T	G	rs9661538	NBPF6	intergenic_variant	0.89	-0.015	0.11	0.38	0.38	0.38	-1.3
etc.

Download https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/phewas_results.tsv

Format:

Phenotype	N_Total	N_Cases	N_Controls	Beta	OR	P_Value_x	P_Valid	P_Source_x	OR_CI95	CI_Method	CI_Sided	CI_Label	CI_Valid	CI_LO_OR	CI_HI_OR	Used_Ridge	Final_Is_MLE	Used_Firth	Inference_Type	Coef_Source	P_Value_Method	N_Total_Used	N_Cases_Used	N_Controls_Used	Model_Notes	Inversion	P_LRT_Overall	P_Value_y	P_Overall_Valid	P_Source_y	P_Method	P_Value	P_Source	Q_GLOBAL	Sig_Global	CI_Valid_DISPLAY	CI_Method_DISPLAY	CI_Label_DISPLAY	OR_CI95_DISPLAY	CI_LO_OR_DISPLAY	CI_HI_OR_DISPLAY	P_LRT_AncestryxDosage	P_Stage2_Valid	Stage2_P_Method	Stage2_P_Source	Stage2_Inference_Type	LRT_df	LRT_Reason	Stage2_Model_Notes_2	Boot_Engine	Boot_Draws	Boot_Exceed	Boot_Fit_Kind	LRT_Ancestry_Levels	Stage2_Model_Notes	EUR_N	EUR_N_Cases	EUR_N_Controls	EUR_OR	EUR_P	EUR_P_Valid	EUR_P_Source	EUR_Inference_Type	EUR_CI_Method	EUR_CI_Sided	EUR_CI_Label	EUR_CI_Valid	EUR_CI_LO_OR	EUR_CI_HI_OR	EUR_CI95	AFR_N	AFR_N_Cases	AFR_N_Controls	AFR_OR	AFR_P	AFR_P_Valid	AFR_P_Source	AFR_Inference_Type	AFR_CI_Method	AFR_CI_Sided	AFR_CI_Label	AFR_CI_Valid	AFR_CI_LO_OR	AFR_CI_HI_OR	AFR_CI95	AMR_N	AMR_N_Cases	AMR_N_Controls	AMR_OR	AMR_P	AMR_P_Valid	AMR_P_Source	AMR_Inference_Type	AMR_CI_Method	AMR_CI_Sided	AMR_CI_Label	AMR_CI_Valid	AMR_CI_LO_OR	AMR_CI_HI_OR	AMR_CI95	EAS_N	EAS_N_Cases	EAS_N_Controls	EAS_OR	EAS_P	EAS_P_Valid	EAS_P_Source	EAS_Inference_Type	EAS_CI_Method	EAS_CI_Sided	EAS_CI_Label	EAS_CI_Valid	EAS_CI_LO_OR	EAS_CI_HI_OR	EAS_CI95	EAS_REASON	MID_N	MID_N_Cases	MID_N_Controls	MID_OR	MID_P	MID_P_Valid	MID_P_Source	MID_Inference_Type	MID_CI_Method	MID_CI_Sided	MID_CI_Label	MID_CI_Valid	MID_CI_LO_OR	MID_CI_HI_OR	MID_CI95	MID_REASON	SAS_N	SAS_N_Cases	SAS_N_Controls	SAS_OR	SAS_P	SAS_P_Valid	SAS_P_Source	SAS_Inference_Type	SAS_CI_Method	SAS_CI_Sided	SAS_CI_Label	SAS_CI_Valid	SAS_CI_LO_OR	SAS_CI_HI_OR	SAS_CI95	SAS_REASON	EUR_P_FDR	AFR_P_FDR	AMR_P_FDR	EAS_P_FDR	SAS_P_FDR	MID_P_FDR	FINAL_INTERPRETATION
Microcytic_anemia	296698	31702	264996	0.009867789863197	1.009916637040607	0.563501509330062	True	lrt_mle	0.978,1.039	profile	two		True	0.9779845723477241	1.039285357468702	False	True	False	mle	mle	lrt_mle	296698	31702	264996	ridge_seeded_refit;ridge_seeded_refit;inference=mle;ci=profile	chr17-45585160-INV-706887	0.563501509330062	0.563501509330062	True	lrt_mle	lrt_mle			0.9164981520900016	False	True	profile		0.978,1.039	0.9779845723477241	1.039285357468702																																																																																																																		
Post_COVID_19_condition	183678	1545	182133	-0.073952432326626	0.9287158696437521	0.28671879279379003	True	lrt_mle	0.826,1.061	profile	two		True	0.8255996300006551	1.060642284633345	False	True	False	mle	mle	lrt_mle	183678	1545	182133	ridge_seeded_refit;ridge_seeded_refit;inference=mle;ci=profile	chr17-45585160-INV-706887	0.28671879279379003	0.28671879279379003	True	lrt_mle	lrt_mle			0.8084943464739962	False	True	profile		0.826,1.061	0.8255996300006551	1.060642284633345																																																																																																																		
Ventricular_premature_depolarization	218376	8677	209699	-0.004928414297538	0.995083710409501	0.869595795675091	True	lrt_mle	0.943,1.051	profile	two		True	0.943075061531186	1.05059739849461	False	True	False	mle	mle	lrt_mle	218376	8677	209699	ridge_seeded_refit;ridge_seeded_refit;inference=mle;ci=profile	chr17-45585160-INV-706887	0.869595795675091	0.869595795675091	True	lrt_mle	lrt_mle			0.9772511983413777	False	True	profile		0.943,1.051	0.943075061531186	1.05059739849461																																																																																																																		
Other_hearing_abnormality	280405	22853	257552	-0.045994939232777005	0.9550467954480031	0.014854022445291002	True	lrt_mle	0.933,0.991	profile	two		True	0.93330921800174	0.9908455185789231	False	True	False	mle	mle	lrt_mle	280405	22853	257552	ridge_seeded_refit;ridge_seeded_refit;inference=mle;ci=profile	chr17-45585160-INV-706887	0.014854022445291002	0.014854022445291002	True	lrt_mle	lrt_mle			0.29679903759072973	False	True	profile		0.933,0.991	0.93330921800174	0.9908455185789231																																																																																																																		

1. We are matching ONE row in phewas_results to MULTIPLE (or possibly just one) row in phewas_results

2. We use the Phenotype column in phewas_results to look up entries in a mapping that will tell us which phenostring in aggregated_phenotype_results the Phenotype row corresponds to.
3. We use a seperate mapping to tell us which "chrom	pos	ref	alt" key that the Inversion column values in phewas_results correspond to.

Phenotype mappings file:
https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/mappings_final.tsv

Format:
Source_Phenotype	Has_Good_Match	Best_Match	All_Matches	Reasoning	Source_ICD10
Mild cognitive impairment	True	Mild cognitive impairment	Mild cognitive impairment; Memory loss; Delirium dementia and amnestic and other cognitive disorders	The source phenotype 'Mild cognitive impairment' (ICD-9 331.83, ICD-10 G31.84) has an exact name match in the target list. While 'Memory loss' and 'Delirium dementia and amnestic and other cognitive disorders' are related concepts, the verbatim match is the most precise and appropriate mapping.	G31.84
Benign mammary dysplasias	True	Benign mammary dysplasias	Benign mammary dysplasias; Fibroadenosis of breast; Cystic mastopathy; Fibrosclerosis of breast; Other specified benign mammary dysplasias; Other nonmalignant breast conditions	The source phenotype 'Benign mammary dysplasias' (ICD-9 610 series; ICD-10 N60 series) has an exact string match in the UK Biobank candidate list. While several subtypes of this condition (e.g., Fibroadenosis of breast, Cystic mastopathy, Fibrosclerosis of breast) appear in the target list, the parent category 'Benign mammary dysplasias' is the precise semantic and hierarchical equivalent.	N60;N60.0;N60.01;N60.02;N60.09;N60.1;N60.11;N60.12;N60.19;N60.2;N60.21;N60.22;N60.29;N60.3;N60.31;N60.32;N60.39;N60.4;N60.41;N60.42;N60.49;N60.8;N60.81;N60.82;N60.89;N60.9;N60.91;N60.92;N60.99
Other diseases of stomach and duodenum	True	Other disorders of stomach and duodenum	Other disorders of stomach and duodenum; Dyspepsia and other specified disorders of function of stomach; Functional digestive disorders	The source phenotype 'Other diseases of stomach and duodenum' includes ICD-9 codes 536 (Disorders of function of stomach) and 537 (Other disorders of stomach and duodenum) and ICD-10 K31 (Other diseases of stomach and duodenum). The target list contains 'Other disorders of stomach and duodenum', which is a nearly exact string and semantic match for the source name and the K31/537 category. While 'Dyspepsia and other specified disorders of function of stomach' is relevant for the 536 codes, the target 'Other disorders of stomach and duodenum' is the most encompassing and direct match for the source label.	K31;K31.0;K31.1;K31.2;K31.3;K31.6;K31.8;K31.83;K31.84;K31.89;K31.9;K31.A;K31.A0;K31.A1;K31.A11;K31.A12;K31.A13;K31.A14;K31.A15;K31.A19;K31.A2;K31.A21;K31.A22;K31.A29
Abnormal cytological findings in specimens from genital organs	True	Abnormal Papanicolaou smear of cervix and cervical HPV	Abnormal Papanicolaou smear of cervix and cervical HPV; Dysplasia of female genital organs; Cervical intraepithelial neoplasia [CIN] [Cervical dysplasia]; Symptoms involving female genital tract	The source phenotype 'Abnormal cytological findings in specimens from genital organs' is defined by ICD codes (e.g., ICD-9 795.0, ICD-10 R87.61) that specifically refer to abnormal findings on Papanicolaou (Pap) smears and the presence of cervical HPV. The target phenotype 'Abnormal Papanicolaou smear of cervix and cervical HPV' precisely describes this clinical finding and is the most specific and accurate match for the provided definition.	R85.61;R85.610;R85.611;R85.612;R85.613;R85.614;R85.615;R85.616;R85.618;R85.619;R85.81;R85.82;R87.6;R87.61;R87.610;R87.611;R87.612;R87.613;R87.614;R87.615;R87.616;R87.618;R87.619;R87.62;R87.620;R87.621;R87.622;R87.623;R87.624;R87.625;R87.628;R87.629;R87.69;R87.8;R87.81;R87.810;R87.811;R87.82;R87.820;R87.821

To get the phenostring(s) corresponding to a Phenotype value, look up excat match for Phenotype in Source_Phenotype and print warning if Has_Good_Match is not True.
The phenostring value(s) will be in the "All_Matches" column seperated by semicolons as shown. Then, you can use those to look up the potential rows in aggregated_phenotype_results that match our Phenotype.

However, we need to filter down the rows to the ones that match not just the Phenotype, but also the Inversion value for our phewas_results row.

Use https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/best_tagging_snps_qvalues.tsv to accomplish this.
Format:
region	sanitized_region	consensus	p_x	S	REF	ALT	AF	REF_freq_direct	REF_freq_inverted	ALT_freq_direct	ALT_freq_inverted	exclusion_reasons	inversion_region	correlation_r	abs_r	chromosome_hg37	position_hg37	hg37	hg38	position_hg38	q_value
chr1:13104252-13122521	chr1_13104252_13122521	1.0										Low r^2 (<0.5); Selection stats missing for key ('1', 13172124)	1:13104252-13122521	0.460394	0.460394	1	13172124.0	chr1:13172124	chr1:13104656	13104656.0	
chr1:248128773-248428856	chr1_248128773_248428856	0.0										Low haplotype count (<3); Selection stats missing for key ('1', 248342947)	1:248128773-248428856	-1.0	1.0	1	248342947.0	chr1:248342947	chr1:248179645	248179645.0	
chr1:25324280-25369060	chr1_25324280_25369060	0.0	0.040418502284587	0.001697097611209	G	A	0.4577647982723078	0.604938	0.0	0.395062	1.0	Low haplotype count (<3); Low r^2 (<0.5)	1:25324280-25369060	-0.135394	0.135394	1	25685484.0	chr1:25685484	chr1:25358993	25358993.0	
chr1:81650508-81707447	chr1_81650508_81707447	0.0										Low haplotype count (<3); Selection stats missing for key ('1', 82125426)	1:81650508-81707447	-1.0	1.0	1	82125426.0	chr1:82125426	chr1:81659741	81659741.0	
chr10:15742803-15760427	chr10_15742803_15760427	0.0										Low haplotype count (<3); Selection stats missing for key ('10', 15790135)	10:15742803-15760427	-1.0	1.0	10	15790135.0	chr10:15790135	chr10:15748136	15748136.0	
chr10:46983451-47468232	chr10_46983451_47468232	1.0	0.74460575312846	0.0007417309579503	C	T	0.0239275547266123	0.0	0.0	0.0	0.0	Allele mismatch: Selection REF/ALT absent in samples; Low r^2 (<0.5)	10:46983451-47468232	-0.637865	0.637865	10	48568821.0	chr10:48568821	chr10:47170541	47170541.0	


The chr and position in the hg38 column (format chr:pos) corresponds to these two columns in aggregated_phenotype_results.tsv: chrom	pos
Therefore, exact match on that.

THEN, verify that REF	ALT MATCH the  ref	alt columns. Warn if not.
Now, we have found the EXACT ROWS in aggregated_phenotype_results that match a single row in phewas_results.

also, make a note of / save info of which phenotype is "Best_Match"

