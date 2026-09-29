# Live trial graph profile

## 1 · Labels

| Label | Nodes |
|---|---|
| CRO | 33 |
| Chunk | 5,764 |
| Country | 50 |
| Disease | 77 |
| Document | 20 |
| Drug | 0 |
| MeSHTerm | 91 |
| Outcome | 449 |
| PatientPopulation | 20 |
| Section | 464 |
| Site | 1,085 |
| Sponsor | 18 |
| Trial | 20 |
| TrialCategory | 20 |

## 2 · Properties per label

### CRO  (33 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `key` | 33 / 33 | str | 33 distinct · e.g. emd serono; cancer research institute new york city; boehringer ingelheim; medimmune |
| `name` | 33 / 33 | str | 33 distinct · e.g. EMD Serono; Cancer Research Institute, New York City; Boehringer Ingelheim; MedImmune LLC |
| `source` | 33 / 33 | str | registry ×33 |

### Chunk  (5,764 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `chunkId` | 5,764 / 5,764 | str | 5,764 distinct · e.g. nct02788279-cobimetinib-atezolizumab-go30182:cce53c9d320e3985:0; nct02788279-cobimetinib-atezolizumab-go30182:ff067e4686894593:0; nct02788279-cobimetinib-atezolizumab-go30182:33a450130f9cac85:0; nct02788279-cobimetinib-atezolizumab-go30182:f243e947fd02d477:0 |
| `content_type` | 5,764 / 5,764 | str | text ×3702; table ×1034; table_summary ×873; figure ×147; formula ×8 |
| `docId` | 5,764 / 5,764 | str | nct04614948-covid-ad26-janssen ×588; nct04368728-covid-bnt162-pfizer ×466; nct02951156-prostate-cancer ×454; nct02788279-cobimetinib-atezolizumab-go30182 ×398; nct03374254-colon-cancer ×395; nct03434379-hepatocellular-atezo-bev ×389; nct04470427-covid-mrna1273-moderna ×363; nct03548935-obesity-semaglutide ×349; nct04032704-solid-tumors-ladiratuzumab ×318; nct03662659-gastric-cancer-relatlimab ×314; nct02863419-t2d-oral-semaglutide-pioneer4 ×286; nct03164772-nsclc-mrna-vaccine ×258; nct03155620-parkinsons-study ×236; nct03961204-classic-ms ×179; nct04652245-allergic-rhinitis-dymista ×179; nct03181503-prurigo-nodularis-nemolizumab ×172; nct03235752-ulcerative-colitis ×145; nct04280705-covid-actt-remdesivir ×127; nct03753074-hepatitis-b-taf ×84; nct02014597-glaucoma-optokinetic ×64 |
| `n_tokens` | 5,764 / 5,764 | int | 829 distinct · min 8 · median 315.0 · max 1024 · e.g. 1024; 336; 386; 223 |
| `origin` | 5,764 / 5,764 | str | structure ×5764 |
| `page` | 5,764 / 5,764 | int | 249 distinct · min 1 · median 75.0 · max 250 · e.g. 20; 27; 28; 39 |
| `position` | 5,764 / 5,764 | int | 588 distinct · min 0 · median 153.0 · max 587 · e.g. 0; 1; 2; 3 |

### Country  (50 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `key` | 50 / 50 | str | 50 distinct · e.g. united states; australia; belgium; canada |
| `name` | 50 / 50 | str | 50 distinct · e.g. United States; Australia; Belgium; Canada |
| `source` | 50 / 50 | str | registry ×50 |

### Disease  (77 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `key` | 77 / 77 | str | 77 distinct · e.g. glaucoma; colorectal cancer; diabetes; diabetes mellitus type 2 |
| `name` | 77 / 77 | str | 77 distinct · e.g. Glaucoma; Colorectal Cancer; Diabetes; Diabetes Mellitus, Type 2 |
| `source` | 77 / 77 | str | registry ×77 |

### Document  (20 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `docId` | 20 / 20 | str | nct02014597-glaucoma-optokinetic ×1; nct02788279-cobimetinib-atezolizumab-go30182 ×1; nct02863419-t2d-oral-semaglutide-pioneer4 ×1; nct02951156-prostate-cancer ×1; nct03155620-parkinsons-study ×1; nct03164772-nsclc-mrna-vaccine ×1; nct03181503-prurigo-nodularis-nemolizumab ×1; nct03235752-ulcerative-colitis ×1; nct03374254-colon-cancer ×1; nct03434379-hepatocellular-atezo-bev ×1; nct03548935-obesity-semaglutide ×1; nct03662659-gastric-cancer-relatlimab ×1; nct03753074-hepatitis-b-taf ×1; nct03961204-classic-ms ×1; nct04032704-solid-tumors-ladiratuzumab ×1; nct04280705-covid-actt-remdesivir ×1; nct04368728-covid-bnt162-pfizer ×1; nct04470427-covid-mrna1273-moderna ×1; nct04614948-covid-ad26-janssen ×1; nct04652245-allergic-rhinitis-dymista ×1 |
| `nChunks` | 20 / 20 | int | 179 ×2; 64 ×1; 398 ×1; 286 ×1; 454 ×1; 236 ×1; 258 ×1; 172 ×1; 145 ×1; 395 ×1; 389 ×1; 349 ×1; 314 ×1; 84 ×1; 318 ×1; 127 ×1; 466 ×1; 363 ×1; 588 ×1 |
| `nctId` | 20 / 20 | str | NCT02014597 ×1; NCT02788279 ×1; NCT02863419 ×1; NCT02951156 ×1; NCT03155620 ×1; NCT03164772 ×1; NCT03181503 ×1; NCT03235752 ×1; NCT03374254 ×1; NCT03434379 ×1; NCT03548935 ×1; NCT03662659 ×1; NCT03753074 ×1; NCT03961204 ×1; NCT04032704 ×1; NCT04280705 ×1; NCT04368728 ×1; NCT04470427 ×1; NCT04614948 ×1; NCT04652245 ×1 |
| `origin` | 20 / 20 | str | structure ×20 |
| `sourceFile` | 20 / 20 | str | NCT02014597_Glaucoma_Optokinetic.pdf ×1; NCT02788279_Cobimetinib_Atezolizumab_GO30182.pdf ×1; NCT02863419_T2D_Oral_Semaglutide_PIONEER4.pdf ×1; NCT02951156_Prostate_Cancer.pdf ×1; NCT03155620_Parkinsons_Study.pdf ×1; NCT03164772_NSCLC_mRNA_Vaccine.pdf ×1; NCT03181503_Prurigo_Nodularis_Nemolizumab.pdf ×1; NCT03235752_Ulcerative_Colitis.pdf ×1; NCT03374254_Colon_Cancer.pdf ×1; NCT03434379_Hepatocellular_Atezo_Bev.pdf ×1; NCT03548935_Obesity_Semaglutide.pdf ×1; NCT03662659_Gastric_Cancer_Relatlimab.pdf ×1; NCT03753074_Hepatitis_B_TAF.pdf ×1; NCT03961204_Classic_MS.pdf ×1; NCT04032704_Solid_Tumors_Ladiratuzumab.pdf ×1; NCT04280705_COVID_ACTT_Remdesivir.pdf ×1; NCT04368728_COVID_BNT162_Pfizer.pdf ×1; NCT04470427_COVID_mRNA1273_Moderna.pdf ×1; NCT04614948_COVID_Ad26_Janssen.pdf ×1; NCT04652245_Allergic_Rhinitis_Dymista.pdf ×1 |

### Drug  (0 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|

### MeSHTerm  (91 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `key` | 91 / 91 | str | 91 distinct · e.g. glaucoma; colorectal neoplasms; atezolizumab; cobimetinib |
| `source` | 91 / 91 | str | registry ×91 |
| `term` | 91 / 91 | str | 91 distinct · e.g. Glaucoma; Colorectal Neoplasms; atezolizumab; cobimetinib |

### Outcome  (449 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `description` | 449 / 449 | str | 354 distinct · e.g. ; The ordinal scale is an assessment of the clinical status at the first assessment of a giv…; Systemic events included fever, fatigue, headache, chills, vomiting, diarrhea, new or wors…; Pruritus NRS is a scale used by the participants to report the intensity of their pruritus… |
| `key` | 449 / 449 | str | 445 distinct · e.g. overall survival (os); change in body weight (%); change in amylase - ratio to baseline; change in lipase - ratio to baseline |
| `measure` | 449 / 449 | str | 445 distinct · e.g. Overall Survival (OS); Change in Body Weight (%); Change in Amylase - Ratio to Baseline; Change in Lipase - Ratio to Baseline |
| `nctId` | 449 / 449 | str | NCT04368728 ×95; NCT04280705 ×43; NCT03548935 ×42; NCT02863419 ×40; NCT03434379 ×39; NCT04032704 ×29; NCT03753074 ×28; NCT04614948 ×22; NCT04470427 ×20; NCT02951156 ×19; NCT03181503 ×16; NCT03164772 ×12; NCT03961204 ×11; NCT02788279 ×10; NCT03662659 ×10; NCT02014597 ×4; NCT03235752 ×3; NCT03155620 ×2; NCT03374254 ×2; NCT04652245 ×2 |
| `source` | 449 / 449 | str | registry ×449 |
| `timeFrame` | 449 / 449 | str | 183 distinct · e.g. Baseline (week 0) to week 68; At year 4, 8 and 12; Week 0, week 26, week 52; Day 1 through Day 29 |
| `type` | 449 / 449 | str | secondary ×349; primary ×100 |

### PatientPopulation  (20 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `eligibilityCriteria` | 20 / 20 | str | Inclusion Criteria:

Normal controls for Study 1

1. Male or Female
2. age 18 or older
3. … ×1; Inclusion Criteria:

Disease-specific inclusion criteria:

* Histologically confirmed aden… ×1; Inclusion Criteria:

* Informed consent obtained before any trial-related activities. Tria… ×1; Key Inclusion Criteria:

-Any of the following as defined by the WHO, 2016 lymphoid neopla… ×1; Inclusion Criteria:

* ELIGIBILITY CRITERIA FOR ENROLLMENT ONTO APEC1621SC: Patients must … ×1; Inclusion Criteria

1. Histologic confirmation of metastatic NSCLC. For subjects with know… ×1; Inclusion Criteria:

1. Male or female of at least 18 years at screening
2. Clinical diagn… ×1; Inclusion Criteria:

1. Male and female patients 18-70 (inclusive) years of age.
2. Hisory… ×1; Inclusion Criteria:

* At least 18 years of age
* Has a histologically-confirmed, unresect… ×1; Inclusion Criteria:

* Locally advanced or metastatic and/or unresectable Hepatocellular C… ×1; Inclusion Criteria:

Main phase:

* Male or female, age greater than or equal to 18 years … ×1; For more information regarding Bristol-Myers Squibb Clinical Trial participation, please v… ×1; Inclusion Criteria: Patients must meet all of the following criteria to be eligible to par… ×1; Inclusion Criteria:

* Participants with relapsing remitting multiple sclerosis (RRMS) ran… ×1; Inclusion Criteria

* All Cohorts

  * Measurable disease according to RECIST v1.1 as asse… ×1; Inclusion Criteria:

1. Admitted to a hospital with symptoms suggestive of COVID-19 infect… ×1; Inclusion Criteria:

• Male or female participants between the ages of 18 and 55 years, in… ×1; Inclusion Criteria:

* (Part A only) Participants who are at high risk of SARS-CoV-2 infec… ×1; Inclusion Criteria:

* Contraceptive (birth control) use should be consistent with local r… ×1; Inclusion Criteria:

1. Provide written informed consent.
2. Male or female subjects (chil… ×1 |
| `gender` | 20 / 20 | str | ALL ×20 |
| `healthyVolunteers` | 20 / 20 | str | False ×16; True ×4 |
| `maximumAge` | 6 / 20 | str | 21 Years ×1; 70 Years ×1; 80 Years ×1; 65 Years ×1; 99 Years ×1; 55 Years ×1 |
| `minimumAge` | 20 / 20 | str | 18 Years ×17; 12 Months ×1; 40 Years ×1; 12 Years ×1 |
| `nctId` | 20 / 20 | str | NCT02014597 ×1; NCT02788279 ×1; NCT02863419 ×1; NCT02951156 ×1; NCT03155620 ×1; NCT03164772 ×1; NCT03181503 ×1; NCT03235752 ×1; NCT03374254 ×1; NCT03434379 ×1; NCT03548935 ×1; NCT03662659 ×1; NCT03753074 ×1; NCT03961204 ×1; NCT04032704 ×1; NCT04280705 ×1; NCT04368728 ×1; NCT04470427 ×1; NCT04614948 ×1; NCT04652245 ×1 |
| `source` | 20 / 20 | str | registry ×20 |
| `stdAges` | 20 / 20 | list[str] | ['ADULT', 'OLDER_ADULT'] ×17; ['CHILD', 'ADULT'] ×1; ['CHILD', 'ADULT', 'OLDER_ADULT'] ×1; ['ADULT'] ×1 |

### Section  (464 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `docId` | 464 / 464 | str | nct02014597-glaucoma-optokinetic ×49; nct02788279-cobimetinib-atezolizumab-go30182 ×47; nct04470427-covid-mrna1273-moderna ×44; nct04032704-solid-tumors-ladiratuzumab ×32; nct03434379-hepatocellular-atezo-bev ×28; nct02863419-t2d-oral-semaglutide-pioneer4 ×27; nct03235752-ulcerative-colitis ×23; nct03753074-hepatitis-b-taf ×23; nct02951156-prostate-cancer ×22; nct04652245-allergic-rhinitis-dymista ×20; nct03374254-colon-cancer ×19; nct03662659-gastric-cancer-relatlimab ×17; nct03961204-classic-ms ×17; nct03181503-prurigo-nodularis-nemolizumab ×16; nct04614948-covid-ad26-janssen ×16; nct03548935-obesity-semaglutide ×15; nct04368728-covid-bnt162-pfizer ×15; nct03164772-nsclc-mrna-vaccine ×13; nct04280705-covid-actt-remdesivir ×12; nct03155620-parkinsons-study ×9 |
| `heading` | 464 / 464 | str | 401 distinct · e.g. (no heading); PROTOCOL SYNOPSIS; LIST OF ABBREVIATIONS AND DEFINITIONS OF TERMS; 11. REFERENCES |
| `key` | 464 / 464 | str | 377 distinct · e.g. (no heading); 11 references; 4 study design; protocol synopsis |
| `origin` | 464 / 464 | str | structure ×464 |
| `sectionKey` | 464 / 464 | str | 464 distinct · e.g. nct02014597-glaucoma-optokinetic:d3765d8ab8be; nct02014597-glaucoma-optokinetic:0d5a5d755ccd; nct02014597-glaucoma-optokinetic:7458b5e69a7d; nct02014597-glaucoma-optokinetic:477cbd79ffa2 |

### Site  (1,085 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `city` | 1,085 / 1,085 | str | 551 distinct · e.g. Seoul; Chicago; Baltimore; New York |
| `facility` | 1,085 / 1,085 | str | 1,085 distinct · e.g. Alkek Eye Center, Baylor College of Medicine; City of Hope Comprehensive Cancer Center; Yale Cancer Center; Medical Oncology; Georgetown University |
| `key` | 1,085 / 1,085 | str | 1,066 distinct · e.g. meridian clinical research; washington university school medicine; samsung medical center; st george hospital |
| `lat` | 1,069 / 1,085 | float | 550 distinct · min -42.87936 · median 39.47391 · max 64.5461 · e.g. 37.566; 41.85003; 39.29038; 40.71427 |
| `lon` | 1,069 / 1,085 | float | 550 distinct · min -157.85833 · median -74.45182 · max 153.01852 · e.g. 126.9784; -87.65005; -76.61219; -74.00597 |
| `source` | 1,085 / 1,085 | str | registry ×1085 |
| `zip` | 942 / 1,085 | str | 757 distinct · e.g. 21201; 77030; 78229; 37203 |

### Sponsor  (18 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `key` | 18 / 18 | str | benjamin frankfort md phd ×1; hoffmann-la roche ×1; novo nordisk a/s ×1; pfizer ×1; national cancer institute (nci) ×1; ludwig institute cancer research ×1; galderma r&d ×1; i-mab biopharma hongkong limited ×1; merck sharp & dohme ×1; bristol-myers squibb ×1; young-suk lim ×1; emd serono research & development institute ×1; seagen ×1; national institute allergy infectious diseases (niaid) ×1; biontech se ×1; modernatx ×1; janssen vaccines & prevention b v ×1; meda pharma & kg ×1 |
| `name` | 18 / 18 | str | Benjamin Frankfort, MD, PhD ×1; Hoffmann-La Roche ×1; Novo Nordisk A/S ×1; Pfizer ×1; National Cancer Institute (NCI) ×1; Ludwig Institute for Cancer Research ×1; Galderma R&D ×1; I-Mab Biopharma HongKong Limited ×1; Merck Sharp & Dohme LLC ×1; Bristol-Myers Squibb ×1; Young-Suk Lim ×1; EMD Serono Research & Development Institute, Inc. ×1; Seagen Inc. ×1; National Institute of Allergy and Infectious Diseases (NIAID) ×1; BioNTech SE ×1; ModernaTX, Inc. ×1; Janssen Vaccines & Prevention B.V. ×1; MEDA Pharma GmbH & Co. KG ×1 |
| `source` | 18 / 18 | str | registry ×18 |

### Trial  (20 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `acronym` | 7 / 20 | str | HOCD ×1; PIONEER 4 ×1; Javelin DLBCL ×1; IMbrave150 ×1; STEP 1 ×1; ATTENTION ×1; ENSEMBLE 2 ×1 |
| `briefTitle` | 20 / 20 | str | Human Optokinetic Contrast Device (HOCD) to Measure Visual Function and Identify Patients … ×1; A Study to Investigate Efficacy and Safety of Cobimetinib Plus Atezolizumab and Atezolizum… ×1; Efficacy and Safety of Oral Semaglutide Versus Liraglutide and Versus Placebo in Subjects … ×1; Avelumab In Combination Regimens That Include An Immune Agonist, Epigenetic Modulator, CD2… ×1; Targeted Therapy Directed by Genetic Testing in Treating Pediatric Patients With Relapsed … ×1; Phase 1/2 Study of Combination Immunotherapy and Messenger Ribonucleic Acid (mRNA) Vaccine… ×1; Safety and Efficacy of Nemolizumab in PN ×1; Safety and Efficacy of TJ301 IV in Participants With Active Ulcerative Colitis ×1; Safety and Efficacy of Pembrolizumab (MK-3475) Plus Binimetinib Alone or Pembrolizumab Plu… ×1; A Study of Atezolizumab in Combination With Bevacizumab Compared With Sorafenib in Patient… ×1; STEP 1: Research Study Investigating How Well Semaglutide Works in People Suffering From O… ×1; An Investigational Study of Immunotherapy Combinations With Chemotherapy in Patients With … ×1; Effectiveness of TAF in Reducing Clinical Events in CHB Patients Beyond Treatment Indicati… ×1; Long-Term Outcomes and Durability of Effect Following Treatment With Cladribine Tablets fo… ×1; A Study of Ladiratuzumab Vedotin in Advanced Solid Tumors ×1; Adaptive COVID-19 Treatment Trial (ACTT) ×1; Study to Describe the Safety, Tolerability, Immunogenicity, and Efficacy of RNA Vaccine Ca… ×1; A Study to Evaluate Efficacy, Safety, and Immunogenicity of mRNA-1273 Vaccine in Adults Ag… ×1; A Study of Ad26.COV2.S for the Prevention of SARS-CoV-2-mediated COVID-19 in Adults ×1; Dymista Allergen Chamber - Onset of Action Study ×1 |
| `completionDate` | 20 / 20 | str | 2017-08-22 ×1; 2018-12-26 ×1; 2018-03-30 ×1; 2019-12-02 ×1; 2027-01-06 ×1; 2021-10-29 ×1; 2018-09-26 ×1; 2020-12-21 ×1; 2023-07-18 ×1; 2022-11-17 ×1; 2021-03-05 ×1; 2024-01-18 ×1; 2031-12-31 ×1; 2021-05-13 ×1; 2023-11-28 ×1; 2020-05-21 ×1; 2023-02-10 ×1; 2022-12-29 ×1; 2023-06-18 ×1; 2021-12-13 ×1 |
| `enrollmentCount` | 20 / 20 | int | 23 ×1; 363 ×1; 711 ×1; 29 ×1; 1377 ×1; 61 ×1; 70 ×1; 91 ×1; 116 ×1; 558 ×1; 1961 ×1; 274 ×1; 780 ×1; 662 ×1; 205 ×1; 1062 ×1; 46969 ×1; 30415 ×1; 31835 ×1; 216 ×1 |
| `enrollmentType` | 20 / 20 | str | ACTUAL ×19; ESTIMATED ×1 |
| `key` | 20 / 20 | str | nct02014597 ×1; nct02788279 ×1; nct02863419 ×1; nct02951156 ×1; nct03155620 ×1; nct03164772 ×1; nct03181503 ×1; nct03235752 ×1; nct03374254 ×1; nct03434379 ×1; nct03548935 ×1; nct03662659 ×1; nct03753074 ×1; nct03961204 ×1; nct04032704 ×1; nct04280705 ×1; nct04368728 ×1; nct04470427 ×1; nct04614948 ×1; nct04652245 ×1 |
| `lastUpdateSubmitDate` | 20 / 20 | str | 2023-05-09 ×1; 2019-12-03 ×1; 2022-07-11 ×1; 2020-11-20 ×1; 2026-07-23 ×1; 2022-10-03 ×1; 2020-02-07 ×1; 2021-01-03 ×1; 2024-11-05 ×1; 2023-10-05 ×1; 2021-11-18 ×1; 2025-01-14 ×1; 2024-12-12 ×1; 2023-10-26 ×1; 2025-02-25 ×1; 2022-03-09 ×1; 2026-03-03 ×1; 2024-03-19 ×1; 2025-01-31 ×1; 2023-02-24 ×1 |
| `nctId` | 20 / 20 | str | NCT02014597 ×1; NCT02788279 ×1; NCT02863419 ×1; NCT02951156 ×1; NCT03155620 ×1; NCT03164772 ×1; NCT03181503 ×1; NCT03235752 ×1; NCT03374254 ×1; NCT03434379 ×1; NCT03548935 ×1; NCT03662659 ×1; NCT03753074 ×1; NCT03961204 ×1; NCT04032704 ×1; NCT04280705 ×1; NCT04368728 ×1; NCT04470427 ×1; NCT04614948 ×1; NCT04652245 ×1 |
| `officialTitle` | 20 / 20 | str | Use of a Novel, Objective Optokinetic Contrast Device to Determine Scotopic Range Visual F… ×1; A Phase III, Open-Label, Multicenter, Three-Arm, Randomized Study to Investigate the Effic… ×1; Efficacy and Safety of Oral Semaglutide Versus Liraglutide and Versus Placebo in Subjects … ×1; PHASE 1B/PHASE 3 MULTICENTER STUDY OF AVELUMAB (MSB0010718C) IN COMBINATION REGIMENS THAT … ×1; NCI-COG Pediatric MATCH (Molecular Analysis for Therapy Choice) Screening Protocol ×1; A Phase 1/2 Study of Combination Immunotherapy and mRNA Vaccine in Subjects With Non-small… ×1; A Study to Assess the Safety and Efficacy of Nemolizumab (CD14152) in Subjects With Prurig… ×1; A Phase II, Randomized, Double-blind, Placebo-controlled Study to Evaluate the Safety and … ×1; A Phase 1b Multi-cohort Study of the Combination of Pembrolizumab (MK-3475) Plus Binimetin… ×1; A Phase III, Open-Label, Randomized Study of Atezolizumab in Combination With Bevacizumab … ×1; Effect and Safety of Semaglutide 2.4 mg Once-weekly in Subjects With Overweight or Obesity ×1; A Randomized, Open-label, Phase II Clinical Trial of Relatlimab (Anti-LAG-3) and Nivolumab… ×1; A Multinational, Multicenter, Open-label, Randomized Controlled Trial to Investigate the E… ×1; Evaluating the Long-Term Outcomes and Durability of Effect Following Treatment With Cladri… ×1; Open-Label Phase 2 Study of Ladiratuzumab Vedotin (LV) for Unresectable Locally Advanced o… ×1; A Multicenter, Adaptive, Randomized Blinded Controlled Trial of the Safety and Efficacy of… ×1; A PHASE 1/2/3, PLACEBO-CONTROLLED, RANDOMIZED, OBSERVER-BLIND, DOSE-FINDING STUDY TO EVALU… ×1; A Phase 3, Randomized, Stratified, Observer-Blind, Placebo-Controlled Study to Evaluate th… ×1; A Randomized, Double-blind, Placebo-controlled Phase 3 Study to Assess the Efficacy and Sa… ×1; Clinical Trial to Assess Onset of Action of Azelastine Hydrochloride and Fluticasone Propi… ×1 |
| `overallStatus` | 20 / 20 | str | COMPLETED ×15; TERMINATED ×3; ACTIVE_NOT_RECRUITING ×2 |
| `phase` | 20 / 20 | str | PHASE3 ×8; PHASE2 ×5; PHASE4 ×3; NA ×1; PHASE1, PHASE2 ×1; PHASE1 ×1; PHASE2, PHASE3 ×1 |
| `primaryCompletionDate` | 20 / 20 | str | 2017-08-22 ×1; 2018-03-09 ×1; 2017-08-19 ×1; 2019-12-02 ×1; 2025-03-31 ×1; 2021-10-29 ×1; 2018-09-26 ×1; 2020-12-21 ×1; 2021-09-08 ×1; 2020-08-31 ×1; 2020-03-30 ×1; 2020-08-27 ×1; 2031-12-31 ×1; 2021-02-27 ×1; 2023-11-28 ×1; 2020-05-21 ×1; 2023-02-10 ×1; 2022-12-29 ×1; 2023-06-18 ×1; 2021-12-04 ×1 |
| `source` | 20 / 20 | str | registry ×20 |
| `startDate` | 20 / 20 | str | 2015-05 ×1; 2016-07-05 ×1; 2016-08-10 ×1; 2016-12-16 ×1; 2017-07-31 ×1; 2017-12-20 ×1; 2017-10-02 ×1; 2018-02-06 ×1; 2018-02-16 ×1; 2018-03-15 ×1; 2018-06-04 ×1; 2018-10-16 ×1; 2019-02-18 ×1; 2019-08-15 ×1; 2019-10-09 ×1; 2020-02-21 ×1; 2020-04-29 ×1; 2020-07-27 ×1; 2020-11-12 ×1; 2020-12-14 ×1 |
| `studyType` | 20 / 20 | str | INTERVENTIONAL ×20 |

### TrialCategory  (20 nodes)

| Property | Present on | Types | Values |
|---|---|---|---|
| `key` | 20 / 20 | str | glaucoma ×1; colorectal cancer ×1; diabetes ×1; diffuse large b-cell lymphoma ×1; advanced malignant solid neoplasm ×1; metastatic non-small cell lung cancer ×1; prurigo nodularis ×1; active ulcerative colitis ×1; metastatic colorectal cancer ×1; carcinoma hepatocellular ×1; metabolism nutrition disorder ×1; gastric cancer ×1; chronic hepatitis b ×1; multiple sclerosis (ms) ×1; small cell lung cancer ×1; covid-19 ×1; sars-cov-2 infection ×1; sars-cov-2 ×1; participants with without stable co-morbidities associated with progression to severe covi… ×1; seasonal allergic rhinitis ×1 |
| `name` | 20 / 20 | str | Glaucoma ×1; Colorectal Cancer ×1; Diabetes ×1; Diffuse Large B-Cell Lymphoma ×1; Advanced Malignant Solid Neoplasm ×1; Metastatic Non-small Cell Lung Cancer ×1; Prurigo Nodularis ×1; Active Ulcerative Colitis ×1; Metastatic Colorectal Cancer ×1; Carcinoma, Hepatocellular ×1; Metabolism and Nutrition Disorder ×1; Gastric Cancer ×1; Chronic Hepatitis b ×1; Multiple Sclerosis (MS) ×1; Small Cell Lung Cancer ×1; COVID-19 ×1; SARS-CoV-2 Infection ×1; SARS-CoV-2 ×1; Participants With or Without Stable Co-morbidities Associated With Progression to Severe C… ×1; Seasonal Allergic Rhinitis ×1 |
| `source` | 20 / 20 | str | registry ×20 |

## 3 · Relationships

| Pattern | Count | Relationship properties |
|---|---|---|
| (:Chunk)-[:NEXT]->(:Chunk) | 5,743 | — |
| (:Document)-[:ABOUT]->(:Trial) | 20 | — |
| (:Document)-[:HAS_SECTION]->(:Section) | 464 | — |
| (:Section)-[:HAS_CHUNK]->(:Chunk) | 5,764 | — |
| (:Site)-[:IN_COUNTRY]->(:Country) | 1,109 | — |
| (:Trial)-[:BELONGS_TO]->(:TrialCategory) | 20 | — |
| (:Trial)-[:CONDUCTED_IN]->(:Country) | 162 | — |
| (:Trial)-[:ENROLLS]->(:PatientPopulation) | 20 | — |
| (:Trial)-[:INDEXED_AS]->(:MeSHTerm) | 100 | — |
| (:Trial)-[:LOCATED_AT]->(:Site) | 1,132 | — |
| (:Trial)-[:MANAGED_BY]->(:CRO) | 33 | — |
| (:Trial)-[:MEASURES]->(:Outcome) | 449 | — |
| (:Trial)-[:SPONSORED_BY]->(:Sponsor) | 20 | — |
| (:Trial)-[:TARGETS]->(:Disease) | 78 | — |

## 4 · Per trial (min · median · max)

| Neighbour | Min | Median | Max |
|---|---|---|---|
| LOCATED_AT → Site | 1132 | 1132 | 1132 |
| CONDUCTED_IN → Country | 162 | 162 | 162 |
| MEASURES → Outcome | 449 | 449 | 449 |
| TARGETS → Disease | 78 | 78 | 78 |
| INDEXED_AS → MeSHTerm | 100 | 100 | 100 |
| SPONSORED_BY → Sponsor | 20 | 20 | 20 |
| MANAGED_BY → CRO | 33 | 33 | 33 |
| TESTS → Drug | 0 | 0 | 0 |
| Document ABOUT | 20 | 20 | 20 |

## 5 · Indexes and constraints

| Name | Type | Labels | Properties |
|---|---|---|---|
| category_key | RANGE | ['TrialCategory'] | ['key'] |
| category_name | RANGE | ['TrialCategory'] | ['name'] |
| chunk_id | RANGE | ['Chunk'] | ['chunkId'] |
| chunk_page | RANGE | ['Chunk'] | ['page'] |
| chunk_type | RANGE | ['Chunk'] | ['contentType'] |
| country_key | RANGE | ['Country'] | ['key'] |
| country_name | RANGE | ['Country'] | ['name'] |
| cro_key | RANGE | ['CRO'] | ['key'] |
| cro_name | RANGE | ['CRO'] | ['name'] |
| disease_key | RANGE | ['Disease'] | ['key'] |
| disease_name | RANGE | ['Disease'] | ['name'] |
| document_id | RANGE | ['Document'] | ['docId'] |
| drug_key | RANGE | ['Drug'] | ['key'] |
| drug_name | RANGE | ['Drug'] | ['name'] |
| index_1b9dcc97 | LOOKUP | None | None |
| index_460996c0 | LOOKUP | None | None |
| mesh_key | RANGE | ['MeSHTerm'] | ['key'] |
| mesh_term | RANGE | ['MeSHTerm'] | ['term'] |
| outcome_key | RANGE | ['Outcome'] | ['key'] |
| section_id | RANGE | ['Section'] | ['sectionKey'] |
| section_key | RANGE | ['Section'] | ['key'] |
| site_city | RANGE | ['Site'] | ['city'] |
| site_facility | RANGE | ['Site'] | ['facility'] |
| site_key | RANGE | ['Site'] | ['key'] |
| sponsor_key | RANGE | ['Sponsor'] | ['key'] |
| sponsor_name | RANGE | ['Sponsor'] | ['name'] |
| trial_entity_names | FULLTEXT | ['Trial', 'Sponsor', 'Disease', 'CRO', 'Site', 'Drug'] | ['nctId', 'briefTitle', 'officialTitle', 'acronym', 'name', 'facility'] |
| trial_key | RANGE | ['Trial'] | ['key'] |
| trial_nct_id | RANGE | ['Trial'] | ['nctId'] |
| trial_phase | RANGE | ['Trial'] | ['phase'] |
| trial_status | RANGE | ['Trial'] | ['overallStatus'] |
