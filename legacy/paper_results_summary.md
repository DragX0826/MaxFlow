# MaxFlow Paper Results Summary

> [!IMPORTANT]
> **Phase 63 (BC-MaxRL) Training in Progress**
> The results below are from a previous unstable build (Phase 61/62 placeholders). Phase 63 introduces reward-ranked pairs and InfoNCE alignment, which is expected to resolve the current numerical failures and provide superior alignment.

## Summary Table

| Mode      |   ('Energy (kcal/mol)', 'mean') |   ('Energy (kcal/mol)', 'std') |   ('Time (s)', 'mean') |   ('Time (s)', 'std') |
|:----------|--------------------------------:|-------------------------------:|-----------------------:|----------------------:|
| Baseline  |                          -23.46 |                          12.44 |                  12.08 |                  1.13 |
| Full-SOTA |                        21795    |                       26626.8  |                  12.69 |                  0.9  |

## Detailed Results

| Mode      | Target              |   Energy (kcal/mol) |   Time (s) | Status   |
|:----------|:--------------------|--------------------:|-----------:|:---------|
| Baseline  | TIM_BARREL_DE_NOVO  |              -26.35 |      13.5  | PASS     |
| Baseline  | MPRO_SARS_COV_2     |              -32.4  |      13.13 | PASS     |
| Baseline  | METALLOENZYME_ZN    |              -14.38 |      12.26 | PASS     |
| Baseline  | G_COUPLED_RECEPTOR  |               -6.54 |      10.49 | PASS     |
| Baseline  | C_MYC_PPI_INTERFACE |              -32.62 |      11.21 | PASS     |
| Baseline  | COVALENT_KRAS_G12C  |              -39.25 |      13.97 | PASS     |
| Baseline  | MACROCYCLE_PCSK9    |              -16.74 |      11.94 | PASS     |
| Baseline  | PROTAC_TERNARY      |              -36.64 |      11.8  | PASS     |
| Baseline  | FE_S_CLUSTER_ENZYME |              -25.65 |      11.34 | PASS     |
| Baseline  | ALLOSTERIC_PTP1B    |               -3.99 |      11.16 | PASS     |
| Full-SOTA | TIM_BARREL_DE_NOVO  |             7632.33 |      11.37 | FAIL     |
| Full-SOTA | MPRO_SARS_COV_2     |            21409.6  |      13.4  | FAIL     |
| Full-SOTA | METALLOENZYME_ZN    |             1365.73 |      12.8  | FAIL     |
| Full-SOTA | G_COUPLED_RECEPTOR  |              717.79 |      14.46 | FAIL     |
| Full-SOTA | C_MYC_PPI_INTERFACE |             3412.69 |      12.6  | FAIL     |
| Full-SOTA | COVALENT_KRAS_G12C  |            22762.6  |      12.38 | FAIL     |
| Full-SOTA | MACROCYCLE_PCSK9    |            88140.6  |      12.39 | FAIL     |
| Full-SOTA | PROTAC_TERNARY      |            38907.1  |      11.45 | FAIL     |
| Full-SOTA | FE_S_CLUSTER_ENZYME |            27646.4  |      13.08 | FAIL     |
| Full-SOTA | ALLOSTERIC_PTP1B    |             5954.91 |      12.96 | FAIL     |
