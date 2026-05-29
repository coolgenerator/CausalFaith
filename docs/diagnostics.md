# K562 Perturb-seq: Intervention Quality and Confounding Diagnostics

**Author**: Kefan Zhang (intervention quality and confounding diagnostics)
**Dataset**: Replogle 2022 K562 essential-scale CRISPRi Perturb-seq (310,385 cells, 8,563 genes)
**Working subset**: 300 genes selected by coverage and control expression rank

## 1. Summary

Two of the three failure modes posited in the CausalBench paradox can be
quantified directly on the K562 essential-scale data. CRISPRi knockdown
strength varies substantially across perturbations. Among the 300
perturbations whose target gene falls inside the working subset, 46 percent
(138) achieve a residual ratio below 0.20, while 16 percent (47) retain at
least half of the control expression level. Batch confounding is moderate
overall, with a median Jensen-Shannon divergence of 0.089 against the control
distribution over gemgroups. The most batch-confounded perturbations are
concentrated in the 1,749 perturbations whose target gene was not measured;
the 300 KD-measurable perturbations all fall in the low or medium batch
strata. Cell cycle phase distributions are significantly shifted in 337
perturbations at the Bonferroni-corrected level, reflecting a mixture of
biological hits on cell cycle genes and possible cell cycle confounding that
downstream analyses should disentangle by stratification rather than by
filtering.

## 2. Knockdown Efficiency

We define per-perturbation knockdown efficiency as the residual ratio
between mean target-gene expression in perturbed cells and mean target-gene
expression in control cells, computed on counts that have been per-cell
normalized to 10,000 total counts. A residual ratio close to zero corresponds
to a strong intervention; a ratio close to one indicates that the
perturbation barely altered the target. Computations on the log1p layer were
deliberately avoided because the geometric-mean nature of expm1(mean(log1p))
biases the ratio away from the arithmetic mean of normalized counts.

Among the 300 perturbations whose target gene appears in the working subset,
the residual ratio distribution is right-skewed with a median of 0.22 and an
interquartile range of 0.13 to 0.39. Only two perturbations have residual
ratio above 1, and both are within sampling noise (1.00 and 1.05). We
stratify perturbations into three groups by ratio: strong below 0.20,
medium between 0.20 and 0.50, and weak at or above 0.50. The stratum counts
are 138 strong, 115 medium, and 47 weak.

| Stratum | Threshold | Count | Fraction of measurable |
|---------|-----------|-------|------------------------|
| strong  | ratio < 0.20 | 138 | 46% |
| medium  | 0.20 ≤ ratio < 0.50 | 115 | 38% |
| weak    | ratio ≥ 0.50 | 47 | 16% |

These numbers are inconsistent with the perfect-intervention assumption that
underwrites GIES identifiability. Even under the most permissive
interpretation of "approximately perfect", a residual ratio above 0.5
implies that target expression in perturbed cells exceeds half of control,
which is far from a point mass at zero. The 47 weak perturbations are
therefore expected to behave more like soft interventions, consistent with
the assumptions of DCDI-G. The contrast in Module B between GIES and DCDI-G
on the strong stratum versus the weak stratum becomes the operational test
of whether intervention imperfectness explains a portion of the CausalBench
performance gap.

A secondary observation concerns the relationship between residual ratio
and control expression. Highly expressed targets such as RPL3, with a
control normalized mean above 30, tend toward higher residual ratios than
lowly expressed targets. This pattern is weak overall but visible in the
scatter of residual ratio against log control expression. Member 5's
analysis should therefore partial out control expression when stratifying
by KD efficiency, to avoid confounding intervention quality with target
abundance.

## 3. Batch Confounding

For each perturbation we computed the Jensen-Shannon divergence between
the distribution of perturbed cells over gemgroup and the distribution of
control cells over gemgroup, then squared the Jensen-Shannon distance to
obtain a divergence value bounded in [0, 1]. The median Jensen-Shannon
divergence across 2,049 perturbations with at least 10 cells is 0.089.
The seventy-fifth percentile is 0.149 and the ninetieth percentile is 0.247,
indicating that batch heterogeneity is modest for most perturbations but
substantial for a non-negligible minority. The maximum observed value is
0.651, an order of magnitude above the median.

To enable stratified analysis we partitioned perturbations into low,
medium, and high batch strata using the tertiles of the observed
divergence distribution. The resulting strata each contain 683
perturbations.

The crosstabulation of KD stratum against batch stratum reveals a
structural pattern that constrains the joint analysis design:

| KD stratum | batch low | batch medium | batch high |
|------------|-----------|--------------|------------|
| strong (138)              | 113 | 25 | 0 |
| medium (115)              | 103 | 12 | 0 |
| weak (47)                 | 43  | 4  | 0 |
| target_not_measured (1757)| 424 | 642 | 683 |

Every perturbation whose target gene lies inside the 300-gene subset falls
into the low or medium batch stratum. No KD-measurable perturbation
appears in the high batch stratum, while the high batch stratum is
populated entirely by perturbations whose target gene was not measured.
This pattern is a structural consequence of how the 300-gene subset was
constructed: genes were ranked partly on cell-count coverage, which
favors perturbations whose cells are broadly distributed across batches,
producing a low Jensen-Shannon divergence against control.

The practical consequence is that the imperfect-intervention failure mode
and the causal-sufficiency failure mode cannot be evaluated on the same
population of perturbations. We propose the following partition for
Module C:

- Imperfect intervention is tested on the 300 KD-measurable perturbations,
  stratified by residual ratio.
- Causal sufficiency violation is tested on the 1,757 KD-unmeasurable
  perturbations, stratified by batch Jensen-Shannon divergence.

A perturbation can in principle contribute to both analyses, but on this
dataset the two subpopulations are nearly disjoint.

## 4. Cell Cycle Bias

Cell cycle scoring was performed on the full 8,563-gene matrix from a
streaming reimplementation of the Tirosh et al. 2016 algorithm, using the
Regev lab S-phase and G2M-phase marker gene sets. Forty-two of forty-three
S markers and fifty of fifty-four G2M markers were present in the
expression matrix. The K562 population partitions into 41 percent S, 41
percent G2M, and 18 percent G1 cells, consistent with the rapid
proliferation expected of this leukemia line.

For each perturbation we computed two complementary measures of cell cycle
bias: a chi-square test of phase proportions against the control
distribution, and the mean differences in continuous S_score and G2M_score
between perturbed and control cells. The chi-square test was applied to
the 2×K contingency table of phase counts, where columns with zero total
were dropped to avoid spurious warnings.

Three hundred and thirty-seven perturbations exhibit a Bonferroni-corrected
phase shift at the family-wise significance threshold of 2.44 × 10⁻⁵
(α = 0.05 divided by 2,049). Score-difference medians are slightly negative
(median S_score difference equals -0.0085, median G2M_score difference
equals -0.0093), suggesting a small systematic shift toward G1, but no
pronounced bias. Individual perturbations reach absolute score differences
up to 0.21 for S_score and 0.19 for G2M_score.

These cell-cycle-disrupted perturbations comprise two phenomenologically
distinct groups that this module does not attempt to separate. The first
group includes perturbations targeting genes whose function lies in the
cell cycle itself, such as BUB1, KIF11, CDC20, and members of the MCM
family. A phase shift in this group is biologically expected and reflects
correct detection of an intervention rather than a confounder. The second
group includes perturbations whose phase shift arises from indirect
effects such as differential cell death, where phase composition shifts
because some sub-population was preferentially lost during the perturbation
window rather than because the targeted gene regulates cell cycle progression.

Member 5's cross-fitting analysis should treat the cell cycle flag as a
covariate to be partialled out, not as a filter for perturbation quality.
Filtering on this flag would remove genuine biological signal alongside
any confounding.

## 5. Joint Diagnostic Status

Combining the three diagnostics yields a per-perturbation status field
with three levels:

- full: KD measurable and batch diagnostic available (n = 300)
- kd_unmeasurable: target gene not in subset, only batch and cell cycle
  diagnostics available (n = 1,749)
- insufficient: fewer than 10 cells, no reliable diagnostic (n = 8)

This status column is the primary entry point for downstream analyses
that need a quick perturbation filter. Member 5 should restrict the
imperfect-intervention contrasts (GIES versus DCDI-G in Module B) to
status = full, and the causal-sufficiency contrasts (PC versus FCI) to
status in {full, kd_unmeasurable}.

## 6. Limitations

The 300-gene subset was selected by a weighted combination of perturbation
coverage rank and control expression rank, not at random. Conclusions
about CRISPRi knockdown efficiency apply only to high-coverage,
well-expressed essential genes; sensitivity checks on the matched random
subsets (provided by member 1) remain pending.

Knockdown is quantified as a population-level mean ratio, not as a
per-cell quantity. A perturbation classified as "strong" can still
contain individual cells with poor knockdown, and the homogeneity of
the intervention within a perturbation group is not assessed here.

Cell cycle phase is inferred rather than measured. Phase calls depend on
the assumption that K562 cells exhibit canonical Regev-lab marker
expression patterns; deviations from this assumption would bias the chi-square
test. No experimental phase calls were available for cross-validation.

The batch field considered here is gemgroup, the sequencing library
identifier. Other potential confounders such as library size,
mitochondrial fraction, and clonal variation are addressed separately
in Module A's residualization step and are not analyzed in this module.

The squared Jensen-Shannon distance is symmetric and bounded but
sensitive to the number of categories in the batch field. The K562
dataset contains 48 gemgroups, providing reasonable resolution; the
analysis should be revisited if applied to datasets with a small number
of batches.

## 7. Deliverables

| File | Contents |
|------|----------|
| `results/diagnostics/kd_efficiency.csv` | Per-perturbation KD efficiency (2,057 rows) |
| `results/diagnostics/batch_divergence.csv` | Per-perturbation batch Jensen-Shannon divergence |
| `results/diagnostics/cellcycle_bias.csv` | Per-perturbation phase chi-square and score differences |
| `results/diagnostics/perturbation_quality_table.csv` | Joined master table, 17 columns, all diagnostic columns prefixed `diag_` |
| `results/diagnostics/perturbation_quality_summary.json` | Aggregate summary statistics |
| `results/diagnostics/kd_efficiency_summary.json` | KD-specific summary |
| `results/diagnostics/confounding_summary.json` | Batch and cell cycle summary |
| `data/processed/k562_essential/cell_cycle_scores.csv` | Per-cell S_score, G2M_score, phase assignments |
| `results/diagnostics/plots/` | Nine diagnostic figures including marginal distributions and joint scatter plots |

## 8. Reproducibility

```bash
# Phase 1: knockdown efficiency
python scripts/run_intervention_quality.py \
    --processed-dir data/processed/k562_essential \
    --output-dir results/diagnostics

# Phase 2 step 1: cell cycle scoring on the full count matrix (one-shot)
python scripts/run_cell_cycle_scoring.py \
    --raw-h5ad data/causalbench/k562.h5ad \
    --output-csv data/processed/k562_essential/cell_cycle_scores.csv

# Phase 2 step 2: batch divergence and cell cycle diagnostics
python scripts/run_confounding.py \
    --processed-dir data/processed/k562_essential \
    --cellcycle-csv data/processed/k562_essential/cell_cycle_scores.csv \
    --output-dir results/diagnostics

# Phase 3: assemble the master diagnostic table
python scripts/build_perturbation_quality_table.py \
    --diagnostics-dir results/diagnostics
```

End-to-end runtime is approximately ten minutes on a 16 GB Apple Silicon
laptop, dominated by the streaming cell cycle scoring step. All
intermediate artifacts are deterministic given the random seeds recorded
in `kd_efficiency_summary.json` and the cell-cycle scoring script.