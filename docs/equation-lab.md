# Prospective equation laboratory

The user authorized several equations to be compared before any integration
into the CEO. This laboratory does not alter risk-department tickets, allocate
money, authorize purchases, or automatically promote a model.

## Fixed candidate families

1. `market_residual`: a race-level softmax with inverse-odds log offsets and 24
   bounded individual/pair/triple features. Ridge-regularized log-loss fitting
   gives each race equal weight. The market is used once.
2. `pairwise_order`: an antisymmetric logistic comparison between riders, fitted
   only on pairs whose order is identifiable from the official top three. The
   order among finishers below third is never fabricated. The joint weight of
   `(a,b,c)` is `q(a,b) q(a,c) q(b,c)` times each top-three rider's probability of
   preceding each remaining rider. Normalize over **all** ordered triples. This
   is a probabilistic approximation, not a claim that pair outcomes are independent.
3. `state_paths`: a coarse observed top-three queue process through start, bell,
   back straight and finish. States are ranks in the frozen pre-race win model;
   empirical transition rows shrink toward the observed next-stage marginal.
   Multiply the distributions through the three transitions, then map back to
   the race's car numbers. This does not simulate exact speed or remaining energy.

All training settings are in `equation_models.CONFIG`. An initial 50 races on
7 distinct JST days is an operational fitting floor, **not** evidence of accuracy.
The training window is the previous 180 days, at most 2,000 latest complete races.
Each day's first fitted bundle is immutable and uses prior JST days only.
Previous evaluation days may become training data on a later day: this is an
explicit prequential, daily-refit policy. Hyperparameters are not adjusted based
on ongoing results. A code change starts a separate cohort.

## Evidence and capture

`annual_position_v2_ledger.jsonl` supplies original complete canonical pre-close
inputs, source/model fingerprints and unchanged risk tickets. No old forecasts
are regenerated. A hook after a successful canonical capture runs the new
equations, checks actual completion time and quote freshness again, and writes
`annual_equation_ledger.jsonl`. Attempts and immutable model bundles have separate
append-only ledgers. Publication reconciles all ledgers across concurrent writes.

Each source input is complete, has one quote timestamp within 300 seconds, and
was captured 40 to 5 minutes before close. Capped odds are unavailable, not prices.
Result fields cannot enter the input features. Training labels must have been
observed before the training cutoff; ties and conflicting labels are excluded
from training. Evaluation settles ties using all official winning-ticket payouts.

## Observed paths: data required, not invented

`outputs/company/annual_equation_observed_paths.jsonl` accepts records with:

```json
{
  "schema": "observed_top3_stages_v1",
  "race_id": "real race identifier matching frozen evidence",
  "observed_at": "timezone-aware annotation timestamp after the race",
  "source_url": "https://provider.example/actual-race-video",
  "verified": true,
  "orders": {
    "start": [1, 2, 3],
    "bell": [2, 1, 3],
    "back": [2, 3, 1],
    "finish": [3, 2, 1]
  }
}
```

The example is a **schema only**, never training data. `start` means the top
three when the leader passes the finish line with two laps remaining; `bell` is
the first audible bell in the source video; `back` is when the leader crosses
the final back-straight line. Use actual
observations with that fixed phase convention, not reconstructed stories based
on the final result. The finish must equal the official unique winning order.
Conflicting annotations are excluded. Each field-size bucket independently needs
50 verified paths over 7 days. There is no automatic video annotation pipeline in
this change. Until observations exist, the path equation is explicitly untrained.

## Selection and comparison

All three candidates evaluate every ordered triple. Selection is by probability,
without an early first-place pool. Existing eligibility limits are held fixed:
main EV >= 1.10; hole odds >= 100, EV >= 1.25 and probability >= 0.001.
These are uncalibrated research gates, not certified economic value.

For each group separately, match the original risk portfolio's count (1–12) at
100 yen per ticket. Insufficient eligible candidates produce **no comparison**,
not padding. The risk portfolio is copied unchanged, including its timestamp and
prices. Each method/risk comparison uses its latest jointly eligible pre-close
snapshot. Method-to-method descriptive comparisons use a single shared snapshot
and are reported separately; never compare raw rates from different race cohorts.

The confirmatory window is fixed at 56 days from that method/group/cohort's first
eligible snapshot. After the deadline, fewer than 500 paired races or 28 race days
means insufficient evidence; do not extend the window until it wins. Hit-rate and
pooled-ROI differences use day-cluster bootstrap intervals with a predeclared
12-endpoint family correction (3 equations x 2 groups x 2 metrics). Intervals are
approximate, not a guarantee. Missing payoffs exclude that pair from both metrics.
Report rescued/lost hits and dependence on the single largest payout separately.

Reports expose evidence only. Even a positive hit-rate interval does not enable
the CEO. Review calibration, cohort coverage, ROI and payout concentration, then
implement any approved CEO integration as a separate change.

## Operations

The existing prediction/report workflows call the laboratory. To inspect without
creating historical forecasts, run `python src/equation_lab.py report`. `train`
may freeze the current day's model bundle but cannot create a forecast. No CLI
accepts an arbitrary historical capture time. Unit tests use explicit fixture
clocks only. New files and ledgers are covered by the publication source and
evidence checks.
