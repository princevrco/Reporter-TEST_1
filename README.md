# Promenade energy analysis

Daily meter baseline regression, energy forecasting and measurement-and-verification
review tools. Version **0.5.0**. Python **3.11 or 3.12**. Runs on CPU.

The project analyzes all supplied meters independently. It compares measured energy
with a frozen historical baseline, forecasts baseline consumption for a selectable
period, and converts energy differences to CAD at a configurable flat rate.
It runs locally or in a Runpod Pod. It does not connect to CARMA or write BACnet commands.

## Analysis scope

| Output | Interpretation |
| --- | --- |
| Historical baseline | Estimated no-intervention energy from historical meter readings |
| Reporting comparison | Baseline minus measured kWh for matching eligible dates |
| CAD comparison | Signed kWh difference multiplied by the selected CAD/kWh rate |
| Three-month forecast | Future no-intervention baseline consumption; not optimized consumption |
| Optional zone analysis | Estimated thermal-mode durations and statistical meter associations, subject to timing, coverage and identification checks |

The two-file workflow uses a **calendar-only regression**. BMS data supplies timestamp
coverage context, not electrical consumption or verified runtime. Weather and other
operating changes are not adjusted in that workflow. A passed import or completed run
does not establish verified savings, a causal BMS effect, or standards certification.
The full CLI also supports supplied weather and independent covariates, as well as
daily or fixed 15/30/60-minute energy intervals. `run-all` supports daily data only.

The methodology references the supplied IPMVP December 1997 Option C document;
`config/protocol-reference.json` records that reference. Site commissioning, a reviewed
M&V plan, meter boundaries, adjustments and an uncertainty assessment remain required
before decision-grade savings claims. This package does not implement every protocol procedure.

## Install and verify

Extract or clone the repository, then change into the folder containing `pyproject.toml`.
Do not move an existing virtual environment between folders; recreate it instead.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,excel]'
python -m pytest -q
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

A synthetic demonstration is available without private data:

```bash
python examples/demo.py --output results/synthetic-demo --optimizer
```

Synthetic examples are not site results. The test extra includes pytest; the Excel
extra is needed only for Excel imports and workbook tests. Core CSV workflows need
only the base dependencies. Runtime dependencies are specified in `pyproject.toml`.

## Private input files

No real site data, trained models, generated reports, credentials or session captures
are included. Create `local/` inside the repository and place these files there:

```text
promenade-option-c/
  pyproject.toml
  config/promenade-test.json
  local/
    consumption_kwh.zip
    live_bms_history.csv
```

The CARMA ZIP must contain daily **Consumption kWh** CSV exports and their matching
quality JSON files, as produced by the separate exporter. Demand kW, cumulative
counters and monthly exports are not interchangeable with daily interval energy.
An extracted directory of those files is also accepted. The wide BMS CSV must have
a `timestamp` column with explicit UTC offsets. All meter IDs remain separate.

The data ZIP distributed privately in the earlier test can supply these two files.
Do not upload it, its contents, generated results or a virtual environment to GitHub.
`.gitignore` and `.dockerignore` exclude the standard private paths and data formats;
inspect staged files before committing. Custom folders and renamed files still need review.

## Promenade configuration

Use `config/promenade-test.json` for the supplied snapshots. Its full-control start has been confirmed; the other site assumptions remain under review:

- Full-control start: **August 15, 2026**, confirmed by the user for all assets on October 21.
- Baseline end, exclusive: **July 24, 2026**, excluding uncertain early commissioning/control history.
- Daily timezone: `America/Toronto`; CARMA's daily convention still needs confirmation.
- Review flags remain false until the corresponding facts have been verified.

Edit `intervention_timestamp` and `baseline_end_exclusive` after confirming the actual
control history. The baseline end cannot be later than intervention. Earlier asset
optimization must also be excluded from untreated training. Do not change review flags
or loosen validation limits merely to obtain a positive savings result.

`config/site.json` is an example for the advanced CLI, including illustrative BMS
point mappings. It is not a commissioned all-asset configuration.

## Meter download for the next test

Export **daily Consumption kWh for every available meter from January 1, 2025
through October 20, 2026 inclusive**. If CARMA uses an exclusive end date, select
October 21, 2026. Retain meter IDs, descriptions, date conventions and exporter
quality/checksum companions. Keep real zeros; do not replace missing days with zero.
A complete replacement export is preferable when CARMA has revised historical values.

The export should cover both historical training and the entire reporting period.
Recheck June 10 and November 2, 2025 in particular: the earlier virtual-total export
contained exceptionally large daily readings on those dates. A flag is an investigation
request, not proof that the reading is wrong. Confirm corrected values with CARMA.

## Baseline selection and meter quality

The supplied configurations enable the v0.5 daily baseline workflow:

- Fit ordinary least squares, Huber robust regression and weekday median baselines
  over all available historical data, the latest 365 days and the latest 180 days.
- Also test a simple annual sine/cosine calendar pattern for OLS/Huber when each
  fitting window contains at least a year with 90% coverage. It requires no weather
  file; it represents recurring calendar seasonality, not actual weather adjustment.
- Select the estimator/lookback using up to three rolling historical validation folds
  of 28 calendar days. At least two folds and 60 valid training days are required.
- Keep the final 56 historical calendar days as a shared, untouched test period.
  Every candidate is evaluated on those same dates; at least 80% coverage is required.
  Selection uses inner-fold error, never reporting-period savings or outer-test error.
- Refit the selected method on its chosen historical lookback. Preserve candidate
  results and compare reporting-period estimates from validation-passing alternatives.
  A spread above 10% of the selected baseline triggers review.
- Screen for extreme spikes, sustained zero runs and historical level changes.
  Screening thresholds are configurable engineering heuristics, not IPMVP limits.
  Flags preserve the source reading and withhold headline comparisons until reviewed.

The default held-out limits are CVRMSE <=20% and absolute NMBE <=5%. These are project
screens, not proof of standards compliance. A robust fit reduces the influence of
outliers but does not establish that they are invalid. Failed validation, unresolved
quality findings or high method sensitivity withhold the affected savings headline.
Diagnostic differences remain available for investigation in clearly named fields.
Future baseline totals are also withheld when baseline quality or validation fails;
plotted predictions from withheld models are diagnostic only. Existing v0.4 results
must be rerun to receive these checks; summarizing old artifacts cannot add them.

Review these files before interpreting a result:

| File | Purpose |
| --- | --- |
| `meter-consumption.csv` | Source consumption by meter, independent of baseline fitting; available-days and complete-period totals remain separate |
| `data-quality-events.csv` / `.json` | Flagged dates, reasons and anchored event IDs |
| `quality-review-summary.json` | Resolved, unresolved and stale/unmatched review IDs |
| `quality-reviewed-meter.csv` | Analysis copy; corrections/exclusions preserve original kWh |
| `meter-.../baseline-comparison.json` | Candidate forecasts and method sensitivity |
| `meter-.../baseline-model.json` | Selected model, chronological validation and review gates |
| `response/point-coverage.csv` | BMS point cadence, gaps and usable samples |
| `meter-summary.csv` | Released comparisons, diagnostic differences and reasons for withholding |

`merged-meter.csv` remains the merged original input. No automatic deletion,
interpolation, clipping or replacement of unreviewed readings is performed. Only a
documented review can exclude/correct an anchored event in the analysis copy.

To record a decision, copy the event ID into a private JSON ledger such as
`local/quality-reviews.json`:

```json
[
  {
    "event_id": "COPY_EXACT_EVENT_ID_FROM_REPORT",
    "reviewed": true,
    "reviewer": "Reviewer name",
    "reference": "CARMA support ticket or other documented evidence",
    "decision": "accept"
  }
]
```

Use `accept` only after confirming the reading is valid; `exclude` for a confirmed
unusable reading; or `correct` plus `corrected_kwh` for a documented single-day
correction. An exclusion leaves a missing period; it never creates zero consumption.
Add `--quality-reviews local/quality-reviews.json` to the analysis command and use a
new output directory. Event IDs include raw dates/values; stale decisions cannot
silently apply to revised source readings. Re-exporting corrected data is preferable
to manual corrections. Do not approve flags merely to obtain plausible savings.

A quality-only preparation run is available with `--prepare-only`. For revised
CARMA overlaps, first inspect the conflict; use `--overlap-policy prefer-last` only
when intentionally accepting the newer source revision. The merge audit records it.

This implementation follows the robust-loss and chronological-validation principles
in the [Huber regression documentation](https://scikit-learn.org/stable/modules/linear_model.html#huber-regression)
and [time-series validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).
The implementation uses NumPy; scikit-learn is not required.

## Combine the earlier exports with a new download

Use this workflow for the **August 15–October 20, 2026 inclusive** analysis. Place the
earlier exports and new exports under `local/`, preserving the original files.
Supply them in oldest-to-newest order. If the new history already includes all older
data, you can still supply both; identical overlapping records are collapsed, never summed.

```bash
python scripts/analyze_promenade.py \
  --meter local/consumption_kwh.zip local/consumption_kwh-new.zip \
  --bms local/live_bms_history.csv local/live_bms_history-new.csv \
  --start 2026-08-15 --end 2026-10-21 \
  --forecast-start 2026-10-21 --forecast-end 2027-01-21 \
  --rate 0.12 --output results/aug15-oct20
```

New meter inputs may be raw daily Consumption CSV, exporter ZIP or extracted directory.
Every directory/ZIP chunk still requires its exporter quality companion and valid checksum.
The BMS inputs must be wide history/update CSVs with offset-aware timestamps.

Conflicting meter revisions or BMS cells stop the default run. Review the reported
conflict. If the last supplied export is the authoritative revision, rerun into a new
output directory with `--overlap-policy prefer-last`. Every revision is recorded in
`merge-audit.json`; later non-null BMS cells override older ones at the same timestamp.
Missing timestamps are not interpolated. Revised invalid meter data remains invalid;
it is not silently replaced with an older valid number.

The new download must contain **complete meter readings through October 20**. The
earlier snapshot ends October 15 and alone cannot produce complete-window savings.
Incomplete periods remain unavailable while paired-period comparisons remain visible.
The confirmed control date is August 15. The baseline still conservatively excludes
July 24 onward because of uncertain earlier commissioning. Other review flags remain.

### BMS electrical response

The new workflow also learns a **separate Ridge regression** from synchronized meter
energy and BMS features. No manual calibration file or external weather file is required.
It does not change the frozen historical baseline or replace missing meter readings.

Features include equipment-specific and aggregate stage/fan/valve proxies, temperatures,
active setpoints, positive heating/cooling temperature deviations, setpoint changes,
and previous-day values. Time alignment is past-only with a bounded signal age.
An invalid status interrupts propagation; gaps remain unknown. Daily features require
90% bin coverage by default. These are sampled operation proxies, not measured runtime.

The generated `response-config.json` proposes equipment bindings from column names.
Inspect them and verify encodings, units, shared equipment and relevance. A proposed
mapping is marked `semantics_verified: false` and all resulting fits retain that gate.
Discrete roles accept only 0/1; values such as 0.5, 3 or 10 are not automatically
interpreted as discrete stages. Valve roles use 0–100, temperature/setpoint roles -20–60 C.
Adjust source preparation/mappings to the actual point definitions rather than guessing.

For a preparation-only pass, add `--prepare-only` to the command with a separate output
directory. Review its proposed mappings, then provide them on the full run:

```bash
# Use alongside the --meter / --bms arguments above:
# --response-config results/preparation/response-config.json
```

The regression selects well-covered, varying features using only an initial training
prefix, caps their number (default 24 BMS features), adds weekday schedule terms, and tunes the Ridge penalty with expanding
chronological folds. Each fold learns its own scaling. The final 20% is held out for
evaluation; the saved final model then refits all eligible data before the reporting
end. The defaults require at least 40 training days and 8 later holdout days after
feature/energy quality exclusions. Insufficient coverage produces an explicit unavailable
status. A selected feature subset does not establish every asset's separate effect.

Additional outputs:

| File | Purpose |
| --- | --- |
| `merge-audit.json` | Input hashes, identical overlaps and explicit revisions |
| `response-config.json` | Applied or proposed BMS bindings and response settings |
| `response/daily-features.csv` | Coverage-screened daily features and one-day lags |
| `response/feature-audit.json` | Rejected values, signal coverage and feature definitions |
| `response-summary.csv` | Each meter's BMS fit, penalty, held-out errors and blockers |
| `meter-.../response/heldout.csv` | Independent chronological consumption estimates |
| `meter-.../response/fit-diagnostics.csv` | In-sample diagnostics; never independent savings evidence |
| `meter-.../response/model.json` | Checksum-protected final electrical response model |

The dashboard overlays eligible held-out BMS consumption estimates in purple. The
training-period diagnostic fit is deliberately excluded from that plot. Meter minus
BMS-estimated consumption is **prediction error**, not savings. Reported kWh/CAD
differences continue to use historical baseline minus measured consumption.

### Future optimized consumption

The three-month calendar baseline forecast remains available. A BMS response model
also needs a **future operating trajectory** to predict optimized consumption: expected
temperatures, states, setpoints and other mapped signals on the same timeline. Today's
live readings alone do not specify the next three months. No last-value future is invented.

Provide that wide timestamped scenario with `--future-bms local/optimized-scenario.csv`.
It must include prior-day context for lags. The model predicts only dates after its
training end; missing and negative predictions remain unavailable, and operating-range
extrapolation is flagged. The dashboard shows the conditional optimized trajectory in red.
It is not a validated free-running forecast merely because the electrical mapping passes.

For a separate projected control-policy comparison, also provide
`--baseline-future-bms local/old-policy-scenario.csv`. Both trajectories must represent
comparable external conditions and consistent thermal/equipment dynamics. Replacing only
setpoints while retaining the optimized temperatures/runtime is not a valid old-policy
trajectory. The script does not generate or certify these scenarios.
Per-meter `response/scenario-comparison.csv` records conditional kWh/CAD differences
only when both predictions exist and remain inside fitted feature ranges. Site/BMS
review gates still apply; these differences are not realized or verified causal savings.

## Run the daily comparison and forecast

From the repository root with the environment active:

```bash
python scripts/test_promenade.py \
  --start 2026-08-15 --end 2026-10-21 \
  --forecast-start 2026-10-21 --forecast-end 2027-01-21 \
  --rate 0.12 --output results/promenade-test
```

Dates are site-local. **End dates are exclusive.** This example compares August 15–October 20
and forecasts October 21–January 20. Without forecast flags, the forecast starts on the
current site-local date and lasts three calendar months. Reporting defaults cover August 15–October 20; choose reporting dates appropriate to each new export.

The runner validates inputs, fits every meter separately, prints progress, and writes
the dashboard. Use a new output directory for each full run; existing directories are
not overwritten. Alternative input paths are accepted with `--meter` and `--bms`.
Missing readings are not filled with invented energy. All-zero, unidentifiable or
invalid meters may be blocked. Poor held-out performance is recorded as a review gate.

### Reuse existing models

To change forecast dates or the energy rate without repeating import/training:

```bash
python scripts/summarize_promenade.py \
  --results results/promenade-test \
  --forecast-start 2026-10-21 --forecast-end 2027-01-21 \
  --rate 0.12
```

This regenerates the overview, forecast files and summary CSV inside the selected
results directory. It reuses frozen model artifacts and does not retrain them with
new readings. Preserve a results copy if you need the previous forecast output.
Results from versions 0.3.1 and 0.3.2 can be used via an absolute `--results` path.

## Read the results

Download `results/promenade-test/index.html` and open it in a browser. It is a standalone
HTML file with embedded data and requires no web server. Select a meter to see:

- Reporting-period baseline and measured consumption.
- Future baseline forecast, distinguished from measured values.
- Signed complete-period kWh differences and CAD at the chosen rate.
- Validation findings and unverified virtual-total candidates.

| File | Contents |
| --- | --- |
| `overview.json` | Dashboard data, forecast window, rate and selected-boundary totals |
| `meter-summary.csv` | Per-meter forecast, kWh/CAD differences, validation and blocked reasons |
| `forecast-daily.csv` | Daily future baseline predictions by meter |
| `boundary-candidates.csv` | Existing virtual totals, kept separate and unverified |
| `site-total.json` | Total for an explicitly selected boundary, otherwise unavailable |
| `summary.json` | Original comparison run, meter folders and review gates |
| `meter-.../savings/` | Reporting-period JSON, CSV and HTML |
| `meter-.../future-forecast/` | Latest future-forecast JSON, CSV and HTML |
| `meter-import-audit.json`, `bms-audit.json` | Source provenance, import validation and BMS coverage |

Negative differences mean measured usage exceeded the predicted baseline. Complete-period
values remain unavailable when any required date is missing. Paired values refer only
to eligible matched dates; they are not complete-window savings. CAD uses the same
flat rate on both sides and excludes demand charges, taxes, fixed charges and other
bill components. Future optimized usage and future realized savings are unavailable.

### Whole-site totals

Predicting usage cannot resolve an unknown electrical boundary. The CARMA data includes
physical meters, submeters and virtual totals; summing all meters can double-count loads.
Until a non-overlapping boundary is identified, the site total remains unavailable.
The existing virtual totals are candidates, not automatically verified mall totals.

After verifying the appropriate meter or non-overlapping set, supply the actual IDs:

```bash
# Replace MAIN_METER_ID with the verified CARMA code.
python scripts/summarize_promenade.py --total-meters MAIN_METER_ID --rate 0.12
```

Multiple IDs are accepted after `--total-meters`. This is the operator's boundary
declaration, not automatic wiring verification. Missing or blocked selected meters
keep totals unavailable. Paired totals use common eligible dates across all selected
meters; complete totals require the full window for all of them. Model/site review
gates still apply to any numerical total.

## Optional optimizer and advanced CLI

```bash
option-c --help
option-c prepare-optimizer --help
option-c run-all --help
```

Canonical optimizer CSV records use `timestamp` (valid time), `issued_at`, `zone_id`
and `achieved_mode`. Confirmed mode meanings are -1 cooling, 0 thermal OFF and 1 heating.
Optional application/support flags, zone mappings and measured temperature/setpoint
bindings are configured under `optimizer` in the site config. Thermal OFF does not
imply zero fan or electrical load. Unknown application and deadband behavior remain uncertain.

The optional response regression learns from synchronized historical meter and eligible
optimizer records; no separate manual calibration file is required. It needs sufficient
coverage and independent ON/OFF variation (default 60 training days), held-out validation,
and reviewed equipment mappings. Associations are not causal zone savings. It does not
replace measured reporting consumption with an optimizer prediction.

The supplied legacy archive ending September 14 lacks the explicit timing contract
required by the importer. It overlaps part of the confirmed control period but is not a
supported direct input. Its flattened trajectories must not be converted by guessing
zone order, forecast start, issue time or cadence.

## Runpod and optional container

For the first deployment, use a Pod with Python 3.11 or 3.12, clone/extract the code,
change into its repository root, install as above and upload private files under `local/`.
A GPU is unnecessary for this regression workflow. Download `index.html` to view results.

The Dockerfile starts the separate signed-job serverless handler, not the Pod test.
It installs the optional Runpod/Excel dependencies, runs as UID 10001, and requires:

- `OPTION_C_JOB_HMAC_KEY`: a private HMAC key of at least 32 characters.
- `OPTION_C_STORE_ROOT`: a persistent directory writable by UID 10001.

`examples/sign_job.py` signs jobs in a trusted backend. Never commit keys, signed site
jobs, session captures or credentials. The handler supports site-scoped train, forecast,
savings and post-model jobs with bounded inline payloads; large archives belong in
the Pod file workflow. Signed-job tests are offline. A live Docker/serverless deployment
has not been verified by this release; provision storage/permissions and test it before use.

## Verification and known limits

Version 0.5.0 adds robust calendar baselines, common chronological validation,
reviewable meter-quality screening and withheld headlines for unreliable results.
All 78 offline tests passed locally. Tests include raw-value preservation, anchored review decisions, model selection
without holdout leakage, missing-day handling and end-to-end exports. A local Python
wheel is built for release. Docker and live Runpod infrastructure require site testing.
The existing BMS snapshot still needs sufficient synchronized, coverage-screened days;
the package does not lower validation requirements to force a response estimate.

Annual calendar terms approximate recurring seasonal patterns but do not model actual
weather or changes in heating/cooling demand; a three-month
winter forecast can be unreliable even when a numerical validation screen passes.

GitHub Actions runs the offline tests on Python 3.11 and 3.12 after push. No private
inputs, real CARMA access or BACnet writes are used by CI. Local checks and CI do not
certify an M&V plan or validate live Runpod infrastructure.
