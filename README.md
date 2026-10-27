# Promenade electricity forecasting and policy comparison — v0.9.2

Daily meter readings, timestamped BMS/optimizer observations and external weather feed separate historical savings, consumption-model validation and future policy projections. No new hardware or BACnet writes are needed. The original v0.6.2 and newer working checkout have not been overwritten.

## Mapper native observation provenance (v0.9.2)

The companion `live_bms_history.observations.npz` is auto-detected beside `live_bms_history.csv`. An alternative name can be supplied with `--bms-observation-mask local/history.observations.npz`. Currently this option requires one combined BMS CSV. Runpod project jobs accept the companion under `files.bms_observation_mask`.

The companion must come from the native H5 observation timestamps. Never manufacture it by treating all CSV blanks as absent: that would erase real missing records. It stores packed observation bits, signal columns, UTC timestamps and the exact CSV SHA-256. Modified CSVs or mismatched timestamps are rejected. Renaming the files does not change the CSV hash, but editing or re-saving the CSV does. Without a verified companion, blank cells remain conservative invalidations.

For the supplied input bundle, extract its `local/` folder into this project. It contains the unchanged combined CSV and the matching companion, both named for automatic detection. Keep your Alectra workbook and weather CSV in `local/` as before.

```bash
python -m pip install -e '.[test,excel]'
python -m pytest -q
python scripts/run_project.py \
  --meter "local/Promenade_Alectra_Daily_kWh_2026-10-26.xlsx" \
  --meter-kind utility \
  --bms local/live_bms_history.csv \
  --weather local/weather_daily.csv \
  --weather-scenario climatology \
  --start 2026-08-15 --end 2026-10-21 \
  --forecast-start 2026-10-26 --forecast-end 2027-01-26 \
  --rate 0.12 \
  --output results/promenade-v092
```

`--end` is exclusive: October 21 includes October 20. For a report including October 6, use `--end 2026-10-07`. More timestamps do not necessarily create more complete eligible days. Do not shorten the reporting period or reduce validation requirements merely to force a model fit.

## Outputs and their meanings

| Output | Calculation / meaning |
|---|---|
| Actual energy | Sum of actual daily meter kWh, with available-day and complete-period totals separate. |
| Historical baseline comparison | Pre-control meter calendar/weather baseline minus actual optimized meter consumption. |
| BMS consumption reconstruction | Meter-calibrated model fitted against eligible BMS daily features. Training reconstruction is not independent validation. |
| Meter-minus-BMS residual | Consumption-model error, not savings. Independent withheld predictions are separate. |
| Meter-history future forecast | Recent controlled meter/weather pattern continuation, with its own withheld validation and support checks. |
| Future old-policy consumption | Weather-forced thermal rollout and meter-calibrated electrical regression under documented old setpoints/schedules. |
| Future optimized-policy consumption | The same models, dates, weather and initial state, using the planned optimized policy. |
| Projected policy difference | Old-policy prediction minus optimized-policy prediction on identical conditions. Conditional projection, not verified savings. |
| Historical old-policy replay | Separate models calibrated strictly before reporting start; replay old policy against actual historical weather and compare with measured energy. Blocks when earlier calibration/initialization is inadequate. |

CAD values use the assumed flat energy rate (default CAD 0.12/kWh). They exclude demand charges, taxes, fixed charges and Global Adjustment. Models do not claim exact utility-bill savings or automatic IPMVP/ASHRAE certification. Source boundary, measurement, non-routine changes and causal uncertainty still require review.

## Date-level BMS eligibility (corrected in v0.9.2)

The primary command remains `scripts/run_project.py` (installed equivalent `option-c-project`). Missing **dates/daily model inputs**, not blank cells anywhere in a wide row, determine eligibility.

- Retain timestamped rows with sparse/unrelated signal cells. Reject only invalid/missing timestamps at initial import.
- Validate temperature/setpoint observations per signal; invalid values become unavailable for that signal, without discarding other observations in the row.
- Select a fixed asset cohort using earlier BMS availability only (never heldout data or meter targets). Automatically added assets cannot reduce the maximum earlier covered-day count. Explicit signal-config groups are respected without automatically dropping assets.
- With a verified native observation mask, derive daily predictors by aligning each required signal's actual observations within its bounded validity. Alignment-only blanks are not new observations. A genuine native blank immediately invalidates that signal. Without a mask, retain conservative behavior: every row is treated as an observation for every column.
- Observations represent bounded validity until the next observation or maximum age (default 30 minutes), with exact midnight splitting. This is a declared snapshot-validity assumption, not synthesized readings.
- Exclude a whole date when required daily features are missing or covered hours are below the configured quality threshold (default 90%). Completely absent dates are listed in the audit and never filled with zero or interpolated energy.
- Ignore blank columns not used by the fixed model cohort. Keep meter-only calculations separate from BMS-eligible date comparisons.
- Thermal identification accepts genuinely adjacent observations at irregular intervals up to the same bounded age; longer gaps and invalid endpoint values are excluded. Rollout validation does not bridge unobserved gaps.

Outputs: `bms-observations.csv`, `eligible-bms-observations.csv`, `bms-date-audit.json`, `bms-date-coverage.csv` and `asset-daily.csv` show the decisions. `rejected-bms-rows.csv` now records invalid timestamps, not harmless blank signals. `--required-bms-columns` declares additional signals required for date eligibility; it does not reinstate all-column row rejection.

With the combined October 26 Mapper export and its verified native mask, 11,561 timestamp rows occur before October 21. The corrected cohort has 13 assets and 40 eligible dates. The current Alectra input yields 31 training / 5 test matched days for PWST104274 and 31 / 0 for PWST152089. BMS regression therefore remains blocked by the unchanged project screening minimum of 40 training and 8 later test days; these counts are not universal IPMVP requirements.

Legacy helper `strict_history.clean` is retained for reproducing older behavior, but the primary workflow uses `date_history.py`.

## Install

From this project root:

```bash
python -m venv .venv
# Git Bash on Windows:
source .venv/Scripts/activate
# On Runpod/Linux use: source .venv/bin/activate
python -m pip install -e '.[test,excel]'
python -m pytest -q
python -c "import importlib.metadata; print(importlib.metadata.version('promenade-option-c'))"
```

Expected version: **0.9.2**. Reinstall after changing project directories so an older editable installation is not imported.

## First real-data run

Place your files under local/. No private input data is included in the code archive.

```bash
python scripts/run_project.py \
  --meter 'local/Promenade_Alectra_Daily_kWh_2026-10-26.xlsx' \
  --meter-kind utility \
  --bms local/live_bms_history.csv \
  --weather local/weather_daily.csv \
  --weather-scenario climatology \
  --start 2026-08-15 --end 2026-10-21 \
  --forecast-start 2026-10-26 --forecast-end 2027-01-26 \
  --rate 0.12 --output results/promenade-project-first
```

Change dates to your actual available data. Reporting and forecast end dates are exclusive. The future start must not precede reporting end. Choose a new output directory for every run.

This first run generates actual-cohort `old_policy-request.json`, `optimized_policy-request.json` and `initial-state-request.json` with missing values deliberately left unfilled. It also generates the date eligibility audit and any available meter-only outputs. BMS/policy outputs with missing prerequisites have explicit blockers, not invented values.

The supplied October 26 Alectra workbook has 63/67 reporting readings for PWST104274 and 53/67 for PWST152089. Three low/completeness-unverified readings are retained with review flags. The BMS export is now processed by date using its required signals; blank unrelated columns do not discard useful dates. Missing meter dates and source-quality review flags remain unresolved.

## Supply actual policies for the full comparison

Fill COPIES of the generated policy requests and place them in local/old_policy.json and local/optimized_policy.json. Document the reference and whether the assumptions have been reviewed. Do not guess old occupied hours or equipment margins from optimizer training settings.

For every fitted asset provide:

- `weekdays`: occupied days, Monday=0 through Sunday=6.
- `occupied_start` / `occupied_end`: local HH:MM. Equal times mean 24 hours on listed days; overnight schedules use the preceding weekday after midnight.
- `occupied_cooling_c`, `unoccupied_cooling_c`, `occupied_heating_c`, `unoccupied_heating_c`: numeric Celsius or the explicit string `OFF`. Null is not accepted as OFF.
- `control_margin_c`: explicit nonnegative threshold margin. Cooling effective threshold is cooling setpoint plus margin; heating effective threshold is heating setpoint minus margin. This is an explicit continuous-demand surrogate assumption, not an identified equipment hysteresis cycle.

Then rerun with a new output path and add:

```bash
--old-policy local/old_policy.json \
--optimized-policy local/optimized_policy.json
```

If the last usable BMS observation is within 30 minutes before forecast start and the internal thermal models pass validation, initialization uses one step over the actual elapsed time. A longer gap is not held constant: supply `--initial-state local/initial-state.json`, with the exact forecast-start timestamp, source reference and all asset temperatures. Do not invent those temperatures. `--historical-initial-state` serves the independent historical replay when its start state is unavailable.

## External weather

```bash
python scripts/download_weather.py \
  --start 2023-01-01 --end 2026-10-21 \
  --output local/weather_daily.csv
```

Open-Meteo download dates are inclusive; regression window ends are exclusive. Defaults are the supplied Promenade coordinates, America/Toronto, Celsius daily mean and ECMWF IFS historical weather. Change location/timezone for other sites. Original API response/query/hashes are cached. Provider terms apply.

CSV format: `date,temperature_c` (or `outdoor_temp_c`), one local daily mean per date. No duplicates, invalid units, missing values or silent gaps. Weather for historical fitting must cover historical observations and reporting days.

Future options:

1. `--future-weather local/future_weather.csv --future-weather-basis 'Provider/model issue time or explicit scenario reference'`.
2. Explicit `--weather-scenario climatology`: mean of past observations within +/-7 calendar days, at least two historical years; all source dates strictly precede forecast start.

Climatology is a weather scenario, **not a three-month weather forecast**. No future weather is silently inferred when neither option is chosen. Daily mean outdoor temperature is treated as constant intraday forcing in the empirical thermal model; high-frequency weather variation is not measured by this input.

## Model design and validation

Historical baseline retains the robust OLS/Huber/calendar/weather candidate engine and its review gates. Source readings/spikes are not automatically removed.

The new meter/electrical model fits nonnegative Ridge coefficients against daily meter energy, using chronological inner penalty selection and a separate final 14-calendar-day holdout with at least eight complete observations. The BMS model requires at least 40 earlier complete days. Its predictors are weekday/weekend background hours, outdoor heating/cooling degree-hours and pooled asset cooling/heating degree-hours and thermal-call proxy hours. No individual asset kW allocation is claimed. Final refit reconstruction and frozen withheld predictions are distinct.

Thermal model per asset:

`dT/dt = b0 + a*(outdoor-T) - c*max(T-cooling_SP,0) + h*max(heating_SP-T,0) + daily sine/cosine terms`

The heat-transfer/cooling/heating coefficients have sign/bound constraints. Earlier folds select regularization; a later holdout tests free-running local-day trajectories without assimilating later measured temperatures. Default project screen: at least two complete day rollouts, RMSE <=1.5 C and no worse than persistence. This is not a standards-prescribed universal threshold. Unidentifiable or failed asset models block internal policy forecasts; no constant-temperature fallback is used.

Three-month projections exceed the one-day thermal validation horizon and retain that warning. Joint thermal/electrical/weather/causal uncertainty is not fully quantified. Forecasts remain conditional engineering projections. Numerical passes do not establish meter boundaries or causal savings. Source-quality and support warnings remain visible.

## Neural ODE / STGNN adapter

A future temperature trajectory can replace the internal thermal rollout:

`timestamp,asset_id,predicted_zone_temperature_c`

Supply it using `--old-trajectory` or `--optimized-trajectory`, plus the corresponding `--old-trajectory-evidence` / `--optimized-trajectory-evidence` JSON. Evidence must contain `model_id`, `validation_reference`, `policy_sha256` and `conditions_sha256`. Per-meter trajectory request files contain the required hashes, cadence, exact asset cohort, dates and common initial state. No missing slots, duplicate times, changed policy/weather binding or mismatched initial temperature is accepted. Validation references are retained as evidence requiring review, not automatically trusted certification.

This adapter accepts outputs from your models; it does not train or execute an unavailable latest Mapper/STGNN repository.

## Where to inspect results

Open **project.html** for the end-to-end workflow and **index.html** for the historical baseline report.

- project-summary.json / project-meter-summary.csv: separate forecasts, model residuals, projected differences and blockers.
- bms-date-audit.json / bms-date-coverage.csv: date eligibility; rejected-bms-rows.csv: invalid timestamps.
- bms-date-coverage.csv / asset-daily.csv: actual accepted slots and daily features.
- thermal-model.json: dynamics, heldout rollout results and failed assets.
- Per-meter bms-power-model.json / bms-consumption-diagnostics.csv: electrical coefficients, reconstructed and withheld kWh.
- Per-meter meter-weather-forecast.json: recent meter-pattern weather-conditioned forecast.
- Per-meter old_policy-forecast.json / optimized_policy-forecast.json and trajectories: policy-specific projections.
- projected-policy-difference.json: only comparable policy conditions; projected kWh/CAD, not exact savings.
- historical-old-policy-replay.json: independent pre-reporting calibration, if available.
- Per-meter savings/result.json: historical baseline-minus-actual, with partial/full totals and source/model review gates.

Separate meters and overlapping virtual totals are not automatically summed. Current values remain diagnostic when validation, coverage or source review fails. The prior label suggesting every baseline was calendar-only has been corrected.

## Reproducible complete demonstration

```bash
python examples/demo_project.py --output results/synthetic-project
```

Creates 240 days of complete synthetic BMS/meter/weather observations, old/optimized policies and a supplied three-day weather scenario, then runs every principal forecasting/replay path. All resulting values are **synthetic demonstration only**, not Promenade performance.

## Runpod deployment

For a Pod, upload/extract this project, mount private inputs, install the extras and run the commands above. CPU is sufficient for these regressions. GPU hardware is not required unless separately running a Neural ODE/STGNN model.

The Dockerfile includes the package, scripts and writable local/results folders. It starts the serverless handler; the image itself has not been built/deployed during this implementation.

Serverless supports the signed `run_project` action in examples/project-job.json. Put filenames under the persistent `OPTION_C_STORE_ROOT/<site_id>/inputs/` directory. The config site ID must match the signed payload. Only simple site-scoped filenames are accepted; no arbitrary shell commands, paths or downloads. Output is stored under `<site_id>/project-results/<request_id>/`. Results are private mounted artifacts; no public URLs are generated.

Set `OPTION_C_JOB_HMAC_KEY` (at least 32 characters) and persistent `OPTION_C_STORE_ROOT`; sign envelopes with examples/sign_job.py in a trusted backend. Keep secrets and signed jobs private. Service assumptions remain single-worker/shared-secret; a repeated request ID with changed content is rejected. Use a new request ID after an interrupted partial project run.

## Implementation map

- date_history.py: sparse observations, fixed cohort, observed intervals and daily eligibility.
- strict_history.py: simulated complete trajectory aggregation and legacy row helper.
- power_model.py: weather-aware meter and pooled BMS/electrical regressions.
- thermal_model.py: weather-forced thermal identification and free-running validation.
- policy_forecast.py: schedules, common initial conditions, policy rollouts and comparisons.
- weather_scenario.py: explicit historical climatology scenario.
- project_cli.py / scripts/run_project.py: full orchestration and report.
- project_job.py / service.py: signed Runpod project jobs.
- utility.py / ingestion.py / weather.py: Alectra/CARMA input handling and external-weather validation.
- resilient.py / workflow.py / overview.py: historical baseline, avoided-energy comparison and legacy report.
